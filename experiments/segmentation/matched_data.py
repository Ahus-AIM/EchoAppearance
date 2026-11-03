"""Matched preprocessing adapters. No target-dependent geometry or filter fitting."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import torch
import torch.nn.functional as F


def normalize(x):
    lo=x.amin((-2,-1),keepdim=True);span=x.amax((-2,-1),keepdim=True)-lo
    return torch.where(span>=1e-6,(x-lo)/span.clamp_min(1e-6),0.).clamp(0,1)


def preprocess(native,geometry,grid,checkpoint,condition,device='cpu'):
    """Native CNN -> Cartesian sampling -> display head -> resize -> normalization."""
    from image_enhancement import load_filter
    from tasks.utils.device_sampling import cartesian_sampling_grid,sample_stack
    x=torch.as_tensor(np.array(native,copy=True))
    if x.ndim!=3 or not torch.isfinite(x).all() or x.min()<0 or x.max()>255 or not torch.equal(x,x.to(torch.uint8).to(x.dtype)):
        raise ValueError('Expected native 8-bit display codes [T,depth,beam]')
    device=torch.device(device);x=x.to(device)
    coords,support=cartesian_sampling_grid(geometry,grid,source_shape=tuple(x.shape[-2:]),device=device)
    model=load_filter(checkpoint).to(device) if condition=='filtered' else None
    output=[]
    with torch.inference_mode():
        for frames in x.split(1):
            z=frames.float()/255
            if model is None:z=sample_stack(z,coords,support)
            else:
                z=model(z)[:,None]
                z=F.grid_sample(z,coords.expand(len(z),-1,-1,-1),padding_mode='border',align_corners=True)
                z=model.display_head(z)[:,0].clamp(0,1)
                z=torch.where(support[None],z,0.)
            if z.shape[-2:]!=(256,256):z=F.interpolate(z[:,None],(256,256),mode='bilinear',align_corners=False)[:,0]
            output.append(normalize(z))
    return torch.cat(output).cpu().numpy()


def echoxflow_class(checkpoint,condition,cache_root):
    from tasks.segmentation.dataset import RawDataset
    class MatchedEchoXFlow(RawDataset):
        preprocess_device="cpu"
        def _prepare_frames(self,frames,*,panel):
            base,geometry,grid,valid=super()._prepare_frames(frames,panel=panel)
            if condition=='unfiltered':return base,geometry,grid,valid
            if panel.geometry.get('sample_type')!='Tissue(8)':raise ValueError('Expected Tissue(8) source metadata')
            # Content-addressed, bounded-memory cache shared across folds/seeds.
            key=hashlib.sha256(frames.tobytes()+repr((geometry,grid)).encode()+Path(checkpoint).read_bytes()+Path(__file__).read_bytes()).hexdigest()
            path=Path(cache_root)/f'{key}.npy';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():processed=np.load(path,allow_pickle=False)
            else:
                identity=preprocess(frames,geometry,grid,checkpoint,'unfiltered',self.preprocess_device)
                if np.max(np.abs(identity-base))>2e-4:raise ValueError('Disabled filter changes baseline pixels')
                processed=preprocess(frames,geometry,grid,checkpoint,condition,self.preprocess_device)
                tmp=path.with_name(f'{key}.{os.getpid()}.tmp')
                with tmp.open('wb') as stream:np.save(stream,processed,allow_pickle=False)
                tmp.replace(path)
            if processed.shape!=base.shape:raise ValueError('Cached preprocessing shape mismatch')
            return processed,geometry,grid,valid
    return MatchedEchoXFlow


class CamusDataset(torch.utils.data.Dataset):
    """32 evenly spaced context frames; each source frame is scored once in validation.

    Published half-sequence masks are used as provided, without claiming that every
    intermediate frame was independently manually annotated.
    """
    def __init__(self,records,*,root,checkpoint,condition,cache_root,training=False):
        self.records=list(records);self.root=Path(root);self.checkpoint=checkpoint
        self.condition=condition;self.cache_root=Path(cache_root);self.training=training;self.preprocess_device="cpu"
        self.sample_indices_by_case={r.recording_id:[i] for i,r in enumerate(self.records)}
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        import zarr
        if int(zarr.__version__.split(".")[0]) != 2:
            raise RuntimeError("CAMUS adapter requires zarr>=2.18,<3 in the task environment")
        from echoxflow.scan import CartesianGrid,CartesianPixelGrid,rasterize_polygon_pixels,sector_geometry_from_mapping
        from tasks.segmentation.dataset import _target_masks_from_contour_points
        from tasks.segmentation.types import Sample
        from tasks.utils.dataset import cartesian_image_sample_grid,cartesian_sector_mask,resize_stack
        rec=self.records[index];path=self.root/rec.zarr_path
        attrs=json.loads((path/'.zattrs').read_text());manifest=attrs['recording_manifest']
        source=manifest['metadata']['source']
        expected_exam=hashlib.sha256(f"camus:exam:{source['patient_id']}".encode()).hexdigest()[:24]
        if expected_exam!=rec.exam_id:raise ValueError('CAMUS examination does not match its source patient')
        sector=next(s for s in manifest['sectors'] if s['semantic_id']=='bmode')
        payload_signature=[(str(p.relative_to(path)),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(path.rglob('*')) if p.is_file()]
        key=hashlib.sha256(json.dumps(payload_signature).encode()+(path/'.zattrs').read_bytes()+Path(self.checkpoint).read_bytes()+Path(__file__).read_bytes()).hexdigest()
        cache=self.cache_root/f'{rec.recording_id}-{self.condition}-{key}.pt'
        if cache.exists():item=torch.load(cache,weights_only=True)
        else:
            z=zarr.open(str(path),mode='r');native=np.asarray(z[sector['frames']['zarr_path']])
            geometry=sector_geometry_from_mapping(sector['geometry']);grid=CartesianGrid.from_sector_height(geometry,256)
            times=np.asarray(z[sector['timestamps']['zarr_path']])
            for name in ('lv_contour','la_contour'):
                if not np.array_equal(times,np.asarray(z[f'timestamps/{name}'])):raise ValueError('Unaligned CAMUS labels')
            lv,valid=_target_masks_from_contour_points(contour_points=np.asarray(z['annotations/lv_contour']),geometry=geometry,
                content_type='2d_left_ventricular_strain',role_id='lv',image_shape=grid.shape,source_image_shape=native.shape[-2:],
                target_mask_channels=2,base_valid_mask=np.ones(len(native)),output_grid=grid)
            if not valid.all():raise ValueError('Invalid CAMUS LV labels')
            points=np.asarray(z['annotations/la_contour']);pixel=CartesianPixelGrid(grid,shape=grid.shape)
            la=np.stack([rasterize_polygon_pixels(pixel.physical_to_pixel_xy(p),image_shape=grid.shape) for p in points])
            # Different structures share polygon boundaries. Preserve LV and assign remaining LA pixels.
            la=la&~(lv>0).any(axis=0)
            masks=np.concatenate([lv,la[None]],axis=0).astype(np.float32)
            if not (masks.sum(axis=(-2,-1))>0).all():raise ValueError('Empty CAMUS class')
            support=cartesian_sector_mask(geometry,grid,shape=grid.shape).astype(np.float32)
            target=F.interpolate(torch.from_numpy(masks),size=(256,256),mode='nearest')
            region=resize_stack(support[None],(256,256))[0]
            item={'frames':torch.from_numpy(preprocess(native,geometry,grid,self.checkpoint,self.condition,self.preprocess_device)),
                  'target':target,'region':torch.from_numpy(region),'metric':torch.from_numpy(masks),
                  'support':torch.from_numpy(support),'sample_grid':torch.from_numpy(cartesian_image_sample_grid(grid))}
            self.cache_root.mkdir(parents=True,exist_ok=True)
            tmp=cache.with_name(f'.{cache.name}.{os.getpid()}.tmp');torch.save(item,tmp);tmp.replace(cache)
        n=len(item['frames'])
        # Training input has the original T=32. Validation keeps original cadence/count;
        # the external benchmark handles model temporal padding/cropping.
        ids=torch.linspace(0,n-1,32).round().long() if self.training else torch.arange(n)
        frames=item['frames'][ids];target=item['target'][:,ids];region=item['region']
        valid=region[None,None].expand(1,len(ids),-1,-1)
        metric=item['metric'][:,ids];metric_valid=item['support'][None,None].expand(3,len(ids),-1,-1)
        return Sample(frames=frames[None,None],target_masks=target[None],valid_mask=valid[None],
            target_mask_valid=valid.expand(3,-1,-1,-1)[None],record=rec,sample_id=f'{rec.recording_id}:camus',
            coordinate_space='cartesian',data_root=self.root,clip_start=0,clip_stop=n,
            cartesian_metric_target_masks=metric[None],cartesian_metric_valid_mask=metric_valid[None],
            cartesian_metric_sample_grid=item['sample_grid'][None])


def camus_loaders(manifest,cell,config,smoke=False,final=False):
    from echoxflow import load_croissant
    from tasks.utils.dataset import build_train_val_dataloaders
    from plot import download_sources
    source=manifest['source'];splits=manifest['fold_splits'][str(cell['fold'])]
    records=load_croissant(root=source['data_root']).recordings
    partition={k:[r for r in records if r.exam_id in ids] for k,ids in splits.items()}
    if smoke:partition={k:v[:2 if k=='train' else 1] for k,v in partition.items()}
    if not final:partition['validation']=[]
    wanted={r.recording_id:r for rows in partition.values() for r in rows}
    # Explicit run-cell only; prepare never downloads imaging data.
    download_sources({'repo_id':source['repo_id'],'revision':source['revision'],'cache_dir':source['cache_dir'],
                      'recordings':[{'exam_id':r.exam_id,'recording_id':r.recording_id} for r in wanted.values()
                                    if not (Path(source['data_root'])/r.zarr_path/'.zattrs').exists()]})
    ds={k:CamusDataset(rows,root=source['data_root'],checkpoint=source['enhancer_checkpoint'],condition=cell['condition'],
                       cache_root=source['tensor_cache'],training=k=='train') for k,rows in partition.items()}
    return build_train_val_dataloaders(task_name='segmentation',root=source['data_root'],records=list(wanted.values()),
        train_records=partition['train'],val_records=partition['validation'],selection_records=partition['selection'],
        train_dataset=ds['train'],val_dataset=ds['validation'],selection_dataset=ds['selection'],data_config=config.data,
        seed=cell['seed'],train_batch_size=config.data['train_batch_size'],val_batch_size=config.data['val_batch_size'])


def warm_caches(loaders):
    """Prepare immutable appearance inputs on the same GPU before training.

    Workers subsequently read CPU tensors; no CUDA work happens in forked workers.
    All model initialization/training RNG is reset by the external training core.
    """
    for loader in loaders:
        ds=loader.dataset
        if not hasattr(ds,'preprocess_device'):continue
        ds.preprocess_device='cuda:0'
        try:
            for i in range(len(ds)):
                sample=ds[i]
                if not torch.isfinite(sample.frames).all():raise ValueError('Non-finite prepared input')
                if i%100==0 or i+1==len(ds):print(f'Appearance cache: {i+1}/{len(ds)}',flush=True)
                del sample
        finally:ds.preprocess_device='cpu'
    torch.cuda.empty_cache()
