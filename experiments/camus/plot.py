"""Reproduce five paired CAMUS ED examples with the fixed appearance filter.

Downloads five recordings from CAMUS-beamspace on Hugging Face, without the EchoXFlow library.
Geometry comes from the package's image-only fan fit; labels are not used.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from image_enhancement import load_filter
from image_enhancement.scan_conversion import Grid,SectorGeometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_sources(cfg):
    """Download five pinned exam archives and extract only the selected recordings."""
    import tarfile
    from huggingface_hub import hf_hub_download
    root = Path(cfg.get('cache_dir', 'outputs/camus-source')) / cfg['revision']
    paths = []
    for selected in cfg['recordings']:
        relative = f"exams/exam_{selected['exam_id']}/recording_{selected['recording_id']}.zarr"
        archive = hf_hub_download(cfg['repo_id'], f"exams/exam_{selected['exam_id']}.tar",
                                  repo_type='dataset', revision=cfg['revision'])
        with tarfile.open(archive) as tar:
            count = 0
            for member in tar.getmembers():
                if not member.name.startswith(relative + '/'):
                    continue
                parts = Path(member.name).parts
                if member.isdir():
                    continue
                if Path(member.name).is_absolute() or '..' in parts or not member.isfile():
                    raise ValueError('Unexpected archive member in selected recording')
                target = root / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open('wb') as output:
                    output.write(source.read())
                count += 1
        if count == 0:
            raise ValueError(f'Selected recording is absent from {archive}')
        paths.append((selected, root / relative))
    return paths


def run(cfg):
    import zarr
    from scipy.ndimage import map_coordinates
    torch.set_num_threads(2)
    out = Path(cfg['output_dir'])
    chosen = cfg['recordings']
    if len(chosen)!=5 or len({r['recording_id'] for r in chosen})!=5:
        raise ValueError('Use five distinct preselected recordings')
    model=load_filter(cfg.get('checkpoint','weights/appearance.pt'))
    out.mkdir(parents=True,exist_ok=True)
    report={'configuration':cfg,'checkpoint_sha256':sha(cfg.get('checkpoint','weights/appearance.pt')),
            'device':'cpu','geometry_uses_annotations':False,'display_range':[0,1],'display_top_crop_fraction':0.10,'records':[]}
    panels=[];patients=set()
    with torch.inference_mode():
        for selected, source in download_sources(cfg):
            rec = selected['recording_id']
            attrs=json.loads((source/'.zattrs').read_text())
            manifest=attrs['recording_manifest'];camus=manifest['metadata']['source']
            if camus['official_split']!='test' or attrs['exam_id'] in patients:
                raise ValueError('Use distinct official-test patients')
            patients.add(attrs['exam_id'])
            sector=next(s for s in manifest['sectors'] if s['semantic_id']=='bmode')
            frame=np.asarray(zarr.open(str(source),mode='r')[sector['frames']['zarr_path']][camus['ed_frame_index']])
            if frame.dtype!=np.uint8 or frame.ndim!=2:raise ValueError('Expected native uint8 polar image')
            fan=camus['sector_px'];pitch=camus['source_pixel_spacing_mm']/1000
            radius=fan['outer_radius'];left=fan['left_angle_rad'];right=fan['right_angle_rad']
            shape=(math.ceil(2*radius)+1,math.ceil(2*radius*(right-left))+1)
            stored=sector['geometry'];ns,nb=frame.shape
            target_r=torch.linspace(0.,radius*pitch,shape[0])
            target_a=torch.linspace(left,right,shape[1])
            rr,aa=torch.meshgrid(target_r,target_a,indexing='ij')
            stored_left=stored['tilt_rad']-stored['width_rad']/2
            row=(rr-stored['depth_start_m'])/(stored['depth_end_m']-stored['depth_start_m'])*(ns-1)
            col=(aa-stored_left)/stored['width_rad']*(nb-1)
            coords=torch.stack((2*col.clamp(0,nb-1)/(nb-1)-1,2*row.clamp(0,ns-1)/(ns-1)-1),-1)[None]
            inside=(rr>=stored['depth_start_m'])&(rr<=stored['depth_end_m'])&(aa>=stored_left)&(aa<=stored_left+stored['width_rad'])
            polar=F.grid_sample(torch.from_numpy(frame.copy())[None,None].float()/255,coords,
                                mode='bilinear',padding_mode='border',align_corners=True)[0]
            polar=torch.where(inside,polar,0.)
            h,w=camus['source_image_shape']
            geometry=SectorGeometry(0.,radius*pitch,(left+right)/2,right-left)
            grid=Grid(h,w,(-fan['apex_column']-.5)*pitch,(-fan['apex_row']-.5)*pitch,pitch,pitch)
            pair=model.compare(polar,geometry=geometry,grid=grid,sampling='endpoints')
            plain,enhanced,mask=[v[0].numpy() for v in (pair.unfiltered,pair.filtered,pair.support)]
            yy,xx=np.meshgrid(np.arange(h)-fan['apex_row'],np.arange(w)-fan['apex_column'],indexing='ij')
            source_row=np.hypot(xx,yy)/radius*(shape[0]-1)
            source_col=(np.arctan2(xx,yy)-left)/(right-left)*(shape[1]-1)
            independent=map_coordinates(polar[0].numpy(),[source_row,source_col],order=1,mode='nearest')
            identity_error=float(np.max(np.abs(np.where(mask,independent,0.)-plain)))
            if identity_error>2e-4:raise ValueError(f'Disabled-path discrepancy: {identity_error}')
            np.savez_compressed(out/f'{rec}.npz',without=plain,with_cnn=enhanced,support=mask)
            report['records'].append({'recording_id':rec,'view':camus['view'],'phase':'ED',
                'frame_index':camus['ed_frame_index'],'source_metadata_sha256':sha(source/'.zattrs'),
                'source_frame_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),
                'disabled_path_max_abs_error':identity_error})
            panels.append((plain,enhanced,mask,camus['view']))
    if [p[3] for p in panels]!=['A2C','A2C','A4C','A4C','A4C']:raise ValueError('Use A2C/A2C/A4C/A4C/A4C order')
    render(panels,out,width=cfg.get('width_in',6.87),font_pt=8,compact=True)
    (out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Wrote five paired examples to {out}')

def render(panels, out, *, width=6.87, font_pt=8, stem='fig_camus_examples', compact=False):
    """Tight two-row gallery, support crop and display-only 10% top crop."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'cmr10','mathtext.fontset':'cm','font.size':font_pt,
                         'image.composite_image':False,'axes.formatter.use_mathtext':True})
    cropped=[]
    for plain,enhanced,support,view in panels:
        ys,xs=np.nonzero(support)
        top=ys.min()+round(.10*(ys.max()-ys.min()+1))
        crop=np.s_[top:ys.max()+1,xs.min():xs.max()+1]
        cropped.append((plain[crop],enhanced[crop],support[crop],view))
    left,right,gap,row_gap,margin=.18,.01,.015,0.,.025
    panel_w=(width-left-right-gap*(len(cropped)-1))/len(cropped)
    heights=[panel_w*p[0].shape[0]/p[0].shape[1] for p in cropped]
    row_h=max(heights)
    # The rotated label can be taller than a row in the narrower paper layout.
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    label_height=TextPath((0,0),'Beamspace CNN',size=font_pt,
                          prop=FontProperties(family='cmr10')).get_extents().width/72
    margin=max(margin,(label_height-row_h)/2+.04)
    height=2*row_h+row_gap+2*margin
    fig=plt.figure(figsize=(width,height),facecolor='none')
    for col,((plain,enhanced,support,_),ih) in enumerate(zip(cropped,heights)):
        for row,frame in enumerate((plain,enhanced)):
            x=left+col*(panel_w+gap)
            # Pair boundaries meet even when the five source aspect ratios differ.
            boundary=margin+row_h
            y=boundary+row_gap if row==0 else boundary-ih
            ax=fig.add_axes([x/width,y/height,panel_w/width,ih/height],facecolor='none')
            rgba=np.concatenate((np.repeat(frame[...,None],3,-1),support[...,None].astype(float)),axis=-1)
            ax.imshow(rgba,aspect='equal',interpolation='none',rasterized=False,clip_on=False)
            ax.set_axis_off()
    for row,label in enumerate(('Input','Beamspace CNN')):
        y=height-margin-row*(row_h+row_gap)-row_h/2
        fig.text((left/2-.01)/width,y/height,label,rotation=90,ha='center',va='center')
    for suffix in ('pdf','png'):
        fig.savefig(out/f'{stem}.{suffix}',dpi=220,transparent=True)
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('experiments/camus/config.example.json'))
    run(json.loads(parser.parse_args().config.read_text()))
