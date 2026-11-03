"""Prepare/run matched unfiltered->unfiltered and filtered->filtered segmentation.

Prepare reads metadata/configuration only. Run-cell is the explicit training action.
"""
import argparse,csv,copy,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import yaml
from retrain_task3 import sha,write_json,train_cell

ARMS=('unfiltered','filtered')


def camus_splits(exams):
    """Stable patient folds; both views of each examination stay together."""
    ids=sorted(set(exams),key=lambda e:hashlib.sha256(f'camus-cv-20261027:{e}'.encode()).hexdigest())
    folds={e:i%5 for i,e in enumerate(ids)}
    split={}
    for fold in range(5):
        pool=[e for e in ids if folds[e]!=fold]
        selection=sorted(pool,key=lambda e:hashlib.sha256(f'camus-selection:{fold}:{e}'.encode()).hexdigest())[:math.ceil(.125*len(pool))]
        split[str(fold)]={'train':sorted(set(pool)-set(selection)),'selection':sorted(selection),
                          'validation':sorted(e for e in ids if folds[e]==fold)}
    return folds,split


def official_camus_splits(rows, exams):
    """CAMUS validation selects checkpoints; official test is never used to select."""
    by_exam={}
    for row in rows:
        e=row['exam_id'];patient=row['source_patient_id'];name=row['split']
        if e in by_exam or hashlib.sha256(f'camus:exam:{patient}'.encode()).hexdigest()[:24]!=e:
            raise ValueError('Duplicate or inconsistent CAMUS patient identity')
        by_exam[e]=name
    if set(by_exam)!=set(exams):raise ValueError('Official split and catalog differ')
    groups={name:sorted(e for e,n in by_exam.items() if n==name) for name in ('train','validation','test')}
    if [len(groups[n]) for n in ('train','validation','test')]!=[400,50,50]:raise ValueError('Expected official 400/50/50 split')
    return {e:0 for e in exams},{'0':{'train':groups['train'],'selection':groups['validation'],'validation':groups['test']}},by_exam


def prepare(args):
    cfg=json.loads(args.source.read_text());base=json.loads(Path(cfg['echoxflow_manifest']).read_text())
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    seed_values=args.seeds
    if not seed_values or len(set(seed_values))!=len(seed_values) or set(seed_values)-set(base['protocol']['seeds']):raise ValueError('Seeds must be unique and present in the prepared EchoXFlow protocol')
    croissant=Path(cfg['camus_croissant'])
    c=json.loads(croissant.read_text());rows=next(r['data'] for r in c['recordSet'] if r['name']=='recordings')
    exams={r['recordings/exam_id'] for r in rows}
    official_path=Path(cfg['camus_splits'])
    with official_path.open(newline='') as stream:
        folds,splits,official_assignment=official_camus_splits(list(csv.DictReader(stream)),exams)
    if len(exams)!=500 or len(rows)!=1000:raise ValueError('Expected the pinned complete 500-patient CAMUS release')
    if any(sum(r['recordings/exam_id']==e for r in rows)!=2 for e in exams):raise ValueError('CAMUS patients must have both views')
    protocol={**base['protocol'],'seeds':seed_values,'training_input':'matched_to_condition',
              'evaluation_inputs':['same_as_training']}
    sources={'EchoXFlow':copy.deepcopy(base['source']),'CAMUS':{
        **copy.deepcopy(base['source']),'repo_id':cfg['camus_repo_id'],'revision':cfg['camus_revision'],
        'cache_dir':str(out/'camus-source'),'data_root':str(out/'camus-source'/cfg['camus_revision'])}}
    plan={'study':'matched','seeds':seed_values,'conditions':list(ARMS),'datasets':{},
          'old_filtered_scores_reusable':False,'new_training_models':12*len(seed_values),
          'echo_unfiltered_reuse_candidates':5*len(seed_values),'new_models_if_reuse_verified':7*len(seed_values),
          'camus_croissant_sha256':sha(croissant),'camus_split_sha256':sha(official_path),'scripts_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
    for dataset in ('EchoXFlow','CAMUS'):
        root=out/dataset;source=sources[dataset];source['tensor_cache']=str(root/'preprocessed')
        manifest={**copy.deepcopy(base),'study':'matched','dataset':dataset,'source':source,'protocol':protocol,
                  'script_sha256':sha(Path(__file__).with_name('retrain_task3.py')),'cells':[]}
        if dataset=='CAMUS':
            source_root=Path(source['data_root']);source_root.mkdir(parents=True,exist_ok=True)
            (source_root/'croissant.json').write_bytes(croissant.read_bytes())
            manifest['protocol']={**copy.deepcopy(protocol),'folds':[0],'split_design':'official_400_50_50','inner_selection':'official_validation','test_role':'validation'}
            manifest['fold_splits']=splits
            manifest['official_split_assignment']=official_assignment
            manifest['fixed_input_sha256']['official_split_file']=sha(official_path)
            source['official_split_file']=str(official_path.resolve())
            manifest['cohort']={e:{'fold':f,'patient_group':e} for e,f in folds.items()}
            manifest['excluded_unknown_exams']=[]
            manifest['fixed_input_sha256']={'enhancer_checkpoint':sha(source['enhancer_checkpoint']),'official_split_file':sha(official_path)}
        manifest['adapter_sha256']=sha(Path(__file__).with_name('matched_data.py'))
        manifest['study_code_sha256']={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
        manifest['camus_downloader_sha256']=sha(Path(__file__).parents[1]/'camus/plot.py')
        for fold in manifest['protocol']['folds']:
            for seed in seed_values:
                original=next(c for c in base['cells'] if c['fold']==fold and c['seed']==seed)
                for condition in ARMS:
                    cell=root/condition/f'fold={fold}'/f'seed={seed}'
                    if cell.exists():raise ValueError(f'Refusing to overwrite a prepared cell: {cell}')
                    cell.mkdir(parents=True)
                    config=yaml.safe_load((Path(original['path'])/'config.yaml').read_text())
                    config['data'].update(root_dir=source['data_root'])
                    if dataset=='CAMUS':
                        config['model']['out_channels']=4;config['data']['target_mask_channels']=3
                        config['data']['train_sample_selection']='all';config['data']['val_sample_selection']='all'
                    config['artifacts'].update(root_dir=str(cell.parent),run_name=cell.name)
                    (cell/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
                    manifest['cells'].append({'fold':fold,'seed':seed,'condition':condition,'path':str(cell),
                                              'config_sha256':sha(cell/'config.yaml')})
        write_json(root/'manifest.json',manifest);plan['datasets'][dataset]=str(root/'manifest.json')
    estimate=base['estimated_training_gpu_hours']
    scale=len(seed_values)/len(base['protocol']['seeds'])
    plan['estimate']={'new_EchoXFlow_filtered_gpu_hours':{k:estimate[k]*scale for k in ('minimum_epochs','maximum_epochs')},
        'CAMUS':'Unmeasured; same architecture, but different recording counts/lengths. Requires bounded timing pilot.',
        'rough_new_arms_gpu_hours_if_same_epoch_cost':{k:1.4*estimate[k]*scale for k in ('minimum_epochs','maximum_epochs')},
        'excludes':'Preprocessing, downloads, evaluation and scheduling. Historical GPU speed may differ.'}
    write_json(out/'plan.json',plan);print(json.dumps(plan,indent=2))


def run(args):
    manifest=json.loads(args.manifest.read_text())
    if manifest.get('study')!='matched':raise ValueError('Expected a matched study manifest')
    if manifest['adapter_sha256']!=sha(Path(__file__).with_name('matched_data.py')):raise ValueError('Prepared adapter changed')
    for name,expected in manifest['study_code_sha256'].items():
        if sha(Path(__file__).with_name(name))!=expected:raise ValueError(f'Prepared study code changed: {name}')
    if manifest['camus_downloader_sha256']!=sha(Path(__file__).parents[1]/'camus/plot.py'):raise ValueError('Prepared CAMUS downloader changed')
    sys.path.insert(0,str(Path(__file__).parents[1]/'camus'))
    train_cell(args)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--source',type=Path,required=True)
    prep.add_argument('--out',type=Path,default=Path('outputs/matched-segmentation'))
    prep.add_argument('--seeds',type=int,nargs='+',default=[0]);prep.set_defaults(func=prepare)
    cell=sub.add_parser('run-cell');cell.add_argument('--manifest',type=Path,required=True)
    cell.add_argument('--fold',type=int,required=True);cell.add_argument('--seed',type=int,required=True)
    cell.add_argument('--condition',choices=ARMS,required=True)
    cell.add_argument('--smoke',action='store_true',help='Two training batches and one selection recording only')
    cell.set_defaults(func=run)
    args=p.parse_args();args.func(args)
