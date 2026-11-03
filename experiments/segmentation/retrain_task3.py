"""Reproduce repeated five-fold segmentation training and matched appearance evaluation.

Uses an external EchoXFlow checkout and existing Task 3 benchmark configurations.
This never trains the appearance filter or fits registration. Inputs stay read-only.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,csv
import numpy as np
import yaml

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def patient_splits(cohort, heldout_fold, original_selection):
    """Keep original outer folds; close inner selection over known patients."""
    known = {e: r for e, r in cohort.items() if r['patient_group'] is not None}
    outer = {e for e, r in known.items() if r['fold'] == heldout_fold}
    pool = set(known) - outer
    selected_patients = {known[e]['patient_group'] for e in set(original_selection) & pool}
    selection = {e for e in pool if known[e]['patient_group'] in selected_patients}
    train = pool - selection
    split = {'train': sorted(train), 'selection': sorted(selection), 'validation': sorted(outer)}
    patients = {k: {known[e]['patient_group'] for e in exams} for k, exams in split.items()}
    for a, b in [('train', 'selection'), ('train', 'validation'), ('selection', 'validation')]:
        if patients[a] & patients[b]:
            raise ValueError(f'Patient leakage between {a} and {b}')
    if not all(split.values()):
        raise ValueError('Empty patient-disjoint split')
    return split


def prepare(args):
    source = json.loads(args.source.read_text())
    protocol = yaml.safe_load(args.protocol.read_text())
    cohort = {}
    with Path(source['patient_groups']).open(newline='') as stream:
        groups = {row['exam_id']: row['patient_group_id'] or None for row in csv.DictReader(stream)}
    baseline_hashes = {}
    for fold in protocol['folds']:
        cell = Path(source['benchmark_root']) / f'segmentation/paper_r256/mvit_v2_s/cartesian/fold={fold}/seed=0'
        baseline_hashes[fold] = sha(cell / 'val_clip_metrics.csv')
        with (cell / 'val_clip_metrics.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                value = {'fold': fold, 'patient_group': groups.get(row['exam_id'])}
                if row['exam_id'] in cohort and cohort[row['exam_id']] != value:
                    raise ValueError('An examination spans outer folds')
                cohort[row['exam_id']] = value
    root = args.out.resolve()
    release = Path(source['release_root'])
    revision = subprocess.check_output(['git', '-C', str(release), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(release), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Use a clean external EchoXFlow checkout')
    manifest = {'protocol': protocol, 'source': source, 'baseline_csv_sha256': baseline_hashes,
                'release_revision': revision, 'script_sha256': sha(__file__),
                'fixed_input_sha256': {k: sha(source[k]) for k in ('enhancer_checkpoint', 'patient_groups')},
                'cohort': cohort, 'excluded_unknown_exams': sorted(e for e,r in cohort.items() if r['patient_group'] is None),
                'cells': [], 'fold_splits': {}}
    if source.get('enhancer_splits'):
        manifest['fixed_input_sha256']['enhancer_splits'] = sha(source['enhancer_splits'])
    epoch_seconds = []
    for fold in protocol['folds']:
        original = Path(source['benchmark_root']) / f'segmentation/paper_r256/mvit_v2_s/cartesian/fold={fold}/seed=0'
        stopping = json.loads((original / 'early_stopping.json').read_text())
        split = patient_splits(cohort, fold, stopping['selection_exam_ids'])
        manifest['fold_splits'][fold] = split
        scores = list(map(json.loads, (original / 'train_scores.jsonl').read_text().splitlines()))
        epoch_seconds.append(float(np.median([r['elapsed_seconds'] for r in scores])))
        for seed in protocol['seeds']:
            cell = root / f'benchmark/segmentation/paper_r256/mvit_v2_s/cartesian/fold={fold}/seed={seed}'
            if (cell / 'training_state.pt').exists() or (cell / 'weights.pt').exists():
                raise ValueError(f'Refusing to prepare over trained artifacts: {cell}')
            cfg = yaml.safe_load((original / 'config.yaml').read_text())
            cfg['trainer'].update(max_epochs=protocol['maximum_epochs'], early_stopping_patience=protocol['early_stopping_patience'],
                                  early_stopping_min_delta=protocol['early_stopping_min_delta'], accelerator='cuda', devices=1)
            cfg['data'].update(seed=seed, selection_seed=0, root_dir=source['data_root'],
                               num_workers=protocol['workers'], prefetch_factor=protocol['prefetch_factor'],
                               sample_ref_cache_dir=str(root / 'sample_refs'))
            cfg['artifacts'].update(root_dir=str(cell.parent), run_name=cell.name, append_timestamp=False,
                                    write_epoch_previews=False, debug_input_target_pngs=False)
            cell.mkdir(parents=True, exist_ok=True)
            config_path = cell / 'config.yaml'
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            manifest['cells'].append({'fold': fold, 'seed': seed, 'path': str(cell), 'config_sha256': sha(config_path),
                                      'original_config_sha256': sha(original / 'config.yaml')})
    manifest['estimated_training_gpu_hours'] = {
        'minimum_epochs': sum(epoch_seconds) * len(protocol['seeds']) * protocol['minimum_epochs'] / 3600,
        'maximum_epochs': sum(epoch_seconds) * len(protocol['seeds']) * protocol['maximum_epochs'] / 3600,
        'basis': 'Median recorded epoch times of the original five folds; loading and evaluation overhead may differ.'}
    write_json(root / 'manifest.json', manifest)
    print(json.dumps({'manifest': str(root / 'manifest.json'), 'runs': len(manifest['cells']),
                      'known_exams': sum(r['patient_group'] is not None for r in cohort.values()),
                      'excluded_exams': len(manifest['excluded_unknown_exams']),
                      'gpu_hours': manifest['estimated_training_gpu_hours']}, indent=2))


def train_cell(args):
    manifest = json.loads(args.manifest.read_text())
    protocol, source = manifest['protocol'], manifest['source']
    cell = next(c for c in manifest['cells'] if c['fold'] == args.fold and c['seed'] == args.seed
                and c.get('condition') == getattr(args, 'condition', None))
    path = Path(cell['path'])
    smoke = getattr(args, 'smoke', False)
    if (path / 'complete.json').exists() and not smoke:
        print(f'Already trained and evaluated: {path}'); return
    release = Path(source['release_root'])
    revision = subprocess.check_output(['git', '-C', str(release), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(release), 'status', '--porcelain'], text=True).strip()
    if dirty or revision != manifest['release_revision'] or sha(path / 'config.yaml') != cell['config_sha256']:
        raise ValueError('Prepared source/config identity changed')
    if not smoke:
        if sha(__file__) != manifest['script_sha256']:
            raise ValueError('Training runner changed since preparation; use the frozen source snapshot')
        for name, expected in manifest['fixed_input_sha256'].items():
            if sha(source[name]) != expected:
                raise ValueError(f'Frozen input changed: {name}')
    sys.path[:0] = [str(release), str(release / 'src')]
    import torch
    from tasks.registry import task_load_config, task_run_training
    from tasks.segmentation.dataset import RawDataset, discover_records, _dataset_kwargs, _train_dataset_kwargs, _val_dataset_kwargs
    from tasks.utils.dataset import build_train_val_dataloaders, exam_split_summary, shutdown_dataloader_workers
    from tasks.bench import _evaluate_trainable

    torch.set_num_threads(protocol['threads'])
    config = task_load_config('segmentation')(path / 'config.yaml')
    splits = manifest['fold_splits'][str(args.fold)]
    if manifest.get('dataset') == 'CAMUS':
        from matched_data import camus_loaders
        def loaders(final=False):
            return camus_loaders(manifest, cell, config, smoke, final=final)
    else:
        selected = set().union(*map(set, splits.values()))
        records = discover_records(root_dir=source['data_root'], exam_ids=selected,
                                   content_types=tuple(config.data['content_types']))
        partition = {k: [r for r in records if r.exam_id in ids] for k, ids in splits.items()}
        if any({r.exam_id for r in partition[k]} != set(ids) for k, ids in splits.items()):
            raise ValueError('Discovered cohort differs from the prepared patient split')
        if smoke:
            partition = {k: v[:4 if k == 'train' else 1] for k, v in partition.items()}

        DatasetClass = RawDataset
        if manifest.get('study') == 'matched':
            from matched_data import echoxflow_class
            DatasetClass = echoxflow_class(source['enhancer_checkpoint'], cell['condition'], source['tensor_cache'])

        def loaders(final=False):
            kw = _dataset_kwargs(config, config.data, source['data_root'])
            train = DatasetClass(records=partition['train'], **kw, **_train_dataset_kwargs(config.data))
            val_kw = _val_dataset_kwargs(config.data)
            val = DatasetClass(records=partition['validation'], **kw, **val_kw)
            selection = DatasetClass(records=partition['selection'], **kw, **{**val_kw, 'include_cartesian_metrics': False})
            return build_train_val_dataloaders(
                task_name='segmentation', root=source['data_root'], records=records,
                train_records=partition['train'], val_records=partition['validation'], selection_records=partition['selection'],
                train_dataset=train, val_dataset=val, selection_dataset=selection, data_config=config.data, seed=args.seed,
                train_batch_size=config.data['train_batch_size'], val_batch_size=config.data['val_batch_size'])

    def prepare_loaders(final=False):
        loaded=loaders(final=final)
        if manifest.get('study') == 'matched':
            from matched_data import warm_caches
            warm_caches(loaded if final else (loaded[0],loaded[2]))
        return loaded

    if smoke:
        train, val, selection = prepare_loaders()
        smoke_root = args.manifest.resolve().parent / 'preflight' / f"{cell.get('condition', 'unfiltered')}-fold={args.fold}-seed={args.seed}"
        started = time.monotonic()
        try:
            result = task_run_training('segmentation')(
                config_path=path / 'config.yaml', train_dataloader=train, val_dataloader=None,
                selection_dataloader=selection, max_epochs=1, max_train_steps=2, resume=False,
                config_overrides={'artifacts': {'root_dir': str(smoke_root), 'run_name': 'smoke'}},
                config_hash=cell['config_sha256'])
            if not np.isfinite(result.loss) or not np.isfinite(result.best_selection_loss):
                raise ValueError('Non-finite real-data preflight loss')
            write_json(smoke_root / 'result.json', {**asdict(result), 'elapsed_seconds': time.monotonic() - started,
                       'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'), 'scope': 'Two training batches and one selection recording'})
        finally:
            shutdown_dataloader_workers(train, val, selection)
        return

    # Two calls reuse the release's full model/optimizer/RNG resume. The first
    # guarantees the minimum budget; the second enables the longer patience.
    history = path / 'train_scores.jsonl'
    completed = max((r['epoch'] for r in map(json.loads, history.read_text().splitlines())), default=0) if history.exists() else 0
    phases = []
    if not (path / 'trained.json').exists():
        if completed < protocol['minimum_epochs']:
            phases.append((protocol['minimum_epochs'], None))
        phases.append((protocol['maximum_epochs'], protocol['early_stopping_patience']))
    for maximum, patience in phases:
        train, val, selection = prepare_loaders()
        summary = exam_split_summary(train=train, val=val, selection=selection)
        summary.update(patient_grouped_selection=True, selection_seed=0, minimum_epochs=protocol['minimum_epochs'])
        try:
            result = task_run_training('segmentation')(
                config_path=path / 'config.yaml', data_root=source['data_root'], train_dataloader=train,
                val_dataloader=None, selection_dataloader=selection, max_epochs=maximum,
                early_stopping_patience=patience, early_stopping_min_delta=protocol['early_stopping_min_delta'],
                selection_split=summary, resume=True, config_hash=cell['config_sha256'])
        finally:
            shutdown_dataloader_workers(train, val, selection)
    if phases:
        write_json(path / 'trained.json', {**asdict(result), 'weights_sha256': sha(path / 'weights.pt'),
                                          'plateau_rule_met': result.stopped_early and result.epochs_completed >= protocol['minimum_epochs'],
                                          'maximum_budget_reached': result.epochs_completed == protocol['maximum_epochs'],
                                          'config_hash': cell['config_sha256']})
    elif sha(path / 'weights.pt') != json.loads((path / 'trained.json').read_text())['weights_sha256']:
        raise ValueError('Completed training weights changed before evaluation')
    train, val, selection = prepare_loaders(final=True)
    try:
        evaluation = _evaluate_trainable('segmentation', config_path=path / 'config.yaml', weights=path / 'weights.pt', val_loader=val)
        evaluation.accumulator.write_csv(path / 'val_clip_metrics.csv', rename={'cartesian_dice_mean': 'dice_mean', 'cartesian_soft_dice_mean': 'soft_dice_mean'})
        write_json(path / 'val_metrics.json', {'metrics': evaluation.exam_means, 'n_val_clips': evaluation.n_clips,
                                             'n_val_exams': evaluation.n_exams, 'source': manifest.get('study', 'historical input-shift study'),
                                             'condition': cell.get('condition', 'unfiltered')})
    finally:
        shutdown_dataloader_workers(train, val, selection)
    write_json(path / 'complete.json', {'trained': True, 'baseline_evaluated': True,
               'condition': cell.get('condition', 'unfiltered'),
               'metrics_sha256': sha(path / 'val_clip_metrics.csv')})


def paired_cell(args):
    """Reproduce the new baseline, then evaluate its paired appearance input."""
    if args.dependency_path:
        # The release environment owns Torch; append only missing project dependencies.
        sys.path.append(str(args.dependency_path.resolve()))
    from evaluate_task3 import run
    manifest = json.loads(args.manifest.read_text())
    cell = next(c for c in manifest['cells'] if c['fold'] == args.fold and c['seed'] == args.seed
                and c.get('condition') == getattr(args, 'condition', None))
    path = Path(cell['path'])
    if not (path / 'complete.json').exists():
        raise ValueError('Training and native evaluation must finish before paired evaluation')
    cfg = {**manifest['source'], 'benchmark_root': str(args.manifest.resolve().parent / 'benchmark'),
           'output_dir': str(path / 'paired_evaluation')}
    cfg_path = path / 'paired_evaluation_config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
        raise ValueError('Paired evaluation configuration changed')
    write_json(cfg_path, cfg)
    for limit, reuse in [(1, False), (0, True)]:
        run(argparse.Namespace(config=cfg_path, device='cuda:0', folds=[args.fold], seed=args.seed,
                               limit=limit, reuse_baseline=reuse, workers=manifest['protocol']['workers'], resume=True))
    write_json(path / 'paired_complete.json', {'fold': args.fold, 'seed': args.seed,
               'provenance_sha256': sha(path / 'paired_evaluation/full/provenance.json')})


def pipeline_cell(args):
    train_cell(args)
    paired_cell(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare', help='Prepare fixed patient folds/configurations; does not train')
    prep.add_argument('--source', type=Path, required=True)
    prep.add_argument('--protocol', type=Path, default=Path('experiments/segmentation/protocol.yaml'))
    prep.add_argument('--out', type=Path, default=Path('outputs/segmentation'))
    prep.set_defaults(func=prepare)
    for command, func in [('train-cell', train_cell), ('paired-cell', paired_cell), ('pipeline-cell', pipeline_cell)]:
        cell = subs.add_parser(command)
        cell.add_argument('--manifest', type=Path, default=Path('outputs/segmentation/manifest.json'))
        cell.add_argument('--fold', type=int, required=True)
        cell.add_argument('--seed', type=int, required=True)
        cell.add_argument('--dependency-path', type=Path)
        cell.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)
