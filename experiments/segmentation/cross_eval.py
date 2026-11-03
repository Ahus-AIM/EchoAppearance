"""Evaluate a completed segmentation checkpoint on either appearance domain.

No training, checkpoint selection or original result files are modified.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

from retrain_task3 import sha, write_json


def make_loaders(manifest, cell, config, evaluation_input):
    """Use the original final-evaluation loaders; change only input appearance."""
    from matched_data import camus_loaders, echoxflow_class
    from tasks.segmentation.dataset import discover_records, _dataset_kwargs, _train_dataset_kwargs, _val_dataset_kwargs
    from tasks.utils.dataset import build_train_val_dataloaders

    evaluated = {**cell, 'condition': evaluation_input}
    if manifest['dataset'] == 'CAMUS':
        return camus_loaders(manifest, evaluated, config, final=True)
    source = manifest['source']
    splits = manifest['fold_splits'][str(cell['fold'])]
    records = discover_records(root_dir=source['data_root'], exam_ids=set().union(*map(set, splits.values())),
                               content_types=tuple(config.data['content_types']))
    partition = {k: [r for r in records if r.exam_id in ids] for k, ids in splits.items()}
    if any({r.exam_id for r in partition[k]} != set(ids) for k, ids in splits.items()):
        raise ValueError('Evaluation cohort differs from the frozen split')
    dataset = echoxflow_class(source['enhancer_checkpoint'], evaluation_input, source['tensor_cache'])
    kw = _dataset_kwargs(config, config.data, source['data_root'])
    val_kw = _val_dataset_kwargs(config.data)
    return build_train_val_dataloaders(
        task_name='segmentation', root=source['data_root'], records=records,
        train_records=partition['train'], val_records=partition['validation'], selection_records=partition['selection'],
        train_dataset=dataset(records=partition['train'], **kw, **_train_dataset_kwargs(config.data)),
        val_dataset=dataset(records=partition['validation'], **kw, **val_kw),
        selection_dataset=dataset(records=partition['selection'], **kw, **{**val_kw, 'include_cartesian_metrics': False}),
        data_config=config.data, seed=cell['seed'], train_batch_size=config.data['train_batch_size'],
        val_batch_size=config.data['val_batch_size'])


def score_rows(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    keys = [(r['exam_id'], r['sample_id']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate evaluation sample')
    return dict(zip(keys, rows))


def run(args):
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('study') != 'matched':
        raise ValueError('Expected a frozen matched-training study')
    cell = next(c for c in manifest['cells'] if (c['fold'], c['seed'], c['condition']) ==
                (args.fold, args.seed, args.training_input))
    root = Path(cell['path'])
    trained = json.loads((root / 'trained.json').read_text())
    done = json.loads((root / 'complete.json').read_text())
    if sha(root / 'weights.pt') != trained['weights_sha256'] or sha(root / 'config.yaml') != cell['config_sha256']:
        raise ValueError('Frozen segmentation checkpoint/config changed')
    if done['condition'] != args.training_input or done['metrics_sha256'] != sha(root / 'val_clip_metrics.csv'):
        raise ValueError('Original evaluation changed')
    source = manifest['source']
    release = Path(source['release_root'])
    revision = subprocess.check_output(['git', '-C', str(release), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != manifest['release_revision'] or subprocess.check_output(['git', '-C', str(release), 'status', '--porcelain'], text=True).strip():
        raise ValueError('External task source changed')
    for name, expected in manifest['study_code_sha256'].items():
        if sha(Path(args.frozen_code) / name) != expected:
            raise ValueError(f'Frozen study code changed: {name}')
    for key, expected in manifest['fixed_input_sha256'].items():
        if sha(source[key]) != expected:
            raise ValueError(f'Frozen source changed: {key}')
    out = root / 'cross-evaluation' / args.evaluation_input
    contract = {'training_input': args.training_input, 'evaluation_input': args.evaluation_input,
                'fold': args.fold, 'seed': args.seed, 'weights_sha256': trained['weights_sha256'],
                'config_sha256': cell['config_sha256'], 'manifest_sha256': sha(args.manifest),
                'evaluator_sha256': sha(Path(__file__)),
                'appearance_sha256': sha(source['enhancer_checkpoint'])}
    receipt = out / 'complete.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if any(previous.get(k) != v for k, v in contract.items()) or previous['metrics_sha256'] != sha(out / 'val_clip_metrics.csv'):
            raise ValueError('Completed cross-evaluation has changed')
        print(f'Already complete: {out}', flush=True)
        return
    sys.path[:0] = [str(release), str(release / 'src')]
    import torch
    import numpy as np
    from tasks.registry import task_load_config
    from tasks.bench import _evaluate_trainable
    from tasks.utils.dataset import shutdown_dataloader_workers
    from matched_data import warm_caches

    torch.set_num_threads(manifest['protocol']['threads'])
    config = task_load_config('segmentation')(root / 'config.yaml')
    started = time.monotonic()
    loaders = make_loaders(manifest, cell, config, args.evaluation_input)
    try:
        warm_caches(loaders)
        evaluated = _evaluate_trainable('segmentation', config_path=root / 'config.yaml', weights=root / 'weights.pt', val_loader=loaders[1])
        out.mkdir(parents=True, exist_ok=True)
        evaluated.accumulator.write_csv(out / 'val_clip_metrics.csv', rename={
            'cartesian_dice_mean': 'dice_mean', 'cartesian_soft_dice_mean': 'soft_dice_mean'})
        actual, original = score_rows(out / 'val_clip_metrics.csv'), score_rows(root / 'val_clip_metrics.csv')
        if actual.keys() != original.keys():
            raise ValueError('Evaluation input changed the scored clips')
        classes = 3 if manifest['dataset'] == 'CAMUS' else 2
        differences = []
        for key, row in actual.items():
            if float(row['n_valid_frames']) != float(original[key]['n_valid_frames']):
                raise ValueError('Scored-frame counts changed')
            values = [float(row[f'cartesian_dice_ch{i}']) for i in range(classes)]
            if not np.isfinite(values).all() or not all(0 <= v <= 1 for v in values):
                raise ValueError('Invalid Dice score')
            if args.training_input == args.evaluation_input:
                differences.append(np.array(values) - [float(original[key][f'cartesian_dice_ch{i}']) for i in range(classes)])
        control = None
        if differences:
            differences = np.stack(differences)
            # The frozen external runner uses bf16 and cuDNN kernel benchmarking.
            # Different deterministic kernels can differ in their rounding. Bound
            # both per-clip and mean drift, and retain the measured control values.
            if np.abs(differences).max() > 5e-4 or np.abs(differences.mean(axis=0)).max() > 5e-5:
                raise ValueError('Repeat evaluation exceeds bf16 numerical tolerance')
            control = {'max_absolute_clip_dice_difference': np.abs(differences).max().item(),
                       'mean_dice_difference': differences.mean(axis=0).tolist(),
                       'clip_tolerance': 5e-4, 'mean_tolerance': 5e-5}
        if sha(root / 'weights.pt') != contract['weights_sha256']:
            raise ValueError('Checkpoint changed during evaluation')
        write_json(receipt, {**contract, 'metrics_sha256': sha(out / 'val_clip_metrics.csv'),
                            'clips': len(actual), 'elapsed_seconds': time.monotonic() - started,
                            'same_domain_control': control})
        print(json.dumps(json.loads(receipt.read_text())), flush=True)
    finally:
        shutdown_dataloader_workers(*loaders)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--training-input', choices=('unfiltered', 'filtered'), required=True)
    p.add_argument('--evaluation-input', choices=('unfiltered', 'filtered'), required=True)
    p.add_argument('--frozen-code', type=Path, default=Path(__file__).resolve().parent)
    run(p.parse_args())
