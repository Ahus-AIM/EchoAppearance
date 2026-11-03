"""Paired inference of frozen Task 3 folds with/without the frozen appearance CNN.

All private paths come from --config. Reuses the release's exact validation
sample IDs, contour rasterization, evaluation windows and Dice implementation.
No training, case selection, geometry fitting or threshold tuning is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def run(args):
    cfg = json.loads(args.config.read_text())
    release = Path(cfg['release_root']).resolve()
    sys.path[:0] = [str(release), str(release / 'src'), str(Path('src').resolve())]
    from echoxflow import load_croissant
    from tasks.segmentation.camus import load_cell_model
    from tasks.segmentation.dataset import RawDataset, SampleRef, _dataset_kwargs, _val_dataset_kwargs
    from tasks.utils.device_sampling import cartesian_sampling_grid, sample_stack
    from tasks.utils.training.device import move_to_device
    from tasks.utils.training.metrics import segmentation_val_metrics
    from image_enhancement.beam import BeamEnhancer

    class FixedSamples(RawDataset):
        def __init__(self, refs, **kw):
            self.fixed_refs = refs
            super().__init__(records=list({r.record.recording_id: r.record for r in refs}.values()), **kw)

        def _build_sample_refs(self, *, max_samples):
            return self.fixed_refs

        def _prepare_frames(self, frames, *, panel):
            result = super()._prepare_frames(frames, panel=panel)
            if (not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 255
                    or not np.array_equal(frames, frames.astype(np.uint8))):
                raise ValueError('Expected unnormalised native 8-bit B-mode codes')
            if panel.geometry.get('sample_type') != 'Tissue(8)':
                raise ValueError('Expected source Tissue(8) metadata')
            self.native, self.geometry, self.grid = frames, result[1], result[2]
            self.source = {'recording': vars(panel.bmode.recording), 'data_path': panel.bmode.data_path}
            return result

    out = Path(cfg['output_dir']) / ('preflight' if args.limit else 'full')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.get('threads', 4))
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    checkpoint = torch.load(cfg['enhancer_checkpoint'], map_location='cpu', weights_only=True)
    spec = checkpoint['spec']
    if spec.get('depth', 'none') != 'none' or spec.get('domain', 'beam') != 'beam':
        raise ValueError('This adapter requires the unconditioned beamspace CNN')
    enhancer = BeamEnhancer(**spec).eval().to(device)
    enhancer.load_state_dict(checkpoint['state_dict'], strict=True)
    for p in enhancer.parameters():
        p.requires_grad_(False)
    catalog = load_croissant(root=cfg['data_root'])
    records = {r.recording_id: r for r in catalog.recordings}
    group = {r['exam_id']: r['patient_group_id'] for r in read_csv(cfg['patient_groups'])}
    split = json.loads(Path(cfg['enhancer_splits']).read_text()) if cfg.get('enhancer_splits') else {}
    split_groups = {k: {group[records[r].exam_id] for r in ids} for k, ids in split.items()}
    provenance = {
        'config': cfg, 'arguments': {**vars(args), 'config': str(args.config)},
        'release_revision': subprocess.check_output(['git', '-C', str(release), 'rev-parse', 'HEAD'], text=True).strip(),
        'release_dirty': subprocess.check_output(['git', '-C', str(release), 'status', '--porcelain'], text=True).strip(),
        'enhancer_sha256': digest(cfg['enhancer_checkpoint']), 'script_sha256': digest(__file__),
        'torch': torch.__version__, 'numpy': np.__version__, 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'protocol': 'Frozen segmentation folds; only appearance preprocessing changes. Original Task 3 full-scan '
                    'Cartesian grid at height 256, then bilinear resize to 256x256 and per-frame min-max in both arms. '
                    'CNN receives native uint8/255 and runs before resize/normalisation; no target-dependent transform. '
                    'Task 3 spatiotemporal argmax Dice: clip mean, equal examination mean, then five-fold mean/SD.',
        'precision': 'CNN float32; segmentation uses checkpoint bf16-mixed setting on CUDA',
        'overlap_warning': 'Appearance CNN is fixed across folds and was not retrained within segmentation folds. '
                           'This is an input-perturbation comparison, not end-to-end leakage-free cross-validation.',
        'folds': {},
    }
    previous = None
    if args.resume and (out / 'provenance.json').exists():
        previous = json.loads((out / 'provenance.json').read_text())
        for key in ('config', 'release_revision', 'release_dirty', 'enhancer_sha256', 'torch', 'numpy', 'precision', 'protocol'):
            if previous[key] != provenance[key]:
                raise ValueError(f'Resume would change {key}')
        provenance['prior_script_sha256'] = [*previous.get('prior_script_sha256', []), previous['script_sha256']]
    all_rows = []
    started = time.monotonic()
    for fold in args.folds:
        cell = Path(cfg['benchmark_root']) / 'segmentation/paper_r256/mvit_v2_s/cartesian' / f'fold={fold}' / f'seed={getattr(args, "seed", 0)}'
        original = read_csv(cell / 'val_clip_metrics.csv')
        if args.limit:
            original = original[:args.limit]
        refs = []
        for r in original:
            record = records[r['recording_id']]
            if record.exam_id != r['exam_id'] or record.fold != fold:
                raise ValueError('Stored validation sample disagrees with current fold manifest')
            match = re.fullmatch(r'[^/]+/[^/]+/(.+):frames_annotation_(\d+)_(\d+)', r['sample_id'])
            if match is None:
                raise ValueError(f"Unsupported sample ID {r['sample_id']}")
            role, start, last = match.groups()
            refs.append(SampleRef(record, role, int(start), int(last) + 1, r['sample_id'],
                                  content_type='2d_left_ventricular_strain'))
        model, config = load_cell_model(cell, 'mvit_v2_s')
        model.to(device).eval()
        if config.metrics['dice_reduction'] != 'spatiotemporal':
            raise ValueError('Unexpected Dice reduction')
        kw = {**_dataset_kwargs(config, config.data, cfg['data_root']), **_val_dataset_kwargs(config.data),
              'sample_ref_cache_dir': None, 'sample_ref_cache_read_only': True}
        weight_sha = digest(cell / 'weights.pt')
        trained = json.loads((cell / 'trained.json').read_text())
        if weight_sha != trained['weights_sha256']:
            raise ValueError('Checkpoint checksum differs from training manifest')
        fold_info = {'checkpoint_sha256': weight_sha, 'config_sha256': digest(cell / 'config.yaml'),
                     'validation_csv_sha256': digest(cell / 'val_clip_metrics.csv'),
                     'expected_clips': len(refs), 'expected_exams': len({r.record.exam_id for r in refs})}
        provenance['folds'][fold] = fold_info
        if args.reuse_baseline:
            verified_dir = Path(cfg['output_dir']) / 'preflight'
            verified = json.loads((verified_dir / 'provenance.json').read_text())
            checks = [json.loads(line) for line in (verified_dir / f'fold{fold}.jsonl').read_text().splitlines()]
            if (verified['release_revision'] != provenance['release_revision']
                    or verified['folds'][str(fold)]['checkpoint_sha256'] != weight_sha
                    or verified['folds'][str(fold)]['validation_csv_sha256'] != fold_info['validation_csv_sha256']
                    or not checks
                    or any(abs(r[f'without_{n}'] - r[f'archived_{n}']) > 1e-6
                           for r in checks for n in ('cavity', 'myocardium', 'mean'))):
                raise ValueError('Baseline reuse requires matching per-fold reproduction evidence')
            fold_info['baseline_reproduction_clips'] = len(checks)
            fold_info['baseline_reproduction_max_difference'] = max(
                abs(r[f'without_{n}'] - r[f'archived_{n}']) for r in checks for n in ('cavity', 'myocardium', 'mean'))
        finished = []
        log_path = out / f'fold{fold}.jsonl'
        if previous and log_path.exists():
            old_info = previous['folds'].get(str(fold), {})
            for key in ('checkpoint_sha256', 'config_sha256', 'validation_csv_sha256'):
                if old_info.get(key) != fold_info[key]:
                    raise ValueError(f'Resume would change fold {fold} {key}')
            finished = [json.loads(line) for line in log_path.read_text().splitlines()]
            if [r['sample_id'] for r in finished] != [r.sample_id for r in refs[:len(finished)]]:
                raise ValueError('Resume samples do not match the frozen validation order')
            all_rows.extend(finished)
        # Preserve provenance even if a later sample fails; partial files are not a completed five-fold result.
        (out / 'provenance.json').write_text(json.dumps(provenance, indent=2, default=str) + '\n')
        def prepare(ref):
            tic = time.monotonic()
            dataset = FixedSamples([ref], **kw)
            sample = dataset[0]
            return sample, dataset, time.monotonic() - tic

        # Bounded CPU-only lookahead; each reader has its own dataset state.
        # Consume in frozen order, with no more than workers prepared clips held.
        with ThreadPoolExecutor(max_workers=args.workers) as readers, log_path.open('a' if finished else 'w') as log, torch.inference_mode():
            first = len(finished)
            pending = {i: readers.submit(prepare, refs[i]) for i in range(first, min(len(refs), first + args.workers))}
            for i in range(first, len(refs)):
                ref, old = refs[i], original[i]
                tic = time.monotonic()
                sample, dataset, load_seconds = pending.pop(i).result()
                wait_seconds = time.monotonic() - tic
                next_i = i + args.workers
                if next_i < len(refs):
                    pending[next_i] = readers.submit(prepare, refs[next_i])
                baseline = sample.frames
                native = torch.from_numpy(dataset.native).to(device) / 255
                grid, support = cartesian_sampling_grid(dataset.geometry, dataset.grid,
                                                        source_shape=tuple(native.shape[-2:]), device=device)
                # Independent disabled-path check: unchanged beam values must land
                # on the same Task 3 input pixels, including sector support.
                identity = sample_stack(native, grid, support)
                if identity.shape[-2:] != (256, 256):
                    identity = F.interpolate(identity[:, None], (256, 256), mode='bilinear', align_corners=False)[:, 0]
                def normalize(x):
                    low = x.amin((-2, -1), keepdim=True)
                    span = x.amax((-2, -1), keepdim=True) - low
                    return torch.where(span >= 1e-6, (x - low) / span.clamp_min(1e-6), 0.).clamp(0, 1)
                identity_error = float((normalize(identity).cpu() - baseline[0, 0]).abs().max())
                if identity_error > 2e-4:
                    raise ValueError(f'Identity adapter changed geometry/intensity: max error {identity_error}')
                enhanced = []
                for chunk in native.split(cfg.get('enhancer_batch_frames', 4)):
                    z = enhancer.beam(chunk[:, None])
                    z = F.grid_sample(z, grid.expand(len(chunk), -1, -1, -1), mode='bilinear',
                                      padding_mode='border', align_corners=True)
                    z = enhancer.cart(z)[:, 0].clamp(0, 1)
                    z = torch.where(support[None], z, 0.)
                    if z.shape[-2:] != (256, 256):
                        z = F.interpolate(z[:, None], (256, 256), mode='bilinear', align_corners=False)[:, 0]
                    enhanced.append(normalize(z))
                processed = torch.cat(enhanced)[None, None]
                sample = move_to_device(sample, device)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                prep_seconds = time.monotonic() - tic - wait_seconds
                row = {'fold': fold, 'seed': getattr(args, 'seed', 0), 'sample_id': ref.sample_id, 'exam_id': ref.record.exam_id,
                       'recording_id': ref.record.recording_id, 'source': dataset.source,
                       'patient_group': group.get(ref.record.exam_id),
                       'enhancer_patient_split': next((s for s, gs in split_groups.items()
                                                       if group.get(ref.record.exam_id) in gs), 'unavailable'),
                       'frames': int(native.shape[0]), 'identity_max_abs_error': identity_error,
                       'load_seconds': load_seconds, 'wait_seconds': wait_seconds, 'preprocessing_seconds': prep_seconds,
                       'baseline_reused': args.reuse_baseline,
                       'native_frames_sha256': hashlib.sha256(dataset.native.tobytes()).hexdigest(),
                       'annotation_metadata_sha256': digest(Path(cfg['data_root']) / ref.record.zarr_path / '.zattrs')}
                for arm, x in [('without', sample.frames), ('with', processed)]:
                    if arm == 'without' and args.reuse_baseline:
                        for ch, name in [('ch0', 'cavity'), ('ch1', 'myocardium'), ('mean', 'mean')]:
                            row[f'{arm}_{name}'] = float(old['dice_mean' if ch == 'mean' else f'cartesian_dice_{ch}'])
                        row[f'{arm}_valid_frames'] = float(old['n_valid_frames'])
                        continue
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=device.type == 'cuda'):
                        terms = segmentation_val_metrics(model, replace(sample, frames=x), config)
                    for ch, name in [('ch0', 'cavity'), ('ch1', 'myocardium'), ('mean', 'mean')]:
                        row[f'{arm}_{name}'] = float(terms[f'cartesian_dice_{ch}'].mean())
                    row[f'{arm}_valid_frames'] = float(terms['n_valid_frames'].mean())
                if row['without_valid_frames'] != row['with_valid_frames']:
                    raise ValueError('Matched arms have different scored-frame counts')
                for ch, name in [('ch0', 'cavity'), ('ch1', 'myocardium'), ('mean', 'mean')]:
                    row[f'archived_{name}'] = float(old['dice_mean' if ch == 'mean' else f'cartesian_dice_{ch}'])
                row['seconds'] = time.monotonic() - tic
                log.write(json.dumps(row) + '\n'); log.flush()
                all_rows.append(row)
                if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(refs):
                    print(json.dumps({'fold': fold, 'done': i + 1, 'total': len(refs),
                                      'seconds': round(time.monotonic() - started, 1),
                                      'last_without': row['without_mean'], 'last_with': row['with_mean']}), flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    provenance['elapsed_seconds'] = time.monotonic() - started
    provenance['completed_clips'] = len(all_rows)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2, default=str) + '\n')
    summary = summarize(all_rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def summarize(rows):
    folds = sorted({r['fold'] for r in rows})
    result = {'clips': len(rows), 'exams': len({r['exam_id'] for r in rows}), 'folds': folds,
              'identity_max_abs_error': max(r['identity_max_abs_error'] for r in rows), 'scores': {}}
    for arm in ('without', 'with', 'archived'):
        for name in ('cavity', 'myocardium', 'mean'):
            means = []
            for fold in folds:
                exams = defaultdict(list)
                for r in rows:
                    if r['fold'] == fold:
                        exams[r['exam_id']].append(r[f'{arm}_{name}'])
                means.append(float(np.mean([np.nanmean(v) for v in exams.values()])))
            result['scores'][f'{arm}_{name}'] = {'fold_means': means, 'mean': float(np.mean(means)),
                                                'sd': float(np.std(means, ddof=1)) if len(means) > 1 else None}
    result['enhancer_overlap'] = {s: {'clips': sum(r['enhancer_patient_split'] == s for r in rows),
                                    'exams': len({r['exam_id'] for r in rows if r['enhancer_patient_split'] == s})}
                                  for s in sorted({r['enhancer_patient_split'] for r in rows})}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--seed', type=int, default=0, help='Segmentation training seed; original benchmark uses 0')
    parser.add_argument('--limit', type=int, default=0, help='Preflight clips per fold; 0 evaluates all stored samples')
    parser.add_argument('--reuse-baseline', action='store_true', help='Reuse archived scores after per-fold reproduction')
    parser.add_argument('--workers', type=int, default=1, help='Bounded CPU loading threads; inference stays on one device')
    parser.add_argument('--resume', action='store_true', help='Resume matching completed clips from an interrupted run')
    run(parser.parse_args())
