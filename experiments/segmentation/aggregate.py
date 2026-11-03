"""Complete five-fold/seed Dice analysis with paired patient-cluster bootstrap."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
ARMS = ('without','with')
CLASSES = ('cavity','myocardium')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patient_bootstrap(folds, *, replicates, seed):
    """folds: [(seed x examination x condition x class scores, patient ids)]."""
    rng = np.random.default_rng(seed)
    estimates, draws, per_seed = [], [], []
    for values, patients in folds:
        values = np.asarray(values, dtype=float)
        if values.ndim != 4 or values.shape[1] != len(patients) or not np.isfinite(values).all():
            raise ValueError('Expected complete seed/examination/condition/class scores')
        mean_seeds = values.mean(axis=0)
        estimates.append(mean_seeds.mean(axis=0))
        per_seed.append(values.mean(axis=1))
        unique = sorted(set(patients))
        if not unique or any(p is None for p in patients):
            raise ValueError('Patient linkage is required for bootstrap analysis')
        membership = np.equal.outer(unique, patients)
        sums = np.einsum('pe,eac->pac', membership, mean_seeds)
        counts = membership.sum(axis=1)
        selected = rng.integers(len(unique), size=(replicates, len(unique)))
        draws.append(sums[selected].sum(axis=1) / counts[selected].sum(axis=1)[:, None, None])
    fold_means = np.stack(estimates)
    seed_means = np.stack(per_seed).mean(axis=0)
    distribution = np.stack(draws).mean(axis=0)
    return {'mean': fold_means.mean(axis=0), 'ci95': np.quantile(distribution, [.025, .975], axis=0),
            'delta': (fold_means[:, 1] - fold_means[:, 0]).mean(axis=0),
            'delta_ci95': np.quantile(distribution[:, 1] - distribution[:, 0], [.025, .975], axis=0),
            'fold_means': fold_means, 'fold_sd': fold_means.std(axis=0, ddof=1),
            'seed_means': seed_means, 'seed_sd': seed_means.std(axis=0, ddof=1) if len(seed_means) > 1 else None,
            'bootstrap_draws': distribution}


def load_complete(manifest_path, seeds=None):
    manifest = json.loads(manifest_path.read_text())
    if seeds is not None:
        if not seeds or len(set(seeds)) != len(seeds) or set(seeds) - set(manifest['protocol']['seeds']):
            raise ValueError('Requested seed subset is absent or duplicated')
        manifest['protocol']['seeds'] = sorted(seeds)
        manifest['cells'] = [c for c in manifest['cells'] if c['seed'] in seeds]
    protocol = manifest['protocol']
    expected = {(f, s) for f in protocol['folds'] for s in protocol['seeds']}
    if len(manifest['cells']) != len(expected) or {(c['fold'], c['seed']) for c in manifest['cells']} != expected:
        raise ValueError('Manifest does not contain the complete fold x seed design')
    cells, common, provenance_hashes, stopping = {}, {}, {}, []
    overlap, all_exams = {}, set()
    for cell in manifest['cells']:
        fold, seed, path = cell['fold'], cell['seed'], Path(cell['path'])
        complete = json.loads((path / 'paired_complete.json').read_text())
        result_dir = path / 'paired_evaluation/full'
        provenance = json.loads((result_dir / 'provenance.json').read_text())
        trained = json.loads((path / 'trained.json').read_text())
        info = provenance['folds'][str(fold)]
        if complete['provenance_sha256'] != sha(result_dir / 'provenance.json'):
            raise ValueError('Completed paired provenance was modified')
        if provenance['release_revision'] != manifest['release_revision'] or provenance['release_dirty']:
            raise ValueError('Release code identity changed')
        if provenance['arguments']['limit'] or set(provenance['folds']) != {str(fold)}:
            raise ValueError('Expected one complete evaluation fold per trained cell')
        if provenance['arguments']['seed'] != seed:
            raise ValueError('Wrong evaluation seed')
        for actual, wanted in [(sha(path / 'config.yaml'), cell['config_sha256']),
                               (sha(path / 'weights.pt'), trained['weights_sha256']),
                               (sha(path / 'weights.pt'), info['checkpoint_sha256']),
                               (sha(path / 'val_clip_metrics.csv'), info['validation_csv_sha256'])]:
            if actual != wanted:
                raise ValueError('Training/evaluation source identity changed')
        if trained['epochs_completed'] < protocol['minimum_epochs']:
            raise ValueError('A cell did not reach the minimum training budget')
        if not protocol['minimum_epochs'] <= trained['epochs_completed'] <= protocol['maximum_epochs']:
            raise ValueError('Training epoch budget changed')
        frozen = provenance['enhancer_sha256']
        if cells and frozen != next(iter(cells.values()))['enhancer_sha256']:
            raise ValueError('Appearance checkpoint changed across cells')
        rows = [json.loads(line) for line in (result_dir / f'fold{fold}.jsonl').read_text().splitlines()]
        with (path / 'val_clip_metrics.csv').open(newline='') as stream:
            original = list(csv.DictReader(stream))
        if (len(rows) != info['expected_clips'] or provenance['completed_clips'] != len(rows)
                or [r['sample_id'] for r in rows] != [r['sample_id'] for r in original]):
            raise ValueError('Incomplete or reordered paired evaluation')
        if len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate validation sample')
        exams = set(manifest['fold_splits'][str(fold)]['validation'])
        if {r['exam_id'] for r in rows} != exams:
            raise ValueError('Evaluation examinations differ from the prepared cohort')
        by_exam = {}
        signature = []
        for row, old in zip(rows, original):
            patient = manifest['cohort'][row['exam_id']]['patient_group']
            if patient is None or row['patient_group'] != patient or row['fold'] != fold or row['seed'] != seed:
                raise ValueError('Patient, fold or seed identity changed')
            if not row['with_valid_frames'] == row['without_valid_frames'] == float(old['n_valid_frames']) > 0:
                raise ValueError('Scored-frame counts changed')
            if row['identity_max_abs_error'] > 2e-4:
                raise ValueError('Disabled appearance path changed pixel correspondence')
            score = np.array([[row[f'{arm}_{name}'] for name in CLASSES] for arm in ARMS])
            if not np.isfinite(score).all() or (score < 0).any() or (score > 1).any():
                raise ValueError('Invalid class Dice')
            baseline = np.array([float(old[f'cartesian_dice_ch{i}']) for i in range(2)])
            np.testing.assert_allclose(score[0], baseline, rtol=0, atol=1e-6)
            by_exam.setdefault(row['exam_id'], []).append(score)
            prior = overlap.setdefault(row['exam_id'], row['enhancer_patient_split'])
            if prior != row['enhancer_patient_split']:
                raise ValueError('Appearance training overlap classification changed')
            signature.append(tuple(row[k] for k in ('sample_id', 'native_frames_sha256',
                             'annotation_metadata_sha256', 'with_valid_frames')))
        if fold in common and common[fold] != signature:
            raise ValueError('Seed comparisons do not share the same samples and source data')
        common[fold] = signature
        cells[fold, seed] = {'exams': {e: np.mean(v, axis=0) for e, v in by_exam.items()}, 'enhancer_sha256': frozen}
        provenance_hashes[f'fold={fold}/seed={seed}'] = complete['provenance_sha256']
        stopping.append({'fold': fold, 'seed': seed, **{k: trained[k] for k in
                        ('epochs_completed', 'best_epoch', 'best_selection_loss', 'plateau_rule_met', 'maximum_budget_reached')}})
        all_exams.update(exams)
    patients_by_fold = {}
    arrays = []
    for fold in protocol['folds']:
        exams = sorted(manifest['fold_splits'][str(fold)]['validation'])
        patients = [manifest['cohort'][e]['patient_group'] for e in exams]
        for other in patients_by_fold.values():
            if set(patients) & other:
                raise ValueError('Outer folds share patients')
        patients_by_fold[fold] = set(patients)
        arrays.append((np.array([[cells[fold, seed]['exams'][e] for e in exams] for seed in protocol['seeds']]), patients))
    return manifest, arrays, {'clips': sum(len(v) for v in common.values()), 'exams': len(all_exams),
        'patients': len(set().union(*patients_by_fold.values())),
        'excluded_unknown_exams': len(manifest['excluded_unknown_exams']),
        'enhancer_training_overlap_exams': (sum(v == 'train' for v in overlap.values())
             if any(v != 'unavailable' for v in overlap.values()) else None),
        'cell_provenance_sha256': provenance_hashes, 'training_stops': stopping}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/segmentation/manifest.json'))
    parser.add_argument('--out', type=Path, default=Path('outputs/segmentation/analysis'))
    args = parser.parse_args()
    manifest, arrays, evidence = load_complete(args.manifest)
    bootstrap = manifest['protocol']['bootstrap']
    result = patient_bootstrap(arrays, replicates=bootstrap['replicates'], seed=bootstrap['seed'])
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out/'bootstrap_draws.npz', draws=result.pop('bootstrap_draws'))
    summary = {**evidence, 'folds':manifest['protocol']['folds'], 'seeds':manifest['protocol']['seeds'],
               'arms':ARMS, 'classes':CLASSES, 'bootstrap':bootstrap,
               'scores':{k:None if v is None else v.tolist() for k,v in result.items()},
               'limitation':'Fixed appearance model overlaps segmentation evaluation patients; not independent whole-pipeline CV.'}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['scores'],indent=2))
