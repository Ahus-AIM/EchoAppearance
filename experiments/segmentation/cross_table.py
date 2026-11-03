"""Four-row train/evaluation appearance-domain Dice table from frozen checkpoints.

Validates the complete matched design and every cross-domain evaluation receipt,
then writes the LaTeX table plus full-precision (.json) and presentation-audit
(.presentation.json) records.

The displayed mean stays unchanged. The symmetric CI half-width is rounded up
to 0.1 percentage points so that the displayed range contains the full original
asymmetric patient-bootstrap interval, including mean-display rounding.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from aggregate import patient_bootstrap, sha
from cross_eval import score_rows

DATASETS = {'EchoXFlow': ('LV cavity', 'Myocardium'), 'CAMUS': ('LV cavity', 'Myocardium', 'Left atrium')}
ARMS = ('unfiltered', 'filtered')
COMBINATIONS = tuple((train, evaluation) for train in ARMS for evaluation in ARMS)
# (source row in COMBINATIONS, label); matched conditions first, then appearance shifts.
ROWS = [(0, 'None'), (3, 'Training and evaluation'), (1, 'Evaluation only'), (2, 'Training only')]


def analyze_matched(path):
    manifest=json.loads(path.read_text());protocol=manifest['protocol'];classes=DATASETS[manifest['dataset']]
    if manifest.get('study')!='matched':raise ValueError('Historical input-shift results cannot enter this table')
    expected={(f,s,a) for f in protocol['folds'] for s in protocol['seeds'] for a in ARMS}
    cells={(c['fold'],c['seed'],c['condition']):c for c in manifest['cells']}
    if len(cells)!=len(manifest['cells']) or set(cells)!=expected:raise ValueError('Incomplete matched design')
    folders=[]
    for fold in protocol['folds']:
        exams=manifest['fold_splits'][str(fold)]['validation'];values=[];signature=None
        for seed in protocol['seeds']:
            arms=[]
            for arm in ARMS:
                cell=cells[fold,seed,arm];root=Path(cell['path'])
                done=json.loads((root/'complete.json').read_text());trained=json.loads((root/'trained.json').read_text())
                if done['condition']!=arm or done['metrics_sha256']!=sha(root/'val_clip_metrics.csv'):raise ValueError('Wrong condition or changed scores')
                if sha(root/'weights.pt')!=trained['weights_sha256'] or sha(root/'config.yaml')!=cell['config_sha256']:raise ValueError('Changed model/config')
                if not protocol['minimum_epochs']<=trained['epochs_completed']<=protocol['maximum_epochs']:raise ValueError('Incomplete training budget')
                with (root/'val_clip_metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
                current=sorted((r['exam_id'],r['sample_id']) for r in rows)
                if len(current)!=len(set(current)) or (signature is not None and current!=signature):raise ValueError('Arms/seeds use different samples')
                signature=current
                if {r['exam_id'] for r in rows}!=set(exams):raise ValueError('Changed outer fold')
                arms.append([[np.mean([float(r[f'cartesian_dice_ch{i}']) for r in rows if r['exam_id']==e])
                              for i in range(len(classes))] for e in exams])
            values.append(np.stack(arms,axis=1))
        folders.append((np.stack(values),[manifest['cohort'][e]['patient_group'] for e in exams]))
    return patient_bootstrap(folders,replicates=protocol['bootstrap']['replicates'],seed=protocol['bootstrap']['seed'])


def analyze(path):
    manifest = json.loads(path.read_text())
    # Validate the complete original model/config/score matrix before adding inputs.
    matched = analyze_matched(path)
    protocol = manifest['protocol']
    classes = DATASETS[manifest['dataset']]
    cells = {(c['fold'], c['seed'], c['condition']): c for c in manifest['cells']}
    folders = []
    patients_seen = set()
    for fold in protocol['folds']:
        exams = manifest['fold_splits'][str(fold)]['validation']
        patients = [manifest['cohort'][e]['patient_group'] for e in exams]
        if patients_seen.intersection(patients):
            raise ValueError('Test patients overlap across folds')
        patients_seen.update(patients)
        seed_values = []
        signature = None
        for seed in protocol['seeds']:
            rows_by_condition = []
            for training, evaluation in COMBINATIONS:
                cell = cells[fold, seed, training]
                root = Path(cell['path'])
                reference = score_rows(root / 'val_clip_metrics.csv')
                if training == evaluation:
                    rows = reference
                else:
                    out = root / 'cross-evaluation' / evaluation
                    receipt = out / 'complete.json'
                    if not receipt.exists():
                        raise ValueError(f'Missing cross-domain evaluation: {out}')
                    completed = json.loads(receipt.read_text())
                    expected = {'training_input': training, 'evaluation_input': evaluation, 'fold': fold,
                                'seed': seed, 'config_sha256': cell['config_sha256'],
                                'weights_sha256': sha(root / 'weights.pt'), 'manifest_sha256': sha(path),
                                'appearance_sha256': sha(manifest['source']['enhancer_checkpoint'])}
                    if any(completed.get(k) != v for k, v in expected.items()) or completed['metrics_sha256'] != sha(out / 'val_clip_metrics.csv'):
                        raise ValueError('Cross-evaluation provenance or scores changed')
                    rows = score_rows(out / 'val_clip_metrics.csv')
                    if rows.keys() != reference.keys() or any(float(rows[k]['n_valid_frames']) != float(reference[k]['n_valid_frames']) for k in rows):
                        raise ValueError('Cross-domain evaluation changed scored clips/frames')
                current = sorted(rows)
                if signature is not None and current != signature:
                    raise ValueError('Conditions/seeds use different evaluation samples')
                signature = current
                if {k[0] for k in rows} != set(exams):
                    raise ValueError('Evaluation examinations changed')
                values = np.array([[np.mean([float(row[f'cartesian_dice_ch{i}']) for (exam, _), row in rows.items() if exam == e])
                                    for i in range(len(classes))] for e in exams])
                if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                    raise ValueError('Invalid Dice scores')
                rows_by_condition.append(values)
            seed_values.append(np.stack(rows_by_condition, axis=1))
        folders.append((np.stack(seed_values), patients))
    result = patient_bootstrap(folders, replicates=protocol['bootstrap']['replicates'], seed=protocol['bootstrap']['seed'])
    np.testing.assert_allclose(result['mean'][[0, 3]], matched['mean'], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result['ci95'][:, [0, 3]], matched['ci95'], rtol=0, atol=1e-14)
    if manifest['dataset'] == 'CAMUS':
        result['fold_sd'] = None
    # Explicit contrasts replace the two-arm helper's default delta fields.
    result.pop('delta'); result.pop('delta_ci95')
    draws = result['bootstrap_draws']
    result['contrasts'] = {}
    for label, a, b in [('unfiltered_training_input_shift', 1, 0),
                         ('filtered_training_input_shift', 2, 3), ('matched_filtered_minus_unfiltered', 3, 0)]:
        result['contrasts'][label] = {'delta': result['mean'][a] - result['mean'][b],
                                     'ci95': np.quantile(draws[:, a] - draws[:, b], [.025, .975], axis=0)}
    result['combinations'] = COMBINATIONS
    return result


def symmetric_display(mean, lo, hi):
    if not all(math.isfinite(v) for v in (mean, lo, hi)) or lo > hi:
        raise ValueError('Expected a complete finite confidence interval')
    displayed_mean = float(f'{100 * mean:.1f}')
    radius = max(displayed_mean - 100 * lo, 100 * hi - displayed_mean, 0.)
    half_width = math.ceil(10 * radius - 1e-10) / 10
    assert displayed_mean - half_width <= 100 * lo + 1e-10
    assert displayed_mean + half_width >= 100 * hi - 1e-10
    return f'{displayed_mean:.1f}', f'{half_width:.1f}'


def table(results):
    for name, classes in DATASETS.items():
        result = results[name]
        if tuple(map(tuple, result['combinations'])) != COMBINATIONS:
            raise ValueError('Unexpected train/evaluation row order')
        if len(result['mean']) != 4 or any(len(row) != len(classes) for row in result['mean']):
            raise ValueError('Incomplete result matrix')
    lines = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccc@{}}',
             r'\toprule',
             r'& \multicolumn{2}{c}{EchoXFlow} & \multicolumn{3}{c}{CAMUS} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}',
             r'Filtering applied & LV cavity & Myocardium & LV cavity & Myocardium & Left atrium \\',
             r'\midrule']
    audit = []
    for position, (row, label) in enumerate(ROWS):
        if position == 2:
            lines.append(r'\addlinespace[3pt]')
        entries = []
        for name, classes in DATASETS.items():
            result = results[name]
            for i, structure in enumerate(classes):
                mean = result['mean'][row][i]
                lo, hi = (result['ci95'][bound][row][i] for bound in (0, 1))
                number, width = symmetric_display(mean, lo, hi)
                best = mean >= max(values[i] for values in result['mean']) - 1e-12
                number_tex = r'\textbf{' + number + '}' if best else number
                entries.append(r'\diceci{' + number_tex + '}{' + width + '}')
                audit.append({'dataset': name, 'class': structure, 'filtering_applied': label,
                              'source_row': row, 'mean': mean, 'ci95': [lo, hi],
                              'displayed_mean_percent': float(number), 'ci_half_width_percent': float(width),
                              'best_mean': best})
        lines.append(label + ' & ' + ' & '.join(entries) + r' \\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular*}']) + '\n', audit


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        results = {name: analyze(args.root / name / 'manifest.json') for name in DATASETS}
    rendered, audit = table(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    args.out.with_suffix('.json').write_text(json.dumps(results, default=lambda x: x.tolist(), indent=2) + '\n')
    args.out.with_suffix('.presentation.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(args.out.with_suffix('.json').read_bytes()).hexdigest(),
        'interval_display': 'conservative symmetric CI about displayed mean; half-width rounded upward to 0.1 percentage points',
        'cells': audit}, indent=2) + '\n')
    print(args.out)
