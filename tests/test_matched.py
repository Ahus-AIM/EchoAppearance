"""Scientific contracts for the matched training study, without patient data."""
import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]/'experiments/segmentation'


def module(name):
    sys.path.insert(0,str(ROOT))
    try:
        spec=importlib.util.spec_from_file_location(name,ROOT/f'{name}.py')
        result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
    finally:sys.path.pop(0)


def test_camus_patient_splits_are_disjoint_complete_and_order_independent():
    m=module('matched');exams=[f'p{i}' for i in range(500)]
    fold,splits=m.camus_splits(exams)
    assert m.camus_splits(list(reversed(exams)))==(fold,splits)
    assert [list(fold.values()).count(i) for i in range(5)]==[100]*5
    for s in splits.values():
        assert len(s['selection'])==50 and len(s['train'])==350
        assert set.union(*map(set,s.values()))==set(exams)
        assert sum(map(len,s.values()))==500


def test_historical_results_cannot_populate_matched_table(tmp_path):
    import json
    p=tmp_path/'manifest.json';p.write_text(json.dumps({'dataset':'EchoXFlow','protocol':{},'cells':[]}))
    with pytest.raises(ValueError,match='Historical'):module('cross_table').analyze(p)


def test_official_camus_split_is_patient_disjoint_and_uses_validation_for_selection():
    import hashlib
    rows=[]
    for i in range(500):
        patient=f'patient{i:04d}'
        rows.append({'exam_id':hashlib.sha256(f'camus:exam:{patient}'.encode()).hexdigest()[:24],
                     'source_patient_id':patient,'split':'train' if i<400 else 'validation' if i<450 else 'test'})
    exams={r['exam_id'] for r in rows}
    _,splits,_=module('matched').official_camus_splits(rows,exams)
    split=splits['0']
    assert [len(split[n]) for n in ('train','selection','validation')]==[400,50,50]
    assert sum(map(len,split.values()))==len(set.union(*map(set,split.values())))
    assert set(split['selection'])=={r['exam_id'] for r in rows if r['split']=='validation'}
    assert set(split['validation'])=={r['exam_id'] for r in rows if r['split']=='test'}


def cross_fixture(tmp_path):
    import csv, json
    m = module('cross_table')
    checkpoint = tmp_path/'appearance.pt'; checkpoint.write_bytes(b'fixed appearance')
    manifest = {'study':'matched','dataset':'EchoXFlow',
                'protocol':{'folds':[0],'seeds':[0,1],'minimum_epochs':25,'maximum_epochs':50,
                            'bootstrap':{'replicates':100,'seed':123}},
                'source':{'enhancer_checkpoint':str(checkpoint)},
                'fold_splits':{'0':{'validation':['a','b']}},
                'cohort':{'a':{'patient_group':'p1'},'b':{'patient_group':'p2'}},'cells':[]}
    fields = ['exam_id','sample_id','n_valid_frames','cartesian_dice_ch0','cartesian_dice_ch1']
    for seed in [0,1]:
        for train in ['unfiltered','filtered']:
            root = tmp_path/f'{train}-{seed}'; root.mkdir()
            (root/'weights.pt').write_bytes(f'{train}-{seed}'.encode())
            (root/'config.yaml').write_text('frozen: true\n')
            cell = {'fold':0,'seed':seed,'condition':train,'path':str(root),'config_sha256':m.sha(root/'config.yaml')}
            manifest['cells'].append(cell)
            (root/'trained.json').write_text(json.dumps({'epochs_completed':25,'weights_sha256':m.sha(root/'weights.pt')}))
            for evaluation in ['unfiltered','filtered']:
                out = root if evaluation==train else root/'cross-evaluation'/evaluation
                out.mkdir(parents=True,exist_ok=True)
                row = m.COMBINATIONS.index((train,evaluation))
                with (out/'val_clip_metrics.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                    for exam,clip in [('a','a1'),('a','a2'),('b','b1')]:
                        # Unequal clips per exam exercises examination averaging.
                        base=.8+row*.02+seed*.002+(0 if exam=='a' else .04)
                        writer.writerow(dict(zip(fields,[exam,clip,32,base,base-.1])))
            (root/'complete.json').write_text(json.dumps({'condition':train,'metrics_sha256':m.sha(root/'val_clip_metrics.csv')}))
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    for cell in manifest['cells']:
        root=Path(cell['path']);evaluation='filtered' if cell['condition']=='unfiltered' else 'unfiltered'
        out=root/'cross-evaluation'/evaluation
        receipt={'training_input':cell['condition'],'evaluation_input':evaluation,'fold':0,'seed':cell['seed'],
                 'config_sha256':cell['config_sha256'],'weights_sha256':m.sha(root/'weights.pt'),
                 'manifest_sha256':m.sha(path),'appearance_sha256':m.sha(checkpoint),
                 'metrics_sha256':m.sha(out/'val_clip_metrics.csv')}
        (out/'complete.json').write_text(json.dumps(receipt))
    return m,path


def test_four_domain_rows_preserve_exam_weighting_and_matched_bootstrap(tmp_path):
    import warnings
    m,path=cross_fixture(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        result=m.analyze(path)
    np.testing.assert_allclose(result['mean'][:,0],[.821,.841,.861,.881],atol=1e-14)
    np.testing.assert_allclose(result['contrasts']['unfiltered_training_input_shift']['delta'],[.02,.02])
    np.testing.assert_allclose(result['contrasts']['filtered_training_input_shift']['delta'],[-.02,-.02])
    camus={'mean':np.tile(result['mean'][:,:1],(1,3)),'ci95':np.tile(result['ci95'][:,:,:1],(1,1,3)),
           'combinations':result['combinations']}
    text,audit=m.table({'EchoXFlow':result,'CAMUS':camus})
    assert text.count(r'\textbf{')==5 and r'\diceci{\textbf{88.1}}' in text
    labels=('None','Training and evaluation','Evaluation only','Training only')
    assert [l.split(' & ')[0] for l in text.splitlines() if l.startswith(labels)]==list(labels)
    assert all(line.count('&')==5 for line in text.splitlines() if line.startswith(labels))
    assert len(audit)==20 and sum(a['best_mean'] for a in audit)==5


def test_cross_domain_table_refuses_changed_evaluation_samples(tmp_path):
    import json
    m,path=cross_fixture(tmp_path)
    out=tmp_path/'unfiltered-0/cross-evaluation/filtered'
    scores=out/'val_clip_metrics.csv';scores.write_text(scores.read_text().replace('a1','wrong-clip'))
    receipt=out/'complete.json';x=json.loads(receipt.read_text());x['metrics_sha256']=m.sha(scores);receipt.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='scored clips'):m.analyze(path)


def test_cross_domain_table_refuses_missing_evaluation(tmp_path):
    m,path=cross_fixture(tmp_path)
    (tmp_path/'unfiltered-1/cross-evaluation/filtered/complete.json').unlink()
    with pytest.raises(ValueError,match='Missing cross-domain'):m.analyze(path)


def test_displayed_interval_contains_the_original_asymmetric_interval():
    m=module('cross_table')
    mean,width=m.symmetric_display(.8123,.7981,.8302)
    assert (mean,width)==('81.2','1.9')
    with pytest.raises(ValueError):m.symmetric_display(float('nan'),.7,.8)
