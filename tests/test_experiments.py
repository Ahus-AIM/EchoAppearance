"""No-data patient grouping and paired bootstrap tests for the segmentation protocol."""
from pathlib import Path
import importlib.util
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/f'experiments/segmentation/{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_inner_selection_closes_over_patients_and_unknown_linkage_is_excluded():
    pytest.importorskip('numpy');pytest.importorskip('yaml')
    split=module('retrain_task3').patient_splits({'a':{'fold':0,'patient_group':'p'},
        'b':{'fold':1,'patient_group':'q'},'c':{'fold':1,'patient_group':'q'},
        'd':{'fold':1,'patient_group':'r'},'e':{'fold':2,'patient_group':None}},0,['b'])
    assert split=={'train':['d'],'selection':['b','c'],'validation':['a']}


def test_paired_patient_bootstrap_preserves_a_known_constant_effect():
    np=pytest.importorskip('numpy')
    values=np.full((2,3,2,2),.8);values[:,:,1,:]-=.1
    result=module('aggregate').patient_bootstrap([(values,['p','p','q']),(values,['r','r','s'])],replicates=100,seed=5)
    np.testing.assert_allclose(result['delta'],[-.1,-.1],atol=1e-12)
    np.testing.assert_allclose(result['delta_ci95'],[[-.1,-.1],[-.1,-.1]],atol=1e-12)


def test_camus_download_extracts_only_selected_recording_and_pins_revision(tmp_path,monkeypatch):
    import io,sys,tarfile,types
    pytest.importorskip('numpy')
    path=ROOT/'experiments/camus/plot.py'
    spec=importlib.util.spec_from_file_location('camus_plot',path)
    plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)
    archive=tmp_path/'exam.tar';prefix='exams/exam_e/recording_r.zarr'
    with tarfile.open(archive,'w') as tar:
        for name,data in [(prefix+'/.zattrs',b'{}'),('exams/exam_e/recording_other.zarr/.zattrs',b'other')]:
            member=tarfile.TarInfo(name);member.size=len(data);tar.addfile(member,io.BytesIO(data))
    calls=[]
    def download(repo,filename,**kwargs):
        calls.append((repo,filename,kwargs));return str(archive)
    monkeypatch.setitem(sys.modules,'huggingface_hub',types.SimpleNamespace(hf_hub_download=download))
    cfg={'repo_id':'owner/CAMUS-beamspace','revision':'a'*40,'cache_dir':str(tmp_path/'cache'),
         'recordings':[{'exam_id':'e','recording_id':'r','view':'A2C'}]}
    result=plot.download_sources(cfg)
    assert (result[0][1]/'.zattrs').read_text()=='{}'
    assert not list((tmp_path/'cache').rglob('recording_other.zarr'))
    assert calls==[('owner/CAMUS-beamspace','exams/exam_e.tar',{'repo_type':'dataset','revision':'a'*40})]
    with tarfile.open(archive,'w') as tar:
        member=tarfile.TarInfo(prefix+'/../../outside');member.size=1
        tar.addfile(member,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='Unexpected archive member'):
        plot.download_sources(cfg)
