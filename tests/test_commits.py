import importlib.util
from pathlib import Path
import pytest

@pytest.mark.parametrize('subject,expected',[('feat: add tensor filtering',True),
    ('fix(api): retain geometry',True),('feat!: change input contract',True),
    ('chore(release): 0.2.0 [skip ci]',True),('Release models',False),('feat: ',False)])
def test_conventional_subject(subject,expected):
    path=Path(__file__).resolve().parents[1]/'scripts/check_commits.py'
    spec=importlib.util.spec_from_file_location('check_commits',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.valid(subject) is expected
