"""Remote numerical replies must match the frozen local source and request."""
import pytest
from gradscf.model.autoxc import Candidate
from gradscf.model.autoxc.history import source_digest


def ready():
    return {'baseline_count':100,'source':source_digest(),
            'configuration':{'fingerprint':'frozen'}}


def test_remote_rejects_a_different_source_before_model_design(tmp_path):
    from gradscf.model.autoxc.workflows.xc100.distributed import RemoteEvaluator
    data=ready();data['source']='different-code'
    with pytest.raises(RuntimeError,match='source'):
        RemoteEvaluator('jy_gpu_1','/project',data,receipt_directory=tmp_path)


def test_remote_rejects_a_changed_result_identity(tmp_path,monkeypatch):
    import subprocess
    from gradscf.model.autoxc.workflows.xc100.distributed import RemoteEvaluator
    evaluator=RemoteEvaluator('jy_gpu_1','/project',ready(),receipt_directory=tmp_path)
    monkeypatch.setattr(evaluator,'_ssh',lambda *a,**k:'')
    monkeypatch.setattr('subprocess.run',lambda *a,**k:subprocess.CompletedProcess([],0,'',''))
    monkeypatch.setattr(evaluator,'read',lambda name:{'digest':'wrong','fingerprint':'frozen'})
    with pytest.raises(RuntimeError,match='identity'):
        evaluator.evaluate(Candidate('n'))
