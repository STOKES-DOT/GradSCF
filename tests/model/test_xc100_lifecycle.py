"""Long-run readiness and resource release must survive every controller exit."""
import pytest

from gradscf.model.autoxc.history import source_digest


def test_stale_ready_does_not_override_a_terminal_worker(tmp_path,monkeypatch):
    from gradscf.model.autoxc.workflows.xc100.distributed import RemoteEvaluator
    ready={'baseline_count':100,'source':source_digest(),'job_id':'old',
           'configuration':{'fingerprint':'frozen'}}
    def read(self,name):
        return ready if name=='ready.json' else {'phase':'idle_timeout','job_id':'old'}
    monkeypatch.setattr(RemoteEvaluator,'read',read)
    with pytest.raises(RuntimeError,match='stopped'):
        RemoteEvaluator.wait_ready('host','/project',receipt_directory=tmp_path,timeout=.1)


def test_ready_from_another_job_is_not_accepted_while_preparing(tmp_path,monkeypatch):
    from gradscf.model.autoxc.workflows.xc100.distributed import RemoteEvaluator
    ready={'baseline_count':100,'source':source_digest(),'job_id':'old',
           'configuration':{'fingerprint':'frozen'}}
    statuses=iter([{'phase':'preparing_baselines','job_id':'new'},
                   {'phase':'failed','job_id':'new','error':'probe'}])
    monkeypatch.setattr(RemoteEvaluator,'read',lambda self,name:ready if name=='ready.json' else next(statuses))
    monkeypatch.setattr('time.sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='stopped'):
        RemoteEvaluator.wait_ready('host','/project',receipt_directory=tmp_path,timeout=1.)


def test_controller_releases_even_when_preparation_or_sdk_fails(tmp_path,monkeypatch):
    import gradscf.model.autoxc.workflows.xc100.distributed as module
    released=[]
    class Guard:
        def __init__(self,*a,**kw): pass
        @classmethod
        def wait_ready(cls,*a,**kw): raise RuntimeError('preparation failed')
        def release(self): released.append(True)
    monkeypatch.setattr(module,'RemoteEvaluator',Guard)
    with pytest.raises(RuntimeError,match='preparation failed'):
        module.design_campaign('host','/project',output=tmp_path,brief='test')
    assert released==[True]


def test_worker_honors_release_before_loading_data(tmp_path):
    from gradscf.model.autoxc.workflows.xc100.campaign import Campaign,CampaignReleased
    (tmp_path/'release.flag').touch()
    with pytest.raises(CampaignReleased): Campaign(tmp_path).prepare()


def test_fresh_worker_mailboxes_exist_when_ready_is_published(tmp_path, monkeypatch):
    from gradscf.model.autoxc.workflows.xc100.campaign import Campaign

    worker = Campaign(tmp_path)
    worker.rows = [{'system_id': '001', 'fingerprint': 'sample', 'density_error': 0.1}]
    monkeypatch.setattr(worker, 'validate_prepared', lambda: 0.0)

    def publish_request(phase):
        assert phase == 'ready'
        assert (tmp_path / 'ready.json').is_file()
        assert (tmp_path / 'outbox').is_dir()
        (tmp_path / 'inbox' / 'candidate.tmp').write_text('{}')

    monkeypatch.setattr(worker, 'status', publish_request)
    worker.finish_preparation(receipt=None)


def test_controller_releases_when_sdk_context_entry_fails(tmp_path,monkeypatch):
    import gradscf.model.autoxc.workflows.xc100.distributed as module
    from gradscf.model import autoxc
    released=[]
    class Guard:
        def __init__(self,*a,**kw): pass
        @classmethod
        def wait_ready(cls,*a,**kw): return cls()
        def release(self): released.append(True)
    class Designer:
        def __init__(self,*a,**kw): pass
        def __enter__(self): raise RuntimeError('sdk initialization failed')
        def __exit__(self,*a): pass
    monkeypatch.setattr(module,'RemoteEvaluator',Guard)
    monkeypatch.setattr(autoxc,'CodexDesigner',Designer)
    with pytest.raises(RuntimeError,match='sdk initialization failed'):
        module.design_campaign('host','/project',output=tmp_path,brief='test')
    assert released==[True]
