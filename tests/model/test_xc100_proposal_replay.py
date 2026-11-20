"""Source repairs replay formulas, never stale scores or extra model calls."""
def test_saved_proposals_are_replayed_before_new_model_calls():
    from gradscf.model.autoxc.workflows.xc100.distributed import ReplayProposals
    class Designer:
        session_id='existing-thread'
        configuration={'model':'configured'}
        def __init__(self):self.calls=[]
        def propose(self,feedback):self.calls.append(feedback['round']);return {'new':True}
        def resume(self,session):self.session_id=session
    inner=Designer();seed={'proposals':[{'first':1},{'second':2}],
        'session_id':'existing-thread','reason':'source repair; scores recomputed'}
    proxy=ReplayProposals(inner,seed)
    assert proxy.propose({'round':1})=={'first':1}
    assert proxy.propose({'round':2})=={'second':2}
    assert inner.calls==[]
    assert proxy.propose({'round':3})=={'new':True} and inner.calls==[3]
    assert proxy.session_id=='existing-thread'
    proxy.resume('resumed');assert proxy.session_id=='resumed'
    assert proxy.configuration['replayed_proposals']==2


def test_malformed_replay_checkpoint_still_releases_the_worker(tmp_path,monkeypatch):
    import pytest
    import gradscf.model.autoxc.workflows.xc100.distributed as module
    released=[]
    class Guardian:
        def __init__(self,*a,**kw):pass
        def release(self):released.append(True)
    monkeypatch.setattr(module,'RemoteEvaluator',Guardian)
    (tmp_path/'seed_proposals.json').write_text('{broken')
    with pytest.raises(ValueError):module.design_campaign('host','/project',output=tmp_path,brief='test')
    assert released==[True]


def test_design_campaign_replays_two_formulas_and_resumes_existing_thread(tmp_path,monkeypatch):
    import json
    import gradscf.model.autoxc.workflows.xc100.distributed as module
    from gradscf.model import autoxc
    calls=[];released=[];sessions=[]
    def payload(i):return {'formula':str(i),'kind':'potential','parameters':[],'hypothesis':'replay test'}
    (tmp_path/'seed_proposals.json').write_text(json.dumps({'proposals':[payload(1),payload(2)],
        'session_id':'saved-thread','reason':'source repair'}))
    class Worker:
        ready={'policy':{'kind':'potential'}}
        configuration={'fingerprint':'replay-controller'}
        def __init__(self,*a,**kw):pass
        @classmethod
        def wait_ready(cls,*a,**kw):return cls()
        def evaluate(self,candidate):return autoxc.Evaluation(float(candidate.formula),True)
        def release(self):released.append(True)
    class Designer:
        configuration={'model':'configured'}
        def __init__(self,**kw):self.session_id=kw['session_id'];sessions.append(self.session_id)
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def propose(self,feedback):calls.append(feedback['round']);return payload(3)
        def resume(self,session):self.session_id=session
    monkeypatch.setattr(module,'RemoteEvaluator',Worker)
    monkeypatch.setattr(autoxc,'CodexDesigner',Designer)
    result=module.design_campaign('host','/project',output=tmp_path,brief='test',rounds=3)
    assert calls==[3] and sessions==['saved-thread'] and released==[True]
    assert len(result['records'])==3
