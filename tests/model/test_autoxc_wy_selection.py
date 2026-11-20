"""Strict iKS selection never trades a lower WY value for density improvement."""
import json

import pytest

from gradscf.model.autoxc import Evaluation


def score(reward,l1=.02,l2=.1):
    return Evaluation(reward,True,samples={'one':{'density_l1':l1,'density_error':l2}})


def test_wy_selection_keeps_density_secondary_and_rejects_wy_regression():
    from gradscf.model.autoxc.selection import WYSelection
    policy=WYSelection(density_scales=(.02,.01),tolerance=1e-6)
    incumbent={'reward':.1,**policy.state(score(.1))}
    assert policy.better(score(.2,.2,1.),incumbent)
    assert not policy.better(score(.1-1e-8,.001,.001),incumbent)
    assert policy.better(score(.1+1e-8,.01,.05),incumbent)
    assert not policy.better(score(.1+1e-8,.04,.2),incumbent)


def test_wy_selection_requires_measured_finite_density_diagnostics():
    from gradscf.model.autoxc.selection import WYSelection
    policy=WYSelection(density_scales=(.02,.01))
    with pytest.raises(ValueError):policy.state(score(.1,float('nan'),.1))


def test_run_records_and_restores_the_selection_policy(tmp_path):
    from gradscf.model.autoxc import run
    from gradscf.model.autoxc.selection import WYSelection
    class Designer:
        session_id='selection-replay'
        def __init__(self,start):self.number=start
        def propose(self,feedback):
            self.number+=1
            return {'formula':str(self.number),'kind':'potential','parameters':[],'hypothesis':'test'}
    class Evaluator:
        configuration={'fingerprint':'selection-replay'}
        def evaluate(self,candidate):
            return {1:score(.1),2:score(.1+1e-8,.01,.05),3:score(.1-1e-8,.001,.001)}[int(candidate.formula)]
    policy=WYSelection(density_scales=(.02,.01),tolerance=1e-6)
    run(Designer(0),Evaluator(),directory=tmp_path,rounds=2,selection=policy)
    result=run(Designer(2),Evaluator(),directory=tmp_path,rounds=3,selection=policy)
    assert [r['accepted'] for r in result['records']]==[True,True,False]
    assert result['incumbent']['candidate']['formula']=='2'
    assert json.loads((tmp_path/'config.json').read_text())['selection']==policy.configuration
