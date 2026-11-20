"""Infrastructure errors stop a long run; physical rejections still give feedback."""
import json
from types import SimpleNamespace
import pytest

from gradscf.model.autoxc import Evaluation, run


def test_fail_fast_does_not_burn_all_rounds_on_a_dead_model(tmp_path):
    calls = []
    class Broken:
        session_id = 'thread'
        def propose(self, feedback):
            calls.append(1)
            raise TimeoutError('network')
    evaluator = SimpleNamespace(configuration={'fingerprint':'test'},evaluate=lambda c:None)
    with pytest.raises(RuntimeError,match='stopped'):
        run(Broken(),evaluator,directory=tmp_path,rounds=50,fail_fast=True)
    assert calls == [1]
    assert len((tmp_path/'history.jsonl').read_text().splitlines()) == 1


def test_physical_rejection_reason_reaches_the_next_proposal(tmp_path):
    feedback = []
    class Designer:
        session_id = None
        def propose(self, value):
            feedback.append(value)
            return dict(formula='n',parameters=[],kind='potential',hypothesis='test')
    evaluator = SimpleNamespace(configuration={'fingerprint':'test'},
        evaluate=lambda c: Evaluation(None,False,error='Uniform limit failed'))
    run(Designer(),evaluator,directory=tmp_path,rounds=2,fail_fast=True)
    assert feedback[1]['history'][0]['error'] == 'Uniform limit failed'
