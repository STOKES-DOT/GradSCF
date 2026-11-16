"""The controller, not the model, owns evaluation, acceptance and restart."""
from dataclasses import replace
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks


def payload(formula='a*n'):
    return {'formula': formula, 'kind': 'potential', 'hypothesis': 'Reduce the occupied/virtual coupling.',
            'parameters': [{'name': 'a', 'value': .1, 'lower': -.5, 'upper': .5}]}


def evaluator():
    from gradscf.model.autoxc import Evaluator
    target = iks.Target(jnp.diag(jnp.array([2., 0.])), jnp.eye(2), 2)
    problem = iks.WuYang(target, jnp.eye(2), jnp.diag(jnp.array([-1., 1.])),
                        baseline=jnp.array([[0., .2], [.2, 0.]]))
    direction = jnp.array([[0., 1.], [1., 0.]])
    def bind(candidate):
        return {'sample': lambda p: problem.score(problem.baseline_potential+
                                                 candidate.evaluate(p, {'n': direction}))}
    return Evaluator(bind, sample_ids=('sample',), fingerprint='manufactured-v1')


def test_parameter_fit_uses_the_shared_ad_optimizer():
    from gradscf.model.autoxc import Candidate
    result = evaluator().evaluate(Candidate.from_json(payload()))
    assert result.valid and result.reward > 0
    np.testing.assert_allclose(result.parameters['a'], -.2, atol=1e-6)
    assert result.fit['success'] and result.fit['evaluations'] > 0


def test_failed_samples_cannot_be_dropped_from_a_batch():
    from gradscf.model.autoxc import Candidate, Evaluator
    good = iks.WuYangScore(1., 0., 1., 1., jnp.eye(2), jnp.eye(2), 1., 0., jnp.nan, True)
    bad = good._replace(valid=False, reward=jnp.nan)
    e = Evaluator(lambda c: {'ok': lambda p: good, 'bad': lambda p: bad},
                  sample_ids=('ok', 'bad'), fingerprint='batch')
    result = e.evaluate(Candidate('n'))
    assert not result.valid and result.reward is None
    assert set(result.samples) == {'ok', 'bad'}


def test_fixed_batch_identity_is_checked():
    from gradscf.model.autoxc import Candidate, Evaluator
    e = Evaluator(lambda c: {}, sample_ids=('required',), fingerprint='batch')
    with pytest.raises(ValueError, match='sample'):
        e.evaluate(Candidate('n'))


def test_validation_reward_is_separate_from_parameter_fitting():
    from gradscf.model.autoxc import Candidate, Evaluator
    def make_score(reward):
        return iks.WuYangScore(0., 0., reward, 0., jnp.eye(2), jnp.eye(2), 1., 0., jnp.nan, True)
    e = Evaluator(lambda c: {'train': lambda p: make_score(-(p['a']-.3)**2),
                            'validation': lambda p: make_score(-(p['a']+.2)**2)},
                  sample_ids=('train', 'validation'), fit_ids=('train',),
                  reward_ids=('validation',), fingerprint='split-v1')
    result = e.evaluate(Candidate.from_json(payload()))
    np.testing.assert_allclose(result.parameters['a'], .3, atol=1e-7)
    np.testing.assert_allclose(result.reward, -.25, atol=1e-7)


def test_invalid_fitting_point_is_rejected_instead_of_minimized_as_negative_infinity():
    from gradscf.model.autoxc import Candidate, Evaluator
    bad = iks.WuYangScore(0., 0., jnp.nan, 0., jnp.eye(2), jnp.eye(2), 0., 0., jnp.nan, False)
    e = Evaluator(lambda c: {'bad': lambda p: bad}, sample_ids=('bad',), fingerprint='invalid')
    result = e.evaluate(Candidate.from_json(payload()))
    assert not result.valid and result.reward is None
    assert not result.fit['success']


class Replay:
    session_id = 'replay'
    def __init__(self, proposals):
        self.proposals = iter(proposals)
        self.feedback = []
    def propose(self, feedback):
        self.feedback.append(feedback)
        return next(self.proposals)


def test_loop_evaluates_records_and_feeds_back(tmp_path):
    from gradscf.model.autoxc import run
    agent = Replay([payload(), payload('a*n+0')])
    summary = run(agent, evaluator(), directory=tmp_path, rounds=2, context='Manufactured pilot')
    rows = [json.loads(s) for s in (tmp_path/'history.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and rows[0]['accepted']
    assert agent.feedback[1]['incumbent']['parameters']['a'] == pytest.approx(-.2, abs=1e-6)
    assert 'sample' not in json.dumps(agent.feedback[1])
    assert summary['incumbent']['reward'] > 0
    assert rows[0]['candidate']['formula'] == 'a*n'
    assert rows[0]['evaluation']['samples']['sample']['valid']


def test_resume_restores_the_designers_saved_session(tmp_path):
    from gradscf.model.autoxc import run
    class Stateful(Replay):
        def resume(self, session_id): self.session_id = session_id
    run(Stateful([payload()]), evaluator(), directory=tmp_path, rounds=1)
    next_agent = Stateful([payload('a*n+0')])
    next_agent.session_id = 'unused-new-thread'
    result = run(next_agent, evaluator(), directory=tmp_path, rounds=2)
    assert next_agent.session_id == 'replay'
    assert result['records'][1]['parent'] == result['records'][0]['digest']


def test_duplicate_and_malformed_proposals_are_recorded(tmp_path):
    from gradscf.model.autoxc import run
    agent = Replay([payload(), payload(), {'broken': 1}])
    result = run(agent, evaluator(), directory=tmp_path, rounds=3)
    assert [r['status'] for r in result['records']] == ['evaluated', 'duplicate', 'invalid_proposal']
    assert result['records'][2]['raw_proposal'] == {'broken': 1}


def test_resume_does_not_redo_completed_rounds_or_change_rules(tmp_path):
    from gradscf.model.autoxc import run
    run(Replay([payload()]), evaluator(), directory=tmp_path, rounds=1)
    resumed = run(Replay([payload('a*n+0')]), evaluator(), directory=tmp_path, rounds=2)
    assert len(resumed['records']) == 2
    e = evaluator()
    e.fingerprint = 'changed-dataset'
    with pytest.raises(ValueError, match='configuration|fingerprint'):
        run(Replay([]), e, directory=tmp_path, rounds=3)


def test_completed_proposal_survives_interruption_during_evaluation(tmp_path):
    from gradscf.model.autoxc import run
    e = evaluator()
    real = e.evaluate
    e.evaluate = lambda c: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run(Replay([payload()]), e, directory=tmp_path, rounds=1)
    e.evaluate = real
    result = run(Replay([]), e, directory=tmp_path, rounds=1)
    assert len(result['records']) == 1 and result['records'][0]['status'] == 'evaluated'


def test_transport_failure_preserves_the_incumbent_and_is_not_a_score(tmp_path):
    from gradscf.model.autoxc import run
    class Broken:
        session_id = None
        def propose(self, feedback): raise TimeoutError('test timeout')
    result = run(Broken(), evaluator(), directory=tmp_path, rounds=1)
    assert result['records'][0]['status'] == 'proposal_error'
    assert result['records'][0]['evaluation'] is None
    assert result['incumbent']['reward'] == 0.


def test_first_turn_error_keeps_the_saved_sdk_session(tmp_path):
    from gradscf.model.autoxc import run
    class Broken:
        session_id = 'first-sdk-thread'
        def propose(self, feedback): raise TimeoutError('test timeout')
    result = run(Broken(), evaluator(), directory=tmp_path, rounds=1)
    assert result['records'][0]['session_id'] == 'first-sdk-thread'


def test_interrupted_proposal_checkpoints_the_current_sdk_session(tmp_path):
    from gradscf.model.autoxc import run
    class Interrupted:
        session_id = 'before'
        def propose(self, feedback):
            self.session_id = 'started-thread'
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run(Interrupted(), evaluator(), directory=tmp_path, rounds=1)
    assert json.loads((tmp_path/'pending.json').read_text())['session_id'] == 'started-thread'
