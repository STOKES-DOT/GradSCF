"""Deterministic tests of the XC swarm, with a stand-in for expensive scoring."""
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json

import pytest

from gradscf.model.autoxc import Candidate, Evaluation, Parameter


def test_model_owns_xc100_workflow_assets_and_swarm_api():
    from importlib.resources import files
    from gradscf.model.autoxc import swarm, workflows

    assert swarm.run_swarm.__module__ == 'gradscf.model.autoxc.swarm.search'
    assert workflows.xc100.design_campaign.__module__ == 'gradscf.model.autoxc.workflows.xc100.distributed'
    assets = files(workflows.xc100)
    assert sorted(p.name for p in (assets / 'prompts').iterdir()) == ['design.md', 'system.md']
    campaign = json.loads((assets / 'swarm_campaign.json').read_text())
    assert campaign['unbounded_waits'] and campaign['max_seconds'] is None


def test_six_genes_compile_one_candidate_with_shared_and_private_parameters():
    from gradscf.model.autoxc.swarm.genome import Genome, SLOTS

    weight = Parameter('weight', .2, 0., 1.)
    scale = Parameter('scale', .4, .1, 1.)
    genes = {k: Candidate('0') for k in SLOTS}
    genes['x_local'] = Candidate('weight*n+scale*n', (weight, scale))
    genes['c_local'] = Candidate('weight*n**2+scale*n', (weight, scale))
    genome = Genome(genes, shared=('scale',))
    full = genome.compile({'n'})
    assert len(full.parameters) == 3
    assert set(full.features) == {'n'}
    assert float(full.evaluate(full.initial, {'n': 2.})) == pytest.approx(2.8)
    warmed = genome.with_coefficients({**full.initial, 'shared_scale': .8})
    assert warmed.structure == genome.structure
    assert warmed.compile({'n'}).initial['shared_scale'] == .8

    invalid = genome.replace_gene('x_local', Candidate('s'))
    with pytest.raises(ValueError, match='x_local'):
        invalid.compile({'n', 's'})
    invalid = genome.replace_gene('c_nonlocal', Candidate('response'))
    with pytest.raises(ValueError, match='c_nonlocal'):
        invalid.compile({'n', 's'})
    assert invalid.compile({'n', 's', 'response'}).features == {'n', 'response'}

    donor = genome.replace_gene('c_local', Candidate('weight*n**3+scale*n', (weight, scale)))
    donor = donor.replace_gene('x_local', Candidate('weight*n**2+scale*n', (weight, scale)))
    child = genome.cross(donor, ('c_local',))
    assert child.genes['x_local'] == genes['x_local']
    assert child.genes['c_local'] == donor.genes['c_local']
    assert child.structure not in (genome.structure, donor.structure)
    alias = genes['x_local']
    alias = replace(alias, formula='b*n+scale*n', parameters=(replace(weight, name='b'), scale))
    assert genome.replace_gene('x_local', alias).structure == genome.structure
    alternative = genome.replace_gene('c_semilocal', Candidate('.02*tanh(q)'))
    assert alternative.replace_gene('c_semilocal', Candidate('.03*tanh(q)')).structure == alternative.structure
    assert alternative.replace_gene('c_semilocal', Candidate('.02*tanh(s)')).structure != alternative.structure


def test_compiled_genes_are_fitted_together_with_the_real_ad_evaluator():
    import jax.numpy as jnp
    from gradscf.dft.iks import WuYangScore
    from gradscf.model.autoxc import Evaluator
    from gradscf.model.autoxc.swarm.genome import Genome, SLOTS

    genes = {k: Candidate('0') for k in SLOTS}
    weight = Parameter('a', 0., -1., 1.)
    genes['x_local'] = Candidate('a*n', (weight,))
    genes['c_local'] = Candidate('a*n**2', (weight,))
    candidate = Genome(genes).compile({'n'})

    def bind(candidate):
        def sample(n, target):
            def score(parameters):
                reward = -(candidate.evaluate(parameters, {'n': n}) - target)**2
                return WuYangScore(0., 0., reward, 0., jnp.eye(2), jnp.eye(2), 1., 0., 0., True)
            return score
        return {'one': sample(1., .5), 'two': sample(2., 1.6)}

    evaluator = Evaluator(bind, sample_ids=('one', 'two'), fingerprint='joint-fit',
                          maxiter=100, gradient_only=True, require_convergence=True)
    fit = evaluator.evaluate(candidate)
    assert fit.valid
    assert fit.parameters == pytest.approx({'x_local_p_a': .2, 'c_local_p_a': .3}, abs=1e-6)


def test_ppo_increases_probability_of_rewarded_action():
    import numpy as np
    from gradscf.model.autoxc.swarm.ppo import MutationPolicy, Transition

    policy = MutationPolicy(3, 3)
    state = [1., 0., 1.]
    mask = [True, True, False]
    before = policy.probabilities(state, mask)
    row = Transition(state, mask, 1, float(np.log(before[1])), 1.)
    policy.update([row])
    after = policy.probabilities(state, mask)
    assert after[1] > before[1] and after[2] == 0.


@pytest.mark.parametrize('ga', [False, True])
@pytest.mark.parametrize('ppo', [False, True])
def test_swarm_switch_and_budget_and_resume(tmp_path, ga, ppo):
    if ga:
        pytest.importorskip('multi_agent_ga')
    from gradscf.model.autoxc.swarm.genome import Genome, SLOTS
    from gradscf.model.autoxc.swarm.search import SwarmConfig, run_swarm

    genes = {k: Candidate('0') for k in SLOTS}
    genes['x_local'] = Candidate('a*n', (Parameter('a', -.8, -1., 0.),))
    genomes = [Genome(genes)]
    assessments, feedbacks = [], []

    class Scorer:
        configuration = {'fingerprint': 'test-v1',
                         'worker': {'execution': {'cpu_masks': [list(range(8)), list(range(8, 32))]}}}

        def evaluate(self, c):
            assessments.append(c)
            reward = float(c.evaluate(c.initial, {'n': 1., 's': 1.}))
            return Evaluation(reward, True, c.initial, {'001': {
                'density_l1': .01, 'density_error': .005, 'reward': reward,
            }}, fit={'evaluation_seconds': .25})

    @contextmanager
    def factory(slot, session_id):
        class Agent:
            session_id = f'thread-{slot}'

            def propose(self, data):
                feedbacks.append((slot, data))
                local = data['action']['slot'].endswith('_local')
                feature = 'n' if local else 's'
                p = Parameter('weight', .2 + slot / 10, 0., 1.)
                action = data['action']['operation']
                expression = '0' if action == 'remove' else f'weight*{feature}**{slot+2}'
                return Candidate(expression, () if action == 'remove' else (p,)).to_dict()

        yield Agent()

    config = SwarmConfig(agents=3, generations=2, steps=3, ga=ga, ppo=ppo,
                         crossover_probability=1., seed=13)
    kwargs = dict(directory=tmp_path, config=config, registry={'n', 's'},
                  protocol={'experiment': 'swarm-test'})
    result = asyncio.run(run_swarm(factory, Scorer(), genomes, **kwargs))
    assert result['attempts'] == 18
    assert result['scheduling']['allocated_threads'] == 32
    assert result['scheduling']['agent_concurrency'] == 2
    for item in result['scheduling']['per_agent']:
        assert item['segments'] == 2
        assert item['agent_queue_seconds'] >= 0.
        assert item['allocated_cpu_seconds'] == 8 * item['measured_fits'] or not item['measured_fits']
    assert result['best']['evaluation']['valid']
    assert result['best']['evaluation']['reward'] == max(
        r['evaluation']['reward'] for r in result['trials'] if r.get('evaluation')
    )
    if ga:
        assert any(row['operation'] == 'crossover' for row in result['trials'])
    else:
        assert all(row['operation'] == 'mutation' for row in result['trials'])
        for slot, data in feedbacks:
            assert all(r['agent'] == slot for r in data['history'])
    before_calls = len(assessments), len(feedbacks)
    assert asyncio.run(run_swarm(factory, Scorer(), genomes, **kwargs)) == result
    assert (len(assessments), len(feedbacks)) == before_calls
    assert json.loads((tmp_path / 'swarm.json').read_text()) == result


@pytest.mark.parametrize('frozen', [False, True])
def test_swarm_stores_on_policy_trajectory_and_frozen_policy(tmp_path, frozen):
    from gradscf.model.autoxc.swarm.genome import Genome, SLOTS
    from gradscf.model.autoxc.swarm.ppo import MutationPolicy
    from gradscf.model.autoxc.swarm.search import ACTIONS, STATE_SIZE, SwarmConfig, run_swarm

    genes = {key: Candidate('0') for key in SLOTS}
    genes['x_local'] = Candidate('-n')
    seed = Genome(genes)

    class Scorer:
        configuration = {'fingerprint': 'ppo-test'}

        def evaluate(self, candidate):
            reward = float(candidate.evaluate(candidate.initial, {'n': 1., 's': 1.}))
            return Evaluation(reward, True, {})

    @contextmanager
    def factory(slot, session_id):
        class Designer:
            session_id = 'test'

            def propose(self, data):
                feature = 'n' if data['action']['slot'].endswith('_local') else 's'
                return Candidate(f'{feature}**2').to_dict()

        yield Designer()

    initial = MutationPolicy(STATE_SIZE, len(ACTIONS)).checkpoint()
    result = asyncio.run(run_swarm(
        factory, Scorer(), [seed], directory=tmp_path,
        config=SwarmConfig(agents=1, generations=1, steps=2, ppo=True),
        registry={'n', 's'}, protocol={}, frozen_policy=initial if frozen else None,
    ))
    segment = json.loads((tmp_path/'agents/000/000001.json').read_text())
    assert result['attempts'] == len(segment['trajectory']) == 2
    assert result['trials'][0]['reward'] == result['trials'][0]['evaluation']['reward']+1.
    for row in segment['trajectory']:
        assert row['mask'][row['action']]
    assert (segment['policy'] == initial) is frozen


def test_old_single_agent_entry_keeps_original_dispatch(monkeypatch):
    from pathlib import Path
    import runpy
    import sys
    import gradscf.model.autoxc.workflows.xc100.distributed as distributed

    called = {}
    def design(*args, **kwargs):
        called.update(kwargs)
        return {'records': [], 'incumbent': {'reward': 0.}}

    monkeypatch.setattr(distributed, 'design_campaign', design)
    script = Path('examples/iks/xc100_open_xc.py')
    monkeypatch.setattr(sys, 'argv', [str(script), '--remote', '/sample',
                                     '--output', '/sample', '--job-id', '1', '--rounds', '2'])
    runpy.run_path(str(script), run_name='__main__')
    assert called['rounds'] == 2 and called['candidate_mode'] == 'standalone_potential'


def test_swarm_cli_passes_agent_count_and_releases_worker(tmp_path, monkeypatch):
    from pathlib import Path
    import runpy
    import sys
    import gradscf.model.autoxc.workflows.xc100.distributed as distributed
    import gradscf.model.autoxc.swarm.search as swarm

    calls = {}

    class Worker:
        ready = {'policy': {'kind': 'potential', 'potential_is_complete': True,
                            'unbounded_waits': True, 'features': ['n']}}
        configuration = {'max_seconds': None}

        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def wait_ready(cls, *args, **kwargs):
            assert kwargs['timeout'] is None
            return cls()

        def release(self):
            calls['released'] = True

    async def search(*args, **kwargs):
        calls['config'] = kwargs['config']
        calls['registry'] = kwargs['registry']
        return {'attempts': 6, 'best': {'evaluation': {'reward': .1}}}

    monkeypatch.setattr(distributed, 'RemoteEvaluator', Worker)
    monkeypatch.setattr(swarm, 'run_swarm', search)
    script = Path('examples/iks/xc100_swarm.py')
    monkeypatch.setattr(sys, 'argv', [
        str(script), '--remote', '/worker', '--output', str(tmp_path), '--job-id', '42',
        '--agents', '3', '--generations', '1', '--steps', '2', '--no-ga', '--ppo',
    ])
    runpy.run_path(str(script), run_name='__main__')
    assert calls['released'] and calls['registry'] == ['n']
    assert calls['config'].agents == 3 and calls['config'].steps == 2
    assert not calls['config'].ga and calls['config'].ppo
