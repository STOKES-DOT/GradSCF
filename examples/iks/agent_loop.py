"""Replay the design/evaluate/reward loop on a manufactured two-AO problem.

This checks infrastructure, not molecular accuracy or a new PBE parametrization.
Only the proposal source is replayed: every fit and WY score is computed live.
All physics and control logic are imported from src; there is no CLI.
"""
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from gradscf.dft import iks
from gradscf.model import autoxc

jax.config.update('jax_enable_x64', True)

# A small orthonormal quadrature makes the target and energy AD independently
# inspectable. It represents no particular molecule or physical basis set.
ao, _ = jnp.linalg.qr(jnp.array([[1., .2], [.3, .8], [.5, -.2], [.1, .4]]))
weights = jnp.ones(4)
molecule = SimpleNamespace(ao=ao, grid=SimpleNamespace(weights=weights))
target = iks.Target(jnp.diag(jnp.array([2., 0.])), jnp.eye(2), 2)


def inputs(physical):
    return {'n': physical.rho, 'weights': physical.weights}


# Plant a known scalar XC energy so the exact optimum is available for checking.
teacher = autoxc.Candidate('a*n**2', kind='energy',
    parameters=(autoxc.Parameter('a', .2, -.5, .5),)).functional(inputs)
teacher_potential = jax.grad(lambda d: teacher.energy_for_density({'a': .2}, molecule, d))(
    target.density)
base_fock = jnp.diag(jnp.array([-1., 1.]))-teacher_potential
problem = iks.WuYang(target, jnp.eye(2), base_fock, baseline=jnp.zeros((2, 2)),
                    grid=(ao, weights))


def bind(candidate):
    functional = candidate.functional(inputs)
    def score(params):
        return problem.assess(lambda p, d: functional.energy_for_density(p, molecule, d), params)
    return {'two_ao': score}


evaluator = autoxc.Evaluator(bind, sample_ids=('two_ao',),
                           fingerprint='two-ao-quadrature-v1-zero-xc-baseline')


class ReplayDesigner:
    """A recorded proposal sequence exercises the loop without an LLM call."""
    session_id = None

    def propose(self, feedback):
        power = (2, 3, 4)[feedback['round']-1]
        return autoxc.Candidate(f'a*n**{power}', kind='energy',
            parameters=(autoxc.Parameter('a', 0., -.5, .5),),
            hypothesis=f'Test a density power of {power}.').to_dict()


directory = Path('artifacts/iks-agent-loop-replay')
result = autoxc.run(ReplayDesigner(), evaluator, directory=directory, rounds=3,
    context='Manufactured energy-density search. No molecular or PBE claim.')

for record in result['records']:
    report = record['evaluation']
    print('round', record['round'], record['candidate']['formula'],
          'reward', round(report['reward'], 8), 'accepted', record['accepted'])
print('winner:', result['incumbent']['candidate']['formula'])
print('parameters:', result['incumbent']['parameters'])
print('records:', directory/'history.jsonl')

# For a live design session, replace ReplayDesigner with the optional
# context-managed autoxc.CodexDesigner(cwd=..., model=..., effort=...). The same
# evaluator and Python loop remain; the agent never implements the evaluator.

# CPU/float64 output (2026-11-16):
# round 1 a*n**2 reward 1.52269531 accepted True
# round 2 a*n**3 reward 1.52269531 accepted False
# round 3 a*n**4 reward 1.52269531 accepted False
# winner: a*n**2
# parameters: {'a': 0.1999999999999991}
# Different powers can reach the same projector on this two-AO sample. This
# exercise does not identify a unique functional or establish molecular accuracy.
