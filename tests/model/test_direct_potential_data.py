"""Opt-in XC100 GPU acceptance; no dataset is required for ordinary unit tests."""
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.model.autoxc import Candidate, Evaluator, Parameter
from gradscf.model.autoxc.history import source_digest, versions
from gradscf.model.autoxc.workflows.xc100.campaign import write_json, memory
from gradscf.model.autoxc.workflows.xc100.direct import BLOCK_SIZE
from gradscf.model.autoxc.workflows.xc100.molecule import XC100Case
from gradscf.model.autoxc.workflows.xc100.reference import reference_baseline


ROOT = os.environ.get('GRADSCF_XC100_ACCEPTANCE')
pytestmark = pytest.mark.skipif(not ROOT, reason='Opt-in qualified XC100 GPU dataset')


def timed(function):
    start = time.perf_counter()
    value = jax.block_until_ready(function())
    return value, time.perf_counter()-start


@pytest.mark.parametrize('identifier', ('010','017','065'))
def test_direct_potential_real_data(identifier):
    assert jax.default_backend()=='gpu' and len(jax.devices())==1
    assert jax.config.jax_enable_x64 and len(os.sched_getaffinity(0))<=64
    root = Path(ROOT)
    case = XC100Case(root/'data/xc100',identifier,block_size=BLOCK_SIZE)
    case.prepare(baseline=reference_baseline(root,case))
    parameters = tuple(Parameter(name,0.,-.2,.2) for name in 'abcd')
    candidate = Candidate('n**0.3333333333333333*(a*u+b*u**2+c*u**3+d*u**4)',
                          parameters=parameters,kind='potential')
    point = dict(zip('abcd',map(jnp.asarray,(.01,-.005,.002,-.001))))
    zeros = dict.fromkeys('abcd',jnp.asarray(0.))
    legacy = jax.jit(case.candidate_matrix(candidate))
    expected, legacy_cold = timed(lambda: legacy(point))
    _, legacy_warm = timed(lambda: legacy(point))
    directions, legacy_directions = timed(lambda: jax.jit(jax.jacfwd(legacy))(zeros))
    binding, binding_seconds = timed(lambda: case.bind(candidate)[identifier])
    actual = binding.potential(point)
    np.testing.assert_allclose(actual,case.vxc+expected,atol=2e-11,rtol=1e-9)
    for name in 'abcd':
        np.testing.assert_allclose(binding.directions[name],directions[name],atol=2e-11,rtol=1e-9)
    vector = jnp.asarray([point[name] for name in 'abcd'])
    (loss,gradient), score_cold = timed(lambda: binding.loss_and_grad(vector))
    _, score_warm = timed(lambda: binding.loss_and_grad(vector*1.001))
    expected_score = case.problem.score(case.vxc+expected)
    assert bool(expected_score.valid)
    np.testing.assert_allclose(loss,-expected_score.reward,atol=2e-9,rtol=1e-8)
    step=1e-4
    plus,minus = dict(point),dict(point)
    plus['a'],minus['a'] = point['a']+step,point['a']-step
    fd=-(case.problem.score(case.vxc+legacy(plus)).reward-
         case.problem.score(case.vxc+legacy(minus)).reward)/(2*step)
    np.testing.assert_allclose(gradient[0],fd,atol=2e-6,rtol=2e-5)
    diagnosed = binding.diagnose_result(binding(point))
    expected_diagnostic = case.reporter.score(case.vxc+expected)
    assert bool(diagnosed.valid)
    assert np.isfinite(float(diagnosed.density_l1)) and diagnosed.density_l1>=0
    np.testing.assert_allclose(diagnosed.density_error,expected_diagnostic.density_error,
                               atol=2e-10,rtol=1e-8)
    # Another formula uses the same projection/score programs, with new arrays.
    changed = Candidate('n**0.3333333333333333*(a*u+b*u**2+c*u**3+d*u**5)',
                        parameters=parameters,kind='potential')
    second, second_binding = timed(lambda: case.bind(changed)[identifier])
    _, second_score = timed(lambda: second.loss_and_grad(vector))
    probe = Candidate('a*n**0.3333333333333333*u',kind='potential',
                      parameters=(Parameter('a',0.,-.2,.2),))
    checked = Evaluator(case.bind,sample_ids=(identifier,),fingerprint='gradient-acceptance',
        maxiter=100,gradient_tolerance=1e-6,gradient_only=True,require_convergence=True,
        streaming=True).evaluate(probe)
    assert checked.valid and checked.fit['gradient_converged']
    assert checked.fit['projected_gradient_max']<=1e-6
    write_json(root/'validation'/f'potential-{identifier}.json',{
        'passed':True,'system_id':identifier,'nao':case.target.density.shape[0],
        'grid_points':case.grid.ngrids,'block_size':BLOCK_SIZE,'source':source_digest(),
        'device':jax.devices()[0].device_kind,'versions':versions(),
        'job_id':os.environ['SLURM_JOB_ID'],'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'matrix_max_abs_error_hartree':float(jnp.max(jnp.abs(actual-case.vxc-expected))),
        'gradient_fd_abs_error':float(jnp.abs(gradient[0]-fd)),
        'reward':float(-loss),'density_error':float(diagnosed.density_error),
        'density_l1':float(diagnosed.density_l1),'strict_fit':checked.fit,
        'seconds':{'legacy_matrix_cold':legacy_cold,'legacy_matrix_warm':legacy_warm,
            'legacy_directions':legacy_directions,'direct_binding_cold':binding_seconds,
            'score_gradient_cold':score_cold,'score_gradient_warm':score_warm,
            'second_formula_binding':second_binding,'second_formula_score':second_score},
        **memory()})
