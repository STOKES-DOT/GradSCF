"""Fixed-field potential models must avoid the energy/density derivative path."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks
from gradscf.integrals.grids.blocks import AOGrid
from gradscf.model.autoxc import Candidate, Parameter
from gradscf.tools.features import _spin_density_and_gradient
from gradscf.model.autoxc.workflows.xc100.molecule import XC100Case, SphericalGrid


def case():
    grid = AOGrid(jnp.array([[.1*i,.2,.07*i] for i in range(7)]),jnp.linspace(.1,.3,7),
        jnp.array([[0.,0.,0.],[.5,0.,.1]]),jnp.array([[.7],[1.1]]),
        jnp.ones((2,1)),(1,1),((0,0,0),)*2,3)
    result = object.__new__(XC100Case)
    result.floor, result.max_parameters = 1e-12, 4
    result.system_id, result.grid, result.transform = 'toy', grid, jnp.eye(2)
    result.target = iks.Target(jnp.diag(jnp.array([2.,0.])),jnp.eye(2),2)
    result.vxc = jnp.zeros((2,2))
    ao, derivative, _, _, _ = grid.block(0,grid.ngrids,deriv=1)
    rho, gradient = _spin_density_and_gradient(ao,derivative,result.target.density)
    result.arrays = {'rho':np.asarray(rho),'grad_rho':np.asarray(gradient)}
    result.problem = iks.WuYang(result.target,jnp.eye(2),jnp.diag(jnp.array([-1.,1.])),
                               baseline=result.vxc)
    result.reporter = iks.WuYang(result.target,jnp.eye(2),result.problem.base_fock,
                                baseline=result.vxc,grid=SphericalGrid(grid,jnp.eye(2)))
    result.timings = {}
    return result


def test_direct_binding_matches_legacy_quadrature_and_parameter_fd(monkeypatch):
    model = case()
    candidate = Candidate('n**0.3333333333333333*(a*u+b*u**2)',
        parameters=(Parameter('a',0.,-.2,.2),Parameter('b',0.,-.2,.2)),kind='potential')
    point = {'a':jnp.asarray(.01),'b':jnp.asarray(-.005)}
    legacy = model.candidate_matrix(candidate)
    expected = legacy(point)
    def fail(*args,**kw): raise AssertionError('direct binding used the old energy/AO graph')
    monkeypatch.setattr(model,'candidate_matrix',fail)
    bound = model.bind(candidate)['toy']
    np.testing.assert_allclose(bound.potential(point),expected,atol=1e-12)
    vector = jnp.array([.01,-.005])
    loss, gradient = bound.loss_and_grad(vector)
    def objective(x):
        return -model.problem.score(legacy(dict(zip(('a','b'),x)))).reward
    step=1e-4
    fd=jnp.asarray([(objective(vector+step*v)-objective(vector-step*v))/(2*step)
                    for v in jnp.eye(2)])
    np.testing.assert_allclose(loss,objective(vector),atol=1e-10)
    np.testing.assert_allclose(gradient,fd,atol=1e-7,rtol=1e-6)
    solved=bound(point)
    diagnostic=bound.diagnose_result(solved)
    np.testing.assert_allclose(diagnostic.density_error,
        model.reporter.score(expected).density_error,atol=1e-13)


def test_uniform_padding_cannot_create_nonfinite_active_fields():
    from gradscf.model.autoxc.workflows.xc100.direct import fixed_fields, field_program
    rho=jnp.asarray([.1,.2,.3]);gradient=jnp.ones((3,3))*.02
    chunks=fixed_fields(rho,gradient,floor=1e-12,block_size=8)
    candidate=Candidate('a*u/n',parameters=(Parameter('a',0.,-.2,.2),),kind='potential')
    values=field_program(candidate)(chunks[0])
    assert values.shape==(8,5) and np.isfinite(values).all()
    np.testing.assert_allclose(values[3:],0.,atol=1e-14)


@pytest.mark.parametrize('candidate',[
    Candidate('a*n*u',parameters=(Parameter('a',0.,-.2,.2),),kind='energy'),
    Candidate('a*n*z',parameters=(Parameter('a',0.,-.2,.2),),kind='potential'),
])
def test_potential_campaign_rejects_energy_and_laplacian_candidates(candidate):
    from gradscf.model.autoxc.workflows.xc100.campaign import validate_candidate
    with pytest.raises(ValueError): validate_candidate(candidate)
