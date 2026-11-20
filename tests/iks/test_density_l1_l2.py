"""L1 and squared L2 density diagnostics share the same physical quadrature."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.dft import iks


def test_density_l1_and_l2_match_weighted_integrals():
    from gradscf.dft.iks.response import density_errors
    target=iks.Target(jnp.diag(jnp.array([2.,0.])),jnp.eye(2),2)
    ao=jnp.array([[1.,0.],[.5,.4],[.2,1.]])
    weights=jnp.array([.2,.6,.4]);density=jnp.array([[1.8,.1],[.1,.2]])
    actual=density_errors(target,density,(ao,weights))
    rho=jnp.einsum('gi,ij,gj->g',ao,target.density,ao)
    delta=jnp.einsum('gi,ij,gj->g',ao,density-target.density,ao)
    np.testing.assert_allclose(actual.l1,jnp.sum(weights*jnp.abs(delta))/jnp.sum(weights*jnp.abs(rho)),atol=1e-14)
    np.testing.assert_allclose(actual.l2_squared,jnp.sum(weights*delta**2)/jnp.sum(weights*rho**2),atol=1e-14)
    assert actual.valid
    zero=jax.jit(lambda d:density_errors(target,d,(ao,weights)))(target.density)
    assert zero.valid and zero.l1==0 and zero.l2_squared==0


def test_wu_yang_diagnostics_include_l1_without_another_eigensolve(monkeypatch):
    target=iks.Target(jnp.diag(jnp.array([2.,0.])),jnp.eye(2),2)
    common=(target,jnp.eye(2),jnp.diag(jnp.array([-1.,1.])))
    plain=iks.WuYang(*common,baseline=jnp.zeros((2,2)))
    reporter=iks.WuYang(*common,baseline=jnp.zeros((2,2)),grid=(jnp.eye(2),jnp.ones(2)))
    solved=plain.score(jnp.array([[0.,.1],[.1,0.]]))
    def fail(*a,**k):raise AssertionError('density diagnostics diagonalized again')
    monkeypatch.setattr('gradscf.dft.iks.wu_yang._aufbau',fail)
    result=reporter.diagnose(solved)
    assert result.valid and result.density_l1>0 and result.density_error>0


def test_composite_normalization_keeps_pbe_zero_and_rewards_each_improvement():
    from gradscf.dft.iks.objectives import wu_yang_density_loss
    scales=(.002,.003,.00002)
    baseline=wu_yang_density_loss(*scales,scales=scales)
    np.testing.assert_allclose(baseline,1.,atol=1e-14)
    for index,expected in [(0,.5),(1,.25),(2,.25)]:
        terms=list(scales);terms[index]=0.
        score=1-wu_yang_density_loss(*terms,scales=scales)
        np.testing.assert_allclose(score,expected,atol=1e-14)
