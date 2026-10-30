"""Independent formula and AD checks for the external public PBE pilot model."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def example():
    path=Path(__file__).resolve().parents[2]/'examples/iks/pbe_agent_functionals.py'
    spec=spec_from_file_location('pbe_agent_functionals',path)
    module=module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_c001_agrees_with_recorded_formula_and_preserves_uniform_limit():
    model=example()
    rho=np.array([[0.,1e-12,.2,.4],[0.,2e-12,.3,.1]])
    grad=np.array([[[0.,0.,0.],[1e-14,0.,0.],[.1,.2,0.],[.3,0.,.2]],
                   [[0.,0.,0.],[2e-14,0.,0.],[.2,0.,.1],[.1,.2,0.]]])
    n=rho.sum(axis=0);reg=np.sqrt(n*n+1e-24)
    s2=(grad.sum(axis=0)**2).sum(axis=-1)/(4*(3*np.pi**2)**(2/3)*reg**(8/3))
    q=s2/(1+s2);z=(rho[0]-rho[1])/reg
    scale=-.7385587663820223*(.5*(n+reg))**(4/3)*n*n/(n*n+1e-24)
    theta=jnp.array([.13,-.09,.04])
    expected=scale*(theta[0]*q+theta[1]*q*q+theta[2]*q*z*z)
    np.testing.assert_allclose(model.pointwise_c001(theta,jnp.asarray(rho),jnp.asarray(grad)),expected,rtol=2e-13,atol=1e-25)
    np.testing.assert_array_equal(model.pointwise_c001(theta,jnp.asarray(rho),jnp.zeros_like(grad)),0.)
    np.testing.assert_array_equal(model.pointwise_c001(jnp.zeros(3),jnp.asarray(rho),jnp.asarray(grad)),0.)
    candidate=model.get_candidate('c001')
    assert all(lo < x < hi for x,(lo,hi) in zip(candidate['initial'],candidate['bounds']))


def test_c001_density_gradient_and_hvp_match_finite_difference():
    model=example();theta=jnp.array([.13,-.09,.04])
    rho=jnp.array([[.2,.3],[.1,.25]]);grad=jnp.array([[[.1,0.,.2],[.2,.1,0.]],[[.05,.1,0.],[.1,0.,.1]]])
    direction=jnp.array([[.03,-.02],[.01,.04]])
    energy=lambda r:jnp.sum(jnp.array([.7,1.2])*model.pointwise_c001(theta,r,grad))
    derivative=jax.grad(energy);h=1e-5
    np.testing.assert_allclose(jnp.sum(derivative(rho)*direction),
        (energy(rho+h*direction)-energy(rho-h*direction))/(2*h),atol=1e-11,rtol=1e-7)
    np.testing.assert_allclose(jax.jvp(derivative,(rho,),(direction,))[1],
        (derivative(rho+h*direction)-derivative(rho-h*direction))/(2*h),atol=1e-10,rtol=1e-7)


def test_c001_vacuum_is_finite_and_spin_term_is_unidentifiable_in_rks():
    model=example()
    r=jnp.zeros((2,2));g=jnp.zeros((2,2,3));theta=jnp.ones(3)*.1
    fn=lambda x:jnp.sum(model.pointwise_c001(theta,x,g))
    assert jnp.isfinite(fn(r)) and jnp.isfinite(jax.hessian(fn)(r)).all()
    r=jnp.array([[.1,.2],[.1,.2]]);g=jnp.ones((2,2,3))*.03
    np.testing.assert_array_equal(model.term_c001_spin(r,g),0.)
    np.testing.assert_array_equal(jax.grad(lambda x:jnp.sum(model.term_c001_spin(x,g)))(r),0.)


def test_c002_matches_quartic_window_formula_and_small_gradient_limit():
    model=example()
    rho=jnp.array([[.2,.3],[.1,.25]])
    grad=jnp.array([[[.1,0.,.2],[.2,.1,0.]],[[.05,.1,0.],[.1,0.,.1]]])
    n=np.asarray(rho).sum(axis=0);reg=np.sqrt(n*n+1e-24)
    s2=np.sum(np.asarray(grad).sum(axis=0)**2,axis=-1)/(4*(3*np.pi**2)**(2/3)*reg**(8/3))
    q=s2/(1+s2);scale=-.7385587663820223*(.5*(n+reg))**(4/3)*n*n/(n*n+1e-24)
    theta=jnp.array([.1,-.08])
    expected=scale*(theta[0]*q*q+theta[1]*q*q*(1-q)**2)
    np.testing.assert_allclose(model.pointwise_c002(theta,rho,grad),expected,rtol=2e-13,atol=1e-15)
    small=jnp.ones_like(grad)*1e-5
    np.testing.assert_allclose(model.pointwise_c002(jnp.array([1.,0.]),rho,.5*small)/
        model.pointwise_c002(jnp.array([1.,0.]),rho,small),1/16,rtol=1e-6)
    assert len(model.get_candidate('c002')['terms'])==2


def test_c002_density_gradient_and_hvp_match_finite_difference():
    model=example();theta=jnp.array([.1,-.08])
    rho=jnp.array([[.2,.3],[.1,.25]]);grad=jnp.ones((2,2,3))*.07
    direction=jnp.array([[.03,-.02],[.01,.04]])
    fn=lambda r:jnp.sum(model.pointwise_c002(theta,r,grad))
    derivative=jax.grad(fn);h=1e-5
    np.testing.assert_allclose(jnp.sum(derivative(rho)*direction),
        (fn(rho+h*direction)-fn(rho-h*direction))/(2*h),atol=1e-11,rtol=1e-7)
    np.testing.assert_allclose(jax.jvp(derivative,(rho,),(direction,))[1],
        (derivative(rho+h*direction)-derivative(rho-h*direction))/(2*h),atol=1e-10,rtol=1e-7)
