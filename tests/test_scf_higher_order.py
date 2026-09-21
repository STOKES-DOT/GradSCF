"""Analytic force-loss and mixed-derivative contracts for implicit SCF roots."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers.nonlinear import ImplicitFixedPointConfig, implicit_fixed_point_solution
from gradscf.scf.autodiff import attach_scf_backward


def _energy(r, theta, *, callbacks=False):
    options = {}
    if callbacks:
        options['apply_fixed_point_transpose'] = lambda x, p, v: p[0]*v
        options['params_vjp_from_adjoint'] = lambda x, p, v: (x*v, v)
    return implicit_fixed_point_solution((r, theta), solution=theta/(1-r),
        fixed_point=lambda x,p: p[0]*x+p[1],
        config=ImplicitFixedPointConfig(tolerance=1e-12,max_iter=10), **options)


@pytest.mark.parametrize('callbacks', [False, True])
def test_force_loss_parameter_gradient_is_not_silently_zero(callbacks):
    force = lambda r,t: -jax.grad(lambda r,t: _energy(r,t,callbacks=callbacks),argnums=0)(r,t)
    r,t = jnp.array(.2),jnp.array(1.)
    np.testing.assert_allclose(force(r,t),-1.5625,atol=1e-12,rtol=0)
    np.testing.assert_allclose(jax.jit(jax.grad(force,argnums=1))(r,t),-1.5625,atol=1e-11,rtol=0)
    loss = lambda theta: .5*force(r,theta)**2
    np.testing.assert_allclose(jax.grad(loss)(t),2.44140625,atol=1e-10,rtol=0)


def test_nonlinear_root_hessian_includes_solution_and_adjoint_response():
    def energy(p):
        x = attach_scf_backward(p,solution=jax.lax.stop_gradient(jnp.sqrt(p[0]/p[1])),
                                residual=lambda x,p: p[1]*x*x-p[0])
        return x**3
    oracle = lambda p: (p[0]/p[1])**1.5
    p = jnp.array([2.1,1.3])
    expected = jax.hessian(oracle)(p)
    np.testing.assert_allclose(jax.jit(jax.jacrev(jax.grad(energy)))(p),expected,atol=2e-10,rtol=0)
    np.testing.assert_allclose(jax.jit(jax.hessian(energy))(p),expected,atol=2e-10,rtol=0)


def test_implicit_root_jvp_and_jvp_of_gradient_match_analytic():
    p = jnp.array([.2,1.1]);v=jnp.array([.3,-.4])
    energy = lambda p: _energy(p[0],p[1])
    oracle = lambda p: p[1]/(1-p[0])
    np.testing.assert_allclose(jax.jvp(energy,(p,),(v,))[1],jax.jvp(oracle,(p,),(v,))[1],atol=1e-11,rtol=0)
    actual = jax.jit(lambda p,v: jax.jvp(jax.grad(energy),(p,),(v,))[1])(p,v)
    np.testing.assert_allclose(actual,jax.hessian(oracle)(p)@v,atol=2e-10,rtol=0)


def test_fixed_point_argument_mixed_derivative_and_jit_order():
    def energy(r,t):
        return implicit_fixed_point_solution(t,solution=t/(1-r),fixed_point=lambda x,p,a:a['r']*x+p,
                                             fixed_point_args={'r':r})
    fn = jax.jit(energy)
    mixed = jax.grad(jax.grad(fn,argnums=0),argnums=1)(jnp.array(.2),jnp.array(1.))
    np.testing.assert_allclose(mixed,1.5625,atol=1e-10,rtol=0)


def test_implicit_third_derivative_scalar_regression():
    f = lambda r: _energy(r,jnp.array(1.))
    np.testing.assert_allclose(jax.grad(jax.grad(jax.grad(f)))(jnp.array(.2)),
                               6/.8**4,atol=1e-9,rtol=0)


def test_scf_eigensolver_jvp_and_second_response_match_nondegenerate_reference():
    from gradscf.scf.core import _safe_symmetric_eigh
    def observable(p, eig):
        a=jnp.array([[-1.,p],[p,.5]])
        values,vectors=eig(a)
        return values[0]+.2*vectors[0,0]*vectors[1,0]
    f=lambda p:observable(p,_safe_symmetric_eigh)
    reference=lambda p:observable(p,jnp.linalg.eigh)
    p=jnp.array(.17)
    np.testing.assert_allclose(jax.jvp(f,(p,),(jnp.array(1.),))[1],jax.grad(reference)(p),atol=1e-12,rtol=0)
    np.testing.assert_allclose(jax.jit(jax.grad(jax.grad(f)))(p),jax.grad(jax.grad(reference))(p),atol=1e-11,rtol=0)


def test_tiny_adjoint_rhs_retains_relative_accuracy_and_derivatives():
    from gradscf.solvers.linear import solve_implicit_linear_system
    matrix=jnp.array([[1.65,.54],[.54,1.65]])
    def solve(rhs):
        return solve_implicit_linear_system(lambda v:matrix@v,rhs,tol=1e-11,max_iter=10,restart=2)
    rhs=jnp.array([1e-16,1e-16])
    np.testing.assert_allclose(solve(rhs),jnp.linalg.solve(matrix,rhs),atol=1e-28,rtol=0)
    np.testing.assert_allclose(jax.jacfwd(solve)(jnp.zeros(2)),jnp.linalg.inv(matrix),atol=1e-12,rtol=0)


def test_overlap_inverse_sqrt_response_at_repeated_eigenvalues():
    from gradscf.scf.core import _orthogonalizer
    direction=jnp.array([[.1,.2],[.2,-.1]])
    probe=jnp.array([[.3,.7],[.7,-.2]])
    fn=lambda t:jnp.sum(_orthogonalizer(jnp.eye(2)+t*direction,1e-10)*probe)
    first=-.5*jnp.sum(direction*probe)
    second=.75*jnp.sum((direction@direction)*probe)
    np.testing.assert_allclose(jax.grad(fn)(jnp.array(0.)),first,atol=1e-12,rtol=0)
    np.testing.assert_allclose(jax.jit(jax.grad(jax.grad(fn)))(jnp.array(0.)),second,atol=1e-11,rtol=0)


def test_overlap_inverse_sqrt_response_matches_spectral_reference():
    from gradscf.scf.core import _orthogonalizer
    direction=jnp.array([[.1,.2],[.2,-.1]])
    probe=jnp.array([[.3,.7],[.7,-.2]])
    def observable(t, orthogonalizer):
        return jnp.sum(orthogonalizer(jnp.diag(jnp.array([.7,1.3]))+t*direction)*probe)
    def spectral(s):
        w,v=jnp.linalg.eigh(s)
        return (v*w**-.5)@v.T
    fn=lambda t:observable(t,lambda s:_orthogonalizer(s,1e-10))
    reference=lambda t:observable(t,spectral)
    t=jnp.array(.2)
    np.testing.assert_allclose(jax.grad(fn)(t),jax.grad(reference)(t),atol=1e-12,rtol=0)
    np.testing.assert_allclose(jax.hessian(fn)(t),jax.hessian(reference)(t),atol=1e-11,rtol=0)
