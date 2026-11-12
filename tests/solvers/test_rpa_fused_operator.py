"""Exact coupled RPA action reuses one physical product, float64/Hartree."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import EigenSolverConfig, LinearOperator, solve_rpa
from gradscf.solvers.eigen import structured_rpa


def problem(t):
    a=jnp.diag(jnp.array([1.0,1.6,2.3,3.1,4.0]))
    a=a.at[0,1].set(.04+t*.02).at[1,0].set(.04+t*.02)
    b=jnp.eye(5)*(.07+t*.01)
    b=b.at[0,2].set(.02+t*.01).at[2,0].set(.02+t*.01)
    return a,b


def config(method='davidson'):
    return EigenSolverConfig(method=method,nroots=2,max_subspace=5,atol=1e-10,
        maxiter=100,gradient_mode='implicit_eigenvector',adjoint_tol=1e-11,
        adjoint_maxiter=100)


def fused(t,method='davidson'):
    a,b=problem(t)
    def forbidden(x):
        raise AssertionError('The coupled path must not recompute separate A/B actions')
    ao=LinearOperator(a.shape,a.dtype,forbidden,diagonal=jnp.diag(a))
    bo=LinearOperator(b.shape,b.dtype,forbidden,diagonal=jnp.diag(b))
    def h(v):
        return jnp.concatenate((a@v[:5]+b@v[5:],b@v[:5]+a@v[5:]),axis=0)
    op=LinearOperator((10,10),a.dtype,h,matmat=h)
    return solve_rpa(ao,bo,config=config(method),h_operator=op)


@pytest.mark.parametrize('method',['davidson','dense'])
def test_fused_action_matches_separate_operator_and_metric(method):
    state=fused(jnp.asarray(.13),method)
    a,b=problem(.13)
    expected=solve_rpa(a,b,config=config('dense'))
    assert np.all(state.converged) and np.all(state.response_valid)
    np.testing.assert_allclose(state.values,expected.values,atol=2e-10)
    np.testing.assert_allclose(jnp.sum(state.x**2-state.y**2,axis=0),1.,atol=2e-10)
    np.testing.assert_allclose(state.residual_norms,0.,atol=1e-9)


def test_fused_closed_parameters_preserve_nested_jit_jvp_and_vjp():
    def observable(t):
        state=fused(t)
        return jnp.sum((state.x+state.y)**4)+.3*jnp.sum(state.values)
    point=jnp.asarray(.13)
    value,reverse=jax.jit(jax.value_and_grad(observable))(point)
    forward=jax.jit(lambda t:jax.jvp(observable,(t,),(jnp.ones_like(t),))[1])(point)
    finite=(observable(point+1e-4)-observable(point-1e-4))/(2e-4)
    assert np.isfinite(value) and abs(float(reverse))>1e-5
    np.testing.assert_allclose(forward,reverse,atol=2e-10,rtol=2e-8)
    np.testing.assert_allclose(reverse,finite,atol=2e-7,rtol=2e-5)


def test_fused_operator_shape_and_dtype_are_validated():
    a,b=problem(.1)
    bad=LinearOperator((9,9),a.dtype,lambda x:x)
    with pytest.raises(ValueError,match='(?i)(shape|dimension)'):
        solve_rpa(a,b,config=config(),h_operator=bad)


def test_initial_seed_and_matrix_epsilon_reach_existing_numerical_solver(monkeypatch):
    recorded=[]
    original=structured_rpa._davidson_lowest_tdhf
    def observed(*args,**kwargs):
        recorded.append((kwargs['initial_vectors'].shape[1],kwargs.get('matrix_eps')))
        return original(*args,**kwargs)
    monkeypatch.setattr(structured_rpa,'_davidson_lowest_tdhf',observed)
    a,b=problem(.1)
    state=solve_rpa(a,b,config=replace(config(),initial_guess_count=4),matrix_eps=3e-9)
    assert np.all(state.converged)
    assert recorded and recorded[-1][0]>=4 and recorded[-1][1]==3e-9
