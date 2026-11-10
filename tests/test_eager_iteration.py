"""Shared offline iteration works without private JAX trace-state APIs."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.solvers.nonlinear import iterate_eager,NonlinearConfig


def test_eager_array_map_converges_with_current_jax():
    result=iterate_eager(lambda x:.25*x+1.,jnp.zeros(2),
        config=NonlinearConfig(residual_tol=1e-10,maxiter=100))
    assert result.converged
    np.testing.assert_allclose(result.solution,4/3,atol=2e-10)


def test_offline_iteration_rejects_jit_with_a_closed_constant_initial_array():
    initial=jnp.zeros(2)
    with pytest.raises(NotImplementedError,match='eager|AD|JIT'):
        jax.jit(lambda _:iterate_eager(lambda x:.25*x+1.,initial).solution)(0.)


def test_trace_state_and_context_may_be_exported_from_different_namespaces(monkeypatch):
    from jax.extend import core as extended
    take=getattr(extended,'take_current_trace',None) or jax.core.take_current_trace
    monkeypatch.setattr(extended,'take_current_trace',take,raising=False)
    monkeypatch.delattr(jax.core,'take_current_trace',raising=False)
    result=iterate_eager(lambda x:.25*x+1.,jnp.zeros(2),
        config=NonlinearConfig(residual_tol=1e-10,maxiter=100))
    assert result.converged
