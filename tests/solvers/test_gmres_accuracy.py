"""True-residual regression for restarted JAX GMRES's early-inner-exit path."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.solvers import solve_linear, LinearSolverConfig


@pytest.mark.parametrize('scale',[1.,1e-10])
def test_gmres_early_exit_retains_requested_accuracy(scale):
    diagonal = jnp.linspace(1.,229.,343)
    rhs = scale*jnp.sin(jnp.arange(343))
    cfg = LinearSolverConfig(rtol=1e-10,atol=0.,restart=40,maxiter=20)
    result = solve_linear(jnp.diag(diagonal),rhs,config=cfg)
    assert result.converged, result.residual_norm
    np.testing.assert_allclose(result.solution,rhs/diagonal,rtol=1e-8,atol=scale*1e-11)
    # Transpose/adjoint uses the same numerical retry through custom_linear_solve.
    derivative = jax.grad(lambda shift:jnp.sum(solve_linear(
        jnp.diag(diagonal+shift),rhs,config=cfg).solution))(0.)
    np.testing.assert_allclose(derivative,-jnp.sum(rhs/diagonal**2),rtol=1e-8,atol=scale*1e-10)
