"""Checked shared linear solver with multiple right-hand sides."""

import jax
import jax.numpy as jnp
import numpy as np


def test_direct_block_solve_response_and_failure():
    from gradscf.solvers import solve_linear, LinearSolverConfig

    cfg = LinearSolverConfig(method="direct", rtol=1e-12)
    rhs = jnp.arange(6, dtype=float).reshape(2, 3) * 0.1

    def value(x):
        a = jnp.array([[2.0 + x, 0.3], [0.1, 1.4]])
        return jnp.sum(solve_linear(a, rhs, config=cfg).solution ** 2)

    out = solve_linear(jnp.array([[2.0, 0.3], [0.1, 1.4]]), rhs, config=cfg)
    assert out.converged
    np.testing.assert_allclose(
        out.solution, np.linalg.solve([[2.0, 0.3], [0.1, 1.4]], rhs), atol=1e-13
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(value))(0.0), (value(1e-5) - value(-1e-5)) / 2e-5, atol=1e-10
    )
    bad = solve_linear(jnp.zeros((2, 2)), rhs, config=cfg)
    assert not bad.converged and np.isnan(bad.solution).all()
