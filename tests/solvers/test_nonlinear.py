import jax
import jax.numpy as jnp
import numpy as np


def test_diis_supports_configurable_history():
    from gradscf.solvers.nonlinear.diis import diis_solve

    x = jnp.array([[1.0], [4.0], [0.0]])
    errors = jnp.array([[1.0], [-2.0], [0.0]])
    np.testing.assert_allclose(diis_solve(x, errors, jnp.array(2)), [2.0], atol=1e-12)


def test_residual_iteration_and_implicit_response():
    from gradscf.solvers.nonlinear.iterate import NonlinearConfig, solve_nonlinear

    def fn(p):
        r = lambda x: x * x - p
        result = solve_nonlinear(
            r,
            jnp.array([1.0]),
            update=lambda x, rx: x - rx / (2 * x),
            observable=lambda x: jnp.sum(x),
            config=NonlinearConfig(diis_space=3, residual_tol=1e-11, energy_tol=1e-11),
        )
        return result.solution[0]

    np.testing.assert_allclose(jax.jit(fn)(2.0), np.sqrt(2.0), atol=1e-11)
    np.testing.assert_allclose(
        jax.jit(jax.grad(fn))(2.0), 1 / (2 * np.sqrt(2.0)), atol=1e-10
    )
