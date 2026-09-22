"""Metric-normalized real RPA against doubled matrices and finite differences."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def inputs():
    a = jnp.array([[1.2, 0.12, -0.03], [0.12, 1.8, 0.08], [-0.03, 0.08, 2.6]])
    b = jnp.array([[0.14, -0.07, 0.03], [-0.07, 0.2, 0.04], [0.03, 0.04, -0.1]])
    return a, b


def solve(a, b, **kwargs):
    from gradscf.solvers import solve_stable_rpa, EigenSolverConfig

    return solve_stable_rpa(
        a,
        b,
        config=EigenSolverConfig(
            method="dense",
            nroots=kwargs.pop("nroots", 2),
            atol=1e-11,
            gradient_mode=kwargs.pop("gradient_mode", "implicit_eigenvector"),
        ),
        **kwargs,
    )


def test_roots_metric_and_doubled_residual():
    a, b = inputs()
    out = jax.jit(solve)(a, b)
    matrix = np.block([[a, b], [-b, -a]])
    expected = np.sort(np.linalg.eigvals(matrix).real)[3:5]
    np.testing.assert_allclose(out.values, expected, atol=1e-12)
    np.testing.assert_allclose(out.x.T @ out.x - out.y.T @ out.y, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(out.x.T @ out.y - out.y.T @ out.x, 0, atol=1e-12)
    assert np.all(out.converged & out.response_valid) and out.stable
    assert np.max(out.residual_norms) < 1e-12


def test_energy_and_optical_response_matches_reconverged_fd():
    a, b = inputs()
    da, db = b * 0.4, a * 0.05
    dipole = jnp.array([0.1, 0.3, -0.2])

    def objective(t):
        out = solve(a + t * da, b + t * db)
        strengths = out.values * ((out.x + out.y).T @ (dipole + t * 0.03)) ** 2
        return jnp.sum(out.values * jnp.array([0.2, 0.7]) + strengths)

    grad = jax.jit(jax.grad(objective))(0.0)
    jvp = jax.jvp(objective, (0.0,), (1.0,))[1]
    fd = (objective(1e-5) - objective(-1e-5)) / 2e-5
    np.testing.assert_allclose(grad, fd, atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(jvp, grad, atol=1e-11)


def test_b_zero_limit_and_energy_only_stops_amplitudes():
    a, _ = inputs()
    out = solve(a, jnp.zeros_like(a))
    np.testing.assert_allclose(out.values, np.linalg.eigvalsh(a)[:2], atol=1e-12)
    np.testing.assert_allclose(out.y, 0, atol=1e-12)

    def amplitudes(t):
        out = solve(a * t, jnp.eye(3) * 0.1, gradient_mode="eigenvalue_only")
        return out.x, out.y

    tangent = jax.jvp(amplitudes, (1.0,), (1.0,))[1]
    for x in tangent:
        np.testing.assert_array_equal(x, 0)


@pytest.mark.parametrize(
    "a,b",
    [
        (jnp.diag(jnp.array([1.0, -2.0])), jnp.zeros((2, 2))),
        (jnp.eye(2), jnp.diag(jnp.array([2.0, 0.0]))),
        (jnp.array([[1.0, 0.1], [0.0, 2.0]]), jnp.zeros((2, 2))),
    ],
)
def test_invalid_structure_or_instability_not_repaired(a, b):
    out = jax.jit(lambda a: solve(a, b, nroots=1))(a)
    assert not out.stable
    assert not np.any(out.converged | out.response_valid)
    assert np.all(np.isnan(out.values))
    assert not np.all(
        np.isfinite(jax.grad(lambda a: solve(a, b, nroots=1).values.sum())(a))
    )


def test_degenerate_root_derivative_invalid_and_capacity_checked():
    a, b = jnp.eye(3), jnp.zeros((3, 3))
    out = solve(a, b, nroots=1)
    assert out.stable and np.all(out.converged)
    assert not np.any(out.response_valid)
    assert not np.isfinite(
        jax.grad(lambda t: solve(a * t, b, nroots=1).values.sum())(1.0)
    )
    from gradscf.solvers import solve_stable_rpa, EigenSolverConfig

    with pytest.raises(ValueError, match="max_dense"):
        solve_stable_rpa(a, b, config=EigenSolverConfig(method="dense", max_dense=2))


def test_mixed_degenerate_requested_set_is_explicitly_invalid():
    a = jnp.diag(jnp.array([1.0, 2.0, 2.0]))
    b = jnp.zeros_like(a)
    result = solve(a, b, nroots=2)
    # AD of a returned vector cannot isolate NaN factors belonging to unused
    # invalid roots. Fail the whole requested set instead of marking root 0 valid.
    assert not np.any(result.response_valid)
    da = jnp.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.1], [0.3, 0.1, 0.2]])
    invalid = lambda t: solve(a + t * da, b, nroots=2).values[0]
    assert not np.isfinite(jax.grad(invalid)(0.0))
    assert not np.isfinite(jax.jvp(invalid, (0.0,), (1.0,))[1])
    f = lambda t: solve(a + t * da, b, nroots=1).values[0]
    np.testing.assert_allclose(jax.grad(f)(0.0), 0.1, atol=1e-12)
    np.testing.assert_allclose(jax.jvp(f, (0.0,), (1.0,))[1], 0.1, atol=1e-12)


def test_reduced_residual_failure_invalidates_physical_response():
    from gradscf.solvers import solve_stable_rpa, EigenSolverConfig

    a, b = inputs()
    # The reduced problem has squared-energy units. Its absolute residual can
    # exceed tolerance while reconstructed physical residuals remain small.
    cfg = EigenSolverConfig(
        method="dense", nroots=2, atol=1e-10, gradient_mode="implicit_eigenvector"
    )
    result = solve_stable_rpa(a * 1000, b * 1000, config=cfg)
    assert np.max(result.residual_norms) < cfg.atol
    assert not np.any(result.response_valid)


def test_one_mode_analytic_frequency_and_transition_weight():
    a, b = 2.0, 0.4
    result = solve(jnp.array([[a]]), jnp.array([[b]]), nroots=1)
    omega = np.sqrt(a * a - b * b)
    np.testing.assert_allclose(result.values, [omega], atol=1e-14)
    np.testing.assert_allclose(
        (result.x + result.y) ** 2, [[(a - b) / omega]], atol=1e-14
    )
