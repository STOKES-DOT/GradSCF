"""One Hermitian entry and one subspace response for isolated and degenerate spectra."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def configs(target, nroots=2):
    from gradscf.solvers import (
        EigenSolverConfig,
        EigenResponseConfig,
        LinearSolverConfig,
    )

    return dict(
        config=EigenSolverConfig(method="dense", nroots=nroots, atol=1e-11),
        response=EigenResponseConfig(
            target=target, linear_config=LinearSolverConfig(rtol=1e-11)
        ),
    )


def test_degenerate_subspace_and_isolated_rank_one_share_entry():
    from gradscf.solvers import solve_hermitian

    a = jnp.diag(jnp.array([1.0, 1.0, 3.0, 5.0]))
    b = jnp.array(
        [
            [0.2, 0.6, 0.4, 0.1],
            [0.6, -0.3, 0.2, 0.3],
            [0.4, 0.2, 0.1, -0.2],
            [0.1, 0.3, -0.2, 0.4],
        ]
    )
    d = jnp.array([0.3, 0.4, 0.5, 0.2])

    def loss(t):
        out = solve_hermitian(a + t * b, probes=d, **configs("subspace"))
        assert out.values is None and out.vectors is None
        return d @ out.projection + 0.3 * out.eigenvalue_sum

    expected = -0.115 + 0.3 * (b[0, 0] + b[1, 1])
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.0), expected, atol=2e-10)
    np.testing.assert_allclose(jax.jvp(loss, (0.0,), (1.0,))[1], expected, atol=2e-10)
    separated = a.at[1, 1].set(1.5)
    eigen = (
        lambda t: (
            d
            @ solve_hermitian(separated + t * b, **configs("eigenpairs", 1)).vectors[
                :, 0
            ]
        )
        ** 2
    )
    project = (
        lambda t: d
        @ solve_hermitian(
            separated + t * b, probes=d, **configs("subspace", 1)
        ).projection
    )
    np.testing.assert_allclose(jax.grad(eigen)(0.0), jax.grad(project)(0.0), atol=2e-10)


def test_invalid_state_response_does_not_poison_valid_cluster():
    from gradscf.solvers import solve_hermitian

    a = jnp.diag(jnp.array([1.0, 1.0, 3.0]))

    def state(t):
        return solve_hermitian(a * t, **configs("eigenvalues")).values.sum()

    out = solve_hermitian(a, **configs("eigenpairs"))
    assert np.all(out.converged) and not np.any(out.response_valid)
    assert not np.isfinite(jax.grad(state)(1.0))
    cluster = lambda t: solve_hermitian(a * t, **configs("subspace")).eigenvalue_sum
    np.testing.assert_allclose(jax.grad(cluster)(1.0), 2.0, atol=1e-12)


def test_one_forward_solve_with_guard_and_trace_without_probes(monkeypatch):
    import gradscf.solvers.eigen.primal as primal
    from gradscf.solvers import solve_hermitian

    original = primal.dense_vectors
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs["nroots"])
        return original(*args, **kwargs)

    monkeypatch.setattr(primal, "dense_vectors", counted)
    result = solve_hermitian(
        jnp.diag(jnp.array([1.0, 1.0, 3.0, 5.0])), **configs("subspace")
    )
    assert calls == [3]
    assert result.projection is None
    assert result.raw_values.shape == (3,)
    np.testing.assert_allclose(result.eigenvalue_sum, 2.0)


def test_conflicting_response_controls_rejected():
    from gradscf.solvers import solve_hermitian, EigenSolverConfig, EigenResponseConfig

    with pytest.raises(ValueError, match="response"):
        solve_hermitian(
            jnp.eye(3),
            config=EigenSolverConfig(gradient_mode="implicit_eigenvector"),
            response=EigenResponseConfig(target="subspace"),
        )


def test_old_response_entry_points_are_removed():
    from pathlib import Path
    import gradscf.solvers as solvers
    from gradscf.solvers.eigen import davidson, response

    assert not hasattr(solvers, "solve_spectral_projector")
    assert not hasattr(davidson, "implicit_differential_davidson_lowest_symmetric")
    assert not hasattr(
        response, "implicit_differential_davidson_lowest_symmetric_with_eigenvectors"
    )
    assert not (Path(response.__file__).parent / "subspace.py").exists()


@pytest.mark.parametrize("diagonal", [[-0.2, 0.8, 1.5], [-0.2, 0.8]])
def test_tda_preserves_positive_root_selection(diagonal):
    from gradscf.tddft.tda import solve_tda_from_operator

    a = jnp.diag(jnp.asarray(diagonal))
    out = solve_tda_from_operator(
        jnp.asarray(diagonal)[None, :], lambda x: x @ a, jnp.diag(a), nstates=1
    )
    np.testing.assert_allclose(out.excitation_energies, [0.8], atol=1e-12)
    assert out.converged


def test_low_frequency_rpa_uses_frequency_gap_units():
    from gradscf.solvers import solve_stable_rpa, EigenSolverConfig

    a = jnp.diag(jnp.array([1e-5, 2e-5]))
    b = jnp.zeros_like(a)
    cfg = EigenSolverConfig(
        method="dense", nroots=1, atol=1e-12, gradient_mode="implicit_eigenvector"
    )
    f = lambda t: solve_stable_rpa(t * a, b, config=cfg)
    assert np.all(f(1.0).response_valid)
    np.testing.assert_allclose(
        jax.grad(lambda t: f(t).values[0])(1.0), 1e-5, atol=1e-13
    )


def test_interval_boundary_checks_excluded_eigenvalue():
    from gradscf.solvers import solve_hermitian, EigenSolverConfig

    # The lowest excluded root sits on the selection cutoff; the selected
    # positive root is far away, but the selection itself is discontinuous.
    cfg = EigenSolverConfig(method="dense", nroots=1, value_min=0.001)
    out = solve_hermitian(jnp.diag(jnp.array([0.001, 0.8, 1.5])), config=cfg)
    assert not np.any(out.response_valid)


@pytest.mark.parametrize("method", ["dense", "davidson"])
def test_rpa_rejects_hermitian_interval_controls(method):
    from gradscf.solvers import solve_rpa, EigenSolverConfig

    with pytest.raises(ValueError, match="value_min"):
        solve_rpa(
            jnp.eye(3),
            jnp.zeros((3, 3)),
            config=EigenSolverConfig(method=method, value_min=0.1),
        )
