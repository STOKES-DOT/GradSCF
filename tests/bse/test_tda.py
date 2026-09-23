"""Static molecular BSE: independent dense kernels, limits and first-order AD."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def model():
    rng = np.random.default_rng(83)
    factors = rng.normal(size=(5, 5, 5)) * 0.08
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    screening = np.array([-1.1, -0.6, 0.3, 0.8, 1.2])
    qp = screening + np.array([0.03, 0.04, -0.02, -0.04, -0.05])
    dipole = rng.normal(size=(3, 5, 5)) * 0.2
    dipole = (dipole + dipole.transpose(0, 2, 1)) / 2
    return tuple(map(jnp.asarray, (qp, screening, factors, dipole)))


def dense_oracle(qp, screening, factors, occ, vir, screen_occ, screen_vir, singlet):
    """NumPy scalar-loop oracle; no GradSCF screening/kernel routines."""
    l = np.asarray(factors)
    eps = np.eye(len(l))
    for i in screen_occ:
        for a in screen_vir:
            eps += 4 * np.outer(l[:, i, a], l[:, i, a]) / (screening[a] - screening[i])
    pairs = [(i, a) for i in occ for a in vir]
    a = np.empty((len(pairs),) * 2)
    for p, (i, v) in enumerate(pairs):
        for q, (j, w) in enumerate(pairs):
            a[p, q] = (qp[v] - qp[i]) if p == q else 0.0
            a[p, q] += (2 if singlet else 0) * np.dot(l[:, i, v], l[:, j, w])
            a[p, q] -= np.dot(l[:, i, j], np.linalg.solve(eps, l[:, v, w]))
    return eps, a


@pytest.mark.parametrize("singlet", [True, False])
@pytest.mark.parametrize("method", ["dense", "davidson"])
def test_kernel_roots_and_strengths(model, singlet, method):
    from gradscf import bse
    from gradscf.gw.screened import build_static_screening, apply_static_screening

    qp, e, factors, dipole = model
    space = bse.make_bse_space(5, 2, occupied=(1,), virtual=(2, 4))
    screen = bse.make_bse_space(5, 2)
    state = build_static_screening(
        e, factors, occupied=screen.occupied, virtual=screen.virtual
    )
    dielectric, expected = dense_oracle(
        qp,
        e,
        factors,
        space.occupied,
        space.virtual,
        screen.occupied,
        screen.virtual,
        singlet,
    )
    np.testing.assert_allclose(state.dielectric, dielectric, atol=2e-13)
    np.testing.assert_allclose(
        apply_static_screening(state, jnp.eye(5)), np.linalg.inv(dielectric), atol=2e-13
    )
    op = bse.build_tda_operator(
        qp, factors, space, state, singlet=singlet, block_size=2
    )
    np.testing.assert_allclose(op.apply(jnp.eye(space.size)), expected, atol=2e-13)
    np.testing.assert_allclose(op.diagonal, np.diag(expected), atol=2e-13)
    cfg = bse.BSEConfig(
        nroots=2, solver=method, singlet=singlet, conv_tol=1e-11, block_size=2
    )
    out = jax.jit(
        lambda q, e, l: bse.run_bse(q, e, l, space, screening_space=screen, config=cfg)
    )(qp, e, factors)
    values, vectors = np.linalg.eigh(expected)
    assert np.all(out.converged) and np.all(out.response_valid)
    np.testing.assert_allclose(out.excitation_energies, values, atol=2e-11)
    mu = np.sqrt(2) * np.einsum(
        "sia,kia->ks",
        np.asarray(dipole)[:, space.occupied, :][:, :, space.virtual],
        vectors.T.reshape(2, 1, 2),
    )
    strength = (2 / 3) * values * np.sum(mu**2, axis=1) if singlet else np.zeros(2)
    np.testing.assert_allclose(
        bse.oscillator_strengths(out, dipole, space), strength, atol=2e-11
    )


def test_screening_energy_and_window_are_independent(model):
    from gradscf import bse

    qp, e, l, _ = model
    optical = bse.make_bse_space(5, 2, occupied=(1,), virtual=(2,))
    whole = bse.make_bse_space(5, 2)
    cfg = bse.BSEConfig(nroots=1, solver="dense")
    a = bse.run_bse(qp, e, l, optical, config=cfg)
    b = bse.run_bse(qp, e, l, optical, screening_space=optical, config=cfg)
    _, expected = dense_oracle(
        qp, e, l, optical.occupied, optical.virtual, whole.occupied, whole.virtual, True
    )
    np.testing.assert_allclose(a.excitation_energies, np.diag(expected), atol=1e-12)
    assert abs(float(a.excitation_energies[0] - b.excitation_energies[0])) > 1e-8


def test_energy_and_optical_response_include_screening_and_vectors(model):
    from gradscf import bse

    qp, e, l, d = model
    space = bse.make_bse_space(5, 2)
    cfg = bse.BSEConfig(
        nroots=2, solver="davidson", conv_tol=1e-11, adjoint_tol=1e-11, block_size=2
    )
    perturb = jnp.asarray(np.random.default_rng(84).normal(size=l.shape)) * 0.02
    perturb = (perturb + perturb.transpose(0, 2, 1)) / 2

    def value(x):
        out = bse.run_bse(
            qp + x * jnp.arange(5) * 0.01,
            e + x * jnp.arange(5) * 0.02,
            l + x * perturb,
            space,
            config=cfg,
        )
        f = bse.oscillator_strengths(out, d * (1 + 0.03 * x), space)
        return jnp.sum(out.excitation_energies) + 0.3 * jnp.sum(f)

    actual = jax.jit(jax.grad(value))(0.0)
    expected = (value(1e-4) - value(-1e-4)) / 2e-4
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
    np.testing.assert_allclose(jax.jvp(value, (0.0,), (1.0,))[1], actual, atol=2e-9)


def test_coverage_invalid_gaps_modes_and_degeneracy(model):
    from gradscf import bse

    qp, e, l, d = model
    space = bse.make_bse_space(5, 2)
    cfg = bse.BSEConfig(nroots=1, solver="dense")
    mask = jnp.ones(5, bool).at[2].set(False)
    failed = bse.run_bse(qp, e, l, space, qp_computed_mask=mask, config=cfg)
    assert not np.any(failed.converged)
    assert not np.isfinite(
        jax.grad(
            lambda x: bse.run_bse(
                qp * x, e, l, space, qp_computed_mask=mask, config=cfg
            ).excitation_energies[0]
        )(1.0)
    )
    assert bse.BSEConfig(tda=False).solver == "davidson"
    bad = bse.run_bse(qp, e.at[2].set(e[0]), l, space, config=cfg)
    assert not np.any(bad.converged)
    with pytest.raises(ValueError, match="aux"):
        bse.run_bse(qp, e, l, space, config=bse.BSEConfig(nroots=1, max_aux=2))
    with pytest.raises(ValueError, match="dense"):
        bse.run_bse(
            qp, e, l, space, config=bse.BSEConfig(nroots=1, solver="dense", max_dense=2)
        )
    one = bse.make_bse_space(3, 1)
    deg = bse.run_bse(
        jnp.array([-1.0, 1.0, 1.0]),
        jnp.array([-1.0, 1.0, 1.0]),
        jnp.zeros((1, 3, 3)),
        one,
        config=cfg,
    )
    assert deg.converged[0] and not deg.response_valid[0]
    assert not np.isfinite(
        jax.grad(
            lambda x: bse.run_bse(
                jnp.array([-1.0, x, 1.0]),
                jnp.array([-1.0, 1.0, 1.0]),
                jnp.zeros((1, 3, 3)),
                one,
                config=cfg,
            ).excitation_energies[0]
        )(1.0)
    )


def test_energy_only_cannot_silently_differentiate_strength(model):
    from gradscf import bse

    qp, e, l, d = model
    space = bse.make_bse_space(5, 2)
    cfg = bse.BSEConfig(nroots=1, solver="dense", gradient_mode="eigenvalue_only")

    def value(x):
        out = bse.run_bse(qp, e, l * x, space, config=cfg)
        return bse.oscillator_strengths(out, d, space)[0]

    assert np.isfinite(value(1.0))
    assert not np.isfinite(jax.grad(value)(1.0))


def test_large_block_request_does_not_pad_factors_to_requested_size(model):
    from gradscf import bse
    from gradscf.gw.screened import build_static_screening

    qp, e, factors, _ = model
    space = bse.make_bse_space(5, 2)
    screening = build_static_screening(
        e, factors, occupied=space.occupied, virtual=space.virtual
    )
    op = bse.build_tda_operator(qp, factors, space, screening, block_size=10000)
    traced = jax.make_jaxpr(op.matvec)(jnp.ones(space.size))
    assert all(np.size(c) <= 2 * factors.size for c in traced.consts)
