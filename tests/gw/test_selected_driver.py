"""The public array driver applies windows before MO-pair preparation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("iterative", [False, True])
def test_direct_g0w0_avoids_full_mo_factors_and_matches_dense(monkeypatch, iterative):
    from gradscf.gw import g0w0
    from gradscf.gw.freq import scaled_legendre_grid
    from gradscf.gw.polarizability import rho_response_iw
    from gradscf.gw.screened import screened_w_imag_axis
    rng = np.random.default_rng(29)
    factors = jnp.asarray(rng.normal(size=(4, 5, 5))*.05)  # raw factors remain nonsymmetric
    energy = jnp.array([-1., -.4, .2, .6, 1.1])
    coeff = jnp.eye(5)
    green, target, occupied, virtual = (3, 0, 2), (4, 1), (1, 0), (4, 2)
    ov = factors[:, jnp.array(occupied)[:, None], jnp.array(virtual)[None, :]]
    response = lambda w: rho_response_iw(w, (energy[jnp.array(occupied)], energy[jnp.array(virtual)]), ov)
    frequencies, _ = scaled_legendre_grid(8)
    dense = screened_w_imag_axis(factors, response, frequencies)
    static = screened_w_imag_axis(factors, response, jnp.zeros(1))[0]
    args = dict(mo_energy=energy, mo_coeff=coeff, nocc=2, df_factors=factors,
        fock_matrix=jnp.diag(energy), hcore_matrix=jnp.diag(energy),
        density_matrix=jnp.diag(jnp.array([2., 2., 0., 0., 0.])), nw=8,
        evaluate_only=True, orbs=target, g_orbitals=green,
        screening_occupied=occupied, screening_virtual=virtual)
    expected = g0w0.g0w0_cd_restricted(**args, wmn_fixed=dense, wmn_static_fixed=static)
    def forbidden(*args, **kwargs):
        raise AssertionError("Selected direct G0W0 must not construct full Qmn")
    monkeypatch.setattr(g0w0, "_mo_factors", forbidden)
    from gradscf.solvers import LinearSolverConfig
    config = LinearSolverConfig(method="gmres", rtol=1e-11, atol=1e-13, restart=8) if iterative else None
    actual = jax.jit(lambda b: g0w0.g0w0_cd_restricted(**{**args, "df_factors": b},
                   screening_config=config))(factors)
    for name in ("mo_energy", "sigma_qp", "qp_weight", "qp_residual"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   atol=1e-10, rtol=1e-9, equal_nan=True)
    np.testing.assert_array_equal(actual.qp_computed_mask, expected.qp_computed_mask)


def test_unrestricted_selected_driver_does_not_construct_full_mo_factors(monkeypatch):
    from gradscf.gw import g0w0
    rng = np.random.default_rng(5)
    b = jnp.asarray(rng.normal(size=(4, 5, 5))*.05)
    b = .5*(b+b.swapaxes(1, 2))
    e, c = jnp.array([-1., -.4, .2, .6, 1.1]), jnp.eye(5)
    d = jnp.diag(jnp.array([1., 1., 0., 0., 0.]))
    common = dict(df_factors=b, hcore_matrix=jnp.zeros((5, 5)), nw=8,
                  orbs=(1, 2), evaluate_only=True)
    expected = g0w0.g0w0_cd_restricted(**common, mo_energy=e, mo_coeff=c,
        nocc=2, fock_matrix=jnp.diag(e), density_matrix=2*d)
    def forbidden(*args, **kwargs):
        raise AssertionError("UGW must consume selected factors")
    monkeypatch.setattr(g0w0, "_mo_factors", forbidden)
    actual = g0w0.g0w0_cd_unrestricted(**common, mo_energy=(e, e), mo_coeff=(c, c),
        nocc=(2, 2), fock_matrix=(jnp.diag(e), jnp.diag(e)), density_matrix=(d, d))
    for name in ("mo_energy", "sigma_qp", "qp_weight", "qp_residual"):
        for spin in range(2):
            np.testing.assert_allclose(getattr(actual, name)[spin], getattr(expected, name),
                                       atol=1e-10, rtol=1e-9, equal_nan=True)


@pytest.mark.parametrize("iterative", [False, True])
def test_empty_g_window_in_selected_driver_recovers_hf(iterative):
    from gradscf.gw.g0w0 import g0w0_cd_restricted
    from gradscf.integrals.molecular.jk import build_jk_from_df
    from gradscf.solvers import LinearSolverConfig
    b = jnp.ones((3, 3, 3))*.07
    e, c, d = jnp.array([-.8, .3, .9]), jnp.eye(3), jnp.diag(jnp.array([2., 0., 0.]))
    coulomb, exchange = build_jk_from_df(b, d)
    fock = jnp.diag(e)
    cfg = LinearSolverConfig(method="gmres", rtol=1e-11, atol=1e-13) if iterative else None
    result = g0w0_cd_restricted(mo_energy=e, mo_coeff=c, nocc=1, df_factors=b,
        fock_matrix=fock, hcore_matrix=fock-coulomb+.5*exchange, density_matrix=d,
        g_orbitals=(), nw=8, screening_config=cfg)
    np.testing.assert_allclose(result.mo_energy, e, atol=1e-10, rtol=0.)
    np.testing.assert_allclose(result.sigma_qp, 0., atol=1e-12, rtol=0.)
    assert bool(result.converged)
