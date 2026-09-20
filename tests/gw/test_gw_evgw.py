"""evGW (Stage 3) tests: self-consistency of the QP spectrum."""

import numpy as np
import jax.numpy as jnp
import pytest

from gradscf import dft, gto
from gradscf.gw import evgw_cd_restricted, g0w0_cd_restricted

_ATOM = "O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161"
_NW = 40


def _inputs(atom=_ATOM, nocc=5):
    mol = gto.M(atom=atom, basis="sto-3g", cart=True)
    mf = dft.RKS(mol, xc="hf").run()
    res = mf.scf_result
    return dict(
        mo_energy=jnp.asarray(res.mo_energy),
        mo_coeff=jnp.asarray(res.mo_coeff),
        nocc=nocc,
        df_factors=jnp.asarray(mf._scf_inputs.eri_pair_matrix),
        fock_matrix=jnp.asarray(res.fock_matrix),
        hcore_matrix=jnp.asarray(res.hcore_matrix),
        density_matrix=jnp.asarray(res.density_matrix),
    )


def _df(kwargs):
    from gradscf.df import eri_pair_matrix_to_df_factors

    return eri_pair_matrix_to_df_factors(kwargs["df_factors"], nao=kwargs["mo_coeff"].shape[0], tol=1e-12)


def test_evgw_converges_to_fixed_point():
    kwargs = _inputs()
    kwargs["df_factors"] = _df(kwargs)
    res = evgw_cd_restricted(**kwargs, nw=_NW, max_iter=30, tol=1e-7)
    assert res.converged
    assert np.max(np.abs(res.qp_residual)) < 1e-7
    assert np.all(np.isfinite(np.asarray(res.mo_energy)))
    # The corrected one-step linearization also leaves a self-consistent
    # spectrum unchanged, with the mean-field base fixed.
    check = g0w0_cd_restricted(
        mo_energy=kwargs["mo_energy"],
        mo_energy_poles=res.mo_energy,
        mo_coeff=kwargs["mo_coeff"],
        nocc=kwargs["nocc"],
        df_factors=kwargs["df_factors"],
        fock_matrix=kwargs["fock_matrix"],
        hcore_matrix=kwargs["hcore_matrix"],
        density_matrix=kwargs["density_matrix"],
        nw=_NW,
        linearized=True,
    )
    np.testing.assert_allclose(check.mo_energy, res.mo_energy, atol=1e-5)


def test_evgw_shift_vs_g0w0_is_bounded():
    kwargs = _inputs()
    kwargs["df_factors"] = _df(kwargs)
    g0 = g0w0_cd_restricted(**kwargs, nw=_NW)
    ev = evgw_cd_restricted(**kwargs, nw=_NW, max_iter=30, tol=1e-7)
    delta = np.asarray(ev.mo_energy) - np.asarray(g0.mo_energy)
    assert np.all(np.isfinite(delta))
    # This is a sanity bound, not an external evGW accuracy reference.
    assert np.max(np.abs(delta)) < 0.3


@pytest.mark.parametrize("damping", [0.0, 0.5])
def test_evgw_h2_fixed_point_satisfies_dyson_equation(damping):
    from gradscf.df import build_j_from_df
    from gradscf.gw.freq import scaled_legendre_grid
    from gradscf.gw.g0w0 import _exchange_mo, _mo_factors
    from gradscf.gw.polarizability import rho_response_iw
    from gradscf.gw.screened import screened_w_imag_axis
    from gradscf.gw.self_energy import sigma_cd

    kwargs = _inputs("H 0 0 0; H 0 0 0.74", nocc=1)
    kwargs["df_factors"] = _df(kwargs)
    result = evgw_cd_restricted(**kwargs, nw=_NW, max_iter=50, tol=1e-9, damping=damping)
    poles = result.mo_energy
    b = _mo_factors(kwargs["df_factors"], kwargs["mo_coeff"])
    b_ov = b[:, :1, 1:]
    freqs, wts = scaled_legendre_grid(_NW)
    w = screened_w_imag_axis(b, lambda f: rho_response_iw(f, poles, b_ov), freqs)
    j_mat = build_j_from_df(kwargs["df_factors"], kwargs["density_matrix"])
    c = kwargs["mo_coeff"]
    v_mf = c.T @ (kwargs["fock_matrix"] - kwargs["hcore_matrix"] - j_mat) @ c
    delta_v = jnp.diag(_exchange_mo(b, 1) - v_mf)
    residuals = []
    self_energies = []
    for p in range(2):
        ctx = dict(
            mo_energy=poles, wmn_p=w[:, :, p], b_pm=b[:, p, :], b_mp=b[:, :, p],
            channels=((poles, b_ov, 2.0),), ef=poles.mean(), eta=1e-3,
            freqs=freqs, wts=wts,
        )
        sig = sigma_cd(poles[p], ctx)
        self_energies.append(sig)
        residuals.append(poles[p] - kwargs["mo_energy"][p] - sig.real - delta_v[p])
    np.testing.assert_allclose(residuals, 0.0, rtol=0, atol=1e-8)
    np.testing.assert_allclose(result.qp_residual, residuals, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.sigma_qp, self_energies, rtol=0, atol=1e-12)


def test_evgw_damping_cannot_hide_a_large_residual():
    kwargs = _inputs("H 0 0 0; H 0 0 0.74", nocc=1)
    kwargs["df_factors"] = _df(kwargs)
    with pytest.raises(ArithmeticError, match="Dyson residual"):
        evgw_cd_restricted(**kwargs, nw=_NW, max_iter=2, tol=1e-6, damping=0.999999)
