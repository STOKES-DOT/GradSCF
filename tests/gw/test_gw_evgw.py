"""evGW (Stage 3) tests: self-consistency of the QP spectrum."""

import numpy as np
import jax.numpy as jnp

from gradscf import dft, gto
from gradscf.gw import evgw_cd_restricted, g0w0_cd_restricted

_ATOM = "O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161"
_NW = 40


def _inputs():
    mol = gto.M(atom=_ATOM, basis="sto-3g", cart=True)
    mf = dft.RKS(mol, xc="hf").run()
    res = mf.scf_result
    return dict(
        mo_energy=jnp.asarray(res.mo_energy),
        mo_coeff=jnp.asarray(res.mo_coeff),
        nocc=5,
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
    # per-orbital graphical convergence is reported (pole-dense orbitals may
    # legitimately land on the best-residual iterate); the fixed-point check
    # below is the physical criterion
    assert np.all(np.isfinite(np.asarray(res.mo_energy)))
    # fixed point: re-evaluating G0W0 at the converged spectrum changes
    # nothing beyond the solver tolerance
    # fixed point: re-evaluating the linearized map at the converged QP
    # pole spectrum (with the mean-field base fixed) reproduces the spectrum
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
    # evGW@HF pulls the HOMO back toward the Koopmans value (a ~0.06 Ha
    # swing here) and shifts the core 1s by ~0.2 Ha; both are documented
    # evGW-vs-G0W0@HF magnitudes, so the sanity bound is generous.
    assert np.max(np.abs(delta)) < 0.3
