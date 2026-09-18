"""G0W0 drivers (molecular, restricted and unrestricted) with CD self-energy.

Assembles the quasiparticle correction on top of a converged mean-field
(HF or DFT) calculation:

1. Low-rank factors are transformed to the MO basis,
   ``B_Q[m,n] = sum_pq B_Q[p,q] C[p,m] C[q,n]``.
2. The screened interaction ``W_mn(iw)`` is evaluated on the imaginary
   quadrature grid (RPA response summed over spin channels).
3. For each requested orbital the Dyson/QP equation

       w - e_p - [ Re Sigma_p(w) + v^x_pp - v^mf_pp ] = 0

   is solved graphically with the contour-deformation self-energy
   (:mod:`gradscf.gw.self_energy`), where ``v^x`` is the exact-exchange
   potential and ``v^mf`` the mean-field potential minus Coulomb
   (v_xc for DFT, -K for HF).

The orbitals not requested keep their mean-field energies in the output
(by design; PySCF zero-fills them, which is a foot-gun).

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- M. S. Hybertsen and S. G. Louie, Phys. Rev. B 34, 5390 (1986).
  DOI:10.1103/PhysRevB.34.5390
- X. Ren et al., New J. Phys. 14, 053020 (2012).
  DOI:10.1088/1367-2630/14/5/053020 (RI-G0W0)
- T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 14, 4856 (2018)
  (contour-deformation G0W0, PySCF ``gw_cd`` convention).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ..df import build_j_from_df, build_jk_from_df
from .freq import scaled_legendre_grid
from .polarizability import rho_response_iw
from .qp import solve_qp_batch
from .screened import screened_w_imag_axis
from .types import GWResult


def _mo_factors(df_factors: Array, mo_coeff: Array) -> Array:
    """Transform low-rank factors to the MO basis: B_Q[m,n] (naux,nmo,nmo)."""
    return jnp.einsum(
        "Qpq,pm,qn->Qmn",
        jnp.asarray(df_factors),
        jnp.asarray(mo_coeff),
        jnp.asarray(mo_coeff),
        precision=Precision.HIGHEST,
    )


def _exchange_mo(b_mn: Array, nocc: int) -> Array:
    """Exact-exchange potential in the MO basis: v^x[n,m] = -sum_i B[n,i] B[i,m]."""
    return -jnp.einsum(
        "Qni,Qim->nm", b_mn[:, :, :nocc], b_mn[:, :nocc, :], precision=Precision.HIGHEST
    )


def _qp_loop(
    *,
    mo_energy: Array,
    b_mn: Array,
    channels: tuple[tuple[Array, Array, float], ...],
    wmn: Array,
    freqs: Array,
    wts: Array,
    ef: Array,
    eta: float,
    delta_v: Array,
    nocc: int,
    orbs: Sequence[int],
    diff_mode: str,
    conjugate: bool = False,
    del00: Array | None = None,
    delP0: Array | None = None,
    q0: dict | None = None,
    e_mf: Array | None = None,
    linearized: bool = False,
) -> tuple[Array, Array, Array, bool]:
    """Solve the QP equation for every orbital in ``orbs`` (one spin channel).

    All orbitals are solved simultaneously in one vectorized secant loop
    (:func:`gradscf.gw.qp.solve_qp_batch`).  ``mo_energy`` is the pole
    spectrum inside G0/W; ``e_mf`` is the base of the QP equation
    (defaults to ``mo_energy``; the two differ under evGW iterations).
    """
    from .qp import solve_qp_batch
    from .self_energy import sigma_cd

    mo_energy = jnp.asarray(mo_energy)
    e_mf = mo_energy if e_mf is None else jnp.asarray(e_mf)
    orbs = [int(p) for p in orbs]
    shared = {
        "mo_energy": mo_energy,
        "channels": channels,
        "ef": ef,
        "eta": jnp.asarray(eta, dtype=jnp.float64),
        "freqs": freqs,
        "wts": wts,
        "conjugate": conjugate,
    }
    if q0 is not None:
        shared["q0"] = q0
    stacked = {
        "wmn_p": wmn[:, :, jnp.asarray(orbs)].transpose(2, 0, 1),  # (norb, nw, nmo)
        "b_pm": b_mn[:, jnp.asarray(orbs), :].transpose(1, 0, 2),  # (norb, naux, nmo)
        "b_mp": b_mn[:, :, jnp.asarray(orbs)].transpose(2, 0, 1),  # (norb, naux, nmo)
    }
    if del00 is not None:
        stacked["del_w"] = del00[None, :] + delP0[jnp.asarray(orbs)]  # (norb, nw)
        stacked["p_index"] = jnp.asarray(orbs)
    occupied = jnp.asarray([p < nocc for p in orbs])
    if linearized:
        # Z-factor linearized QP update (evGW iterations; smooth in the
        # pole spectrum, unlike the graphical solve in pole-dense regions):
        #   e_qp = e_mf + Z [Re Sigma(w0) + delta_v],  Z = 1/(1 - dSigma/dw)
        # evaluated at the current pole energies w0 = mo_energy[p].
        from .qp import sigma_cd_batch

        omega0 = mo_energy[jnp.asarray(orbs)]
        e_base = e_mf[jnp.asarray(orbs)]
        dv = delta_v[jnp.asarray(orbs)]
        sig0 = sigma_cd_batch(omega0, shared, stacked)
        dw = 1e-4
        sig_p = sigma_cd_batch(omega0 + dw, shared, stacked)
        sig_m = sigma_cd_batch(omega0 - dw, shared, stacked)
        dsig = jnp.real(sig_p - sig_m) / (2.0 * dw)
        zfac = 1.0 / (1.0 - dsig)
        roots = e_base + zfac * (jnp.real(sig0) + dv)
        qp_energy = e_mf.copy()
        qp_energy = qp_energy.at[jnp.asarray(orbs)].set(roots)
        sigma_qp = jnp.zeros_like(qp_energy, dtype=jnp.complex128)
        sigma_qp = sigma_qp.at[jnp.asarray(orbs)].set(sig0)
        converged_mask = jnp.ones_like(qp_energy, dtype=bool)
        return qp_energy, sigma_qp, converged_mask, True
    roots, done = solve_qp_batch(
        e_mf[jnp.asarray(orbs)],
        delta_v[jnp.asarray(orbs)],
        shared,
        stacked,
        occupied=occupied,
        diff_mode=diff_mode,
    )

    qp_energy = e_mf.copy()
    sigma_qp = jnp.zeros_like(qp_energy, dtype=jnp.complex128)
    converged_mask = jnp.ones_like(qp_energy, dtype=bool)
    if isinstance(roots, jax.core.Tracer):
        qp_energy = qp_energy.at[jnp.asarray(orbs)].set(roots)
        return qp_energy, sigma_qp, converged_mask, True
    converged = True
    for i, p in enumerate(orbs):
        converged = converged and bool(done[i])
        converged_mask = converged_mask.at[p].set(bool(done[i]))
        qp_energy = qp_energy.at[p].set(roots[i])
        ctx = {
            "mo_energy": mo_energy,
            "wmn_p": stacked["wmn_p"][i],
            "b_pm": stacked["b_pm"][i],
            "b_mp": stacked["b_mp"][i],
            "channels": channels,
            "ef": ef,
            "eta": jnp.asarray(eta, dtype=jnp.float64),
            "freqs": freqs,
            "wts": wts,
            "conjugate": conjugate,
        }
        if del00 is not None:
            ctx["del_w"] = stacked["del_w"][i]
            ctx["p_index"] = stacked["p_index"][i]
        if q0 is not None:
            ctx["q0"] = {**q0, "p_index": stacked["p_index"][i]}
        sigma_qp = sigma_qp.at[p].set(sigma_cd(roots[i], ctx))
    return qp_energy, sigma_qp, converged_mask, converged


def g0w0_cd_restricted(
    *,
    mo_energy: Array,
    mo_coeff: Array,
    nocc: int,
    df_factors: Array,
    fock_matrix: Array,
    hcore_matrix: Array,
    density_matrix: Array,
    nw: int = 100,
    eta: float = 1e-3,
    orbs: Sequence[int] | None = None,
    diff_mode: str = "implicit",
    mo_energy_poles: Array | None = None,
    linearized: bool = False,
) -> GWResult:
    """Spin-restricted G0W0 with contour deformation.

    Parameters
    ----------
    mo_energy, mo_coeff:
        Mean-field orbital energies ``(nmo,)`` and coefficients
        ``(nao, nmo)``.  ``mo_energy`` is the base of the QP equation.
    mo_energy_poles:
        Optional separate pole spectrum for G0 and W (evGW iterations
        pass the current QP spectrum here while ``mo_energy`` stays at the
        mean-field values).
    nocc:
        Number of doubly occupied orbitals.
    df_factors:
        Low-rank ERI factors ``(naux, nao, nao)`` (spectral factorization,
        see :mod:`gradscf.df`).
    fock_matrix, hcore_matrix, density_matrix:
        Converged mean-field quantities (AO basis).  The mean-field
        potential is reconstructed as ``v^mf = F - h - J[D]`` which equals
        ``v_xc`` for DFT and ``-K`` for HF starting points.
    nw, eta:
        Imaginary-grid size and broadening.
    orbs:
        Orbital indices for the self-energy correction; default all.
    diff_mode:
        ``"implicit"`` or ``"unrolled"``; see
        :func:`gradscf.gw.qp.solve_qp_orbital`.

    Returns
    -------
    :class:`gradscf.gw.types.GWResult`
    """
    mo_energy = jnp.asarray(mo_energy, dtype=jnp.float64)
    mo_coeff = jnp.asarray(mo_coeff, dtype=jnp.float64)
    nocc = int(nocc)
    nmo = mo_energy.shape[0]
    if orbs is None:
        orbs = range(nmo)

    b_mn = _mo_factors(df_factors, mo_coeff)
    b_ov = b_mn[:, :nocc, nocc:]

    j_mat = build_j_from_df(df_factors, jnp.asarray(density_matrix))
    v_mf = jnp.asarray(fock_matrix) - jnp.asarray(hcore_matrix) - j_mat
    v_mf_mo = mo_coeff.T @ v_mf @ mo_coeff
    vk_mo = _exchange_mo(b_mn, nocc)
    delta_v = jnp.diag(vk_mo - v_mf_mo)

    poles = (
        mo_energy
        if mo_energy_poles is None
        else jnp.asarray(mo_energy_poles, dtype=jnp.float64)
    )
    ef = 0.5 * (poles[nocc - 1] + poles[nocc])
    freqs, wts = scaled_legendre_grid(nw)

    def response_fn(omega):
        return rho_response_iw(omega, poles, b_ov, spin_factor=4.0)

    wmn = screened_w_imag_axis(b_mn, response_fn, freqs)

    qp_energy, sigma_qp, converged_mask, converged = _qp_loop(
        mo_energy=poles,
        b_mn=b_mn,
        channels=((poles, b_ov, 2.0),),
        wmn=wmn,
        freqs=freqs,
        wts=wts,
        ef=ef,
        eta=float(eta),
        delta_v=delta_v,
        nocc=nocc,
        orbs=orbs,
        diff_mode=diff_mode,
        e_mf=mo_energy,
        linearized=linearized,
    )
    return GWResult(
        mo_energy=qp_energy,
        mo_coeff=mo_coeff,
        converged=converged,
        sigma_qp=sigma_qp,
        converged_mask=converged_mask,
        nw=int(nw),
    )


def g0w0_cd_unrestricted(
    *,
    mo_energy: tuple[Array, Array],
    mo_coeff: tuple[Array, Array],
    nocc: tuple[int, int],
    df_factors: Array,
    fock_matrix: tuple[Array, Array] | None = None,
    hcore_matrix: Array | None = None,
    density_matrix: tuple[Array, Array] | None = None,
    nw: int = 100,
    eta: float = 1e-3,
    orbs: Sequence[int] | None = None,
    diff_mode: str = "implicit",
    mo_energy_poles: tuple[Array, Array] | None = None,
    linearized: bool = False,
) -> GWResult:
    """Spin-unrestricted G0W0 with contour deformation.

    The RPA response is summed over both spin channels (factor 2 each);
    the screened interaction ``W`` is shared, while the Green's function,
    exchange potential and Fermi estimate are spin-resolved.

    ``v^mf_sigma = F_sigma - h - J[D_total]`` requires the converged Fock
    matrices.  For an HF starting point the Fock matrices may be omitted:
    ``v^mf_sigma = -K_sigma`` is then rebuilt from ``density_matrix`` via
    the low-rank factors (exactly equivalent at the HF level).  For DFT
    starting points ``fock_matrix``, ``hcore_matrix`` and ``density_matrix``
    are all required.

    Parameters mirror :func:`g0w0_cd_restricted` with spin tuples
    ``(alpha, beta)``.  Returns :class:`GWResult` with stacked arrays of
    shape ``(2, nmo)`` / ``(2, nao, nmo)``.
    """
    e_a, e_b = (jnp.asarray(e, dtype=jnp.float64) for e in mo_energy)
    p_a, p_b = (
        (e_a, e_b)
        if mo_energy_poles is None
        else (jnp.asarray(p, dtype=jnp.float64) for p in mo_energy_poles)
    )
    c_a, c_b = (jnp.asarray(c, dtype=jnp.float64) for c in mo_coeff)
    nocc_a, nocc_b = int(nocc[0]), int(nocc[1])
    nmo = e_a.shape[0]
    if orbs is None:
        orbs = range(nmo)

    b_a = _mo_factors(df_factors, c_a)
    b_b = _mo_factors(df_factors, c_b)
    b_ov_a = b_a[:, :nocc_a, nocc_a:]
    b_ov_b = b_b[:, :nocc_b, nocc_b:]

    if fock_matrix is None:
        if density_matrix is None:
            raise ValueError(
                "Unrestricted G0W0 requires either converged Fock matrices "
                "(HF or DFT starts) or the spin densities (HF start only)."
            )
        # HF starting point: v^mf_sigma = -K_sigma rebuilt from the DF
        # factors; delta_v = v^x - v^mf = 0 identically at this level.
        _, k_a = build_jk_from_df(df_factors, jnp.asarray(density_matrix[0]))
        _, k_b = build_jk_from_df(df_factors, jnp.asarray(density_matrix[1]))
        vmf_mo_a = -(c_a.T @ k_a @ c_a)
        vmf_mo_b = -(c_b.T @ k_b @ c_b)
    else:
        if hcore_matrix is None or density_matrix is None:
            raise ValueError(
                "fock_matrix, hcore_matrix and density_matrix must be "
                "provided together."
            )
        dm_tot = jnp.asarray(density_matrix[0]) + jnp.asarray(density_matrix[1])
        j_tot = build_j_from_df(df_factors, dm_tot)
        h = jnp.asarray(hcore_matrix)
        vmf_mo_a = c_a.T @ (jnp.asarray(fock_matrix[0]) - h - j_tot) @ c_a
        vmf_mo_b = c_b.T @ (jnp.asarray(fock_matrix[1]) - h - j_tot) @ c_b

    vk_mo_a = _exchange_mo(b_a, nocc_a)
    vk_mo_b = _exchange_mo(b_b, nocc_b)
    delta_v_a = jnp.diag(vk_mo_a - vmf_mo_a)
    delta_v_b = jnp.diag(vk_mo_b - vmf_mo_b)

    ef_a = 0.5 * (p_a[nocc_a - 1] + p_a[nocc_a])
    ef_b = 0.5 * (p_b[nocc_b - 1] + p_b[nocc_b])

    freqs, wts = scaled_legendre_grid(nw)

    def response_fn(omega):
        return rho_response_iw(omega, p_a, b_ov_a, spin_factor=2.0) + rho_response_iw(
            omega, p_b, b_ov_b, spin_factor=2.0
        )

    # W is spin-independent; build it with the alpha-channel factors (the
    # spectral factors B are orbital-basis independent up to numerical noise).
    wmn = screened_w_imag_axis(b_a, response_fn, freqs)
    channels = ((p_a, b_ov_a, 1.0), (p_b, b_ov_b, 1.0))

    qp_a, sig_a, mask_a, conv_a = _qp_loop(
        mo_energy=p_a,
        b_mn=b_a,
        channels=channels,
        wmn=wmn,
        freqs=freqs,
        wts=wts,
        ef=ef_a,
        eta=float(eta),
        delta_v=delta_v_a,
        nocc=nocc_a,
        orbs=orbs,
        diff_mode=diff_mode,
        e_mf=e_a,
        linearized=linearized,
    )
    wmn_b = screened_w_imag_axis(b_b, response_fn, freqs)
    qp_b, sig_b, mask_b, conv_b = _qp_loop(
        mo_energy=p_b,
        b_mn=b_b,
        channels=channels,
        wmn=wmn_b,
        freqs=freqs,
        wts=wts,
        ef=ef_b,
        eta=float(eta),
        delta_v=delta_v_b,
        nocc=nocc_b,
        orbs=orbs,
        diff_mode=diff_mode,
        e_mf=e_b,
        linearized=linearized,
    )
    return GWResult(
        mo_energy=jnp.stack([qp_a, qp_b]),
        mo_coeff=jnp.stack([c_a, c_b]),
        converged=conv_a and conv_b,
        sigma_qp=jnp.stack([sig_a, sig_b]),
        converged_mask=jnp.stack([mask_a, mask_b]),
        nw=int(nw),
    )


__all__ = ["g0w0_cd_restricted", "g0w0_cd_unrestricted"]
