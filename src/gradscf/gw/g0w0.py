"""G0W0 drivers (molecular, restricted and unrestricted) with CD self-energy.

Assembles the quasiparticle correction on top of a converged mean-field
(HF or DFT) calculation:

1. Low-rank factors are transformed to the MO basis,
   ``B_Q[m,n] = sum_pq B_Q[p,q] C[p,m] C[q,n]``.
2. The screened interaction ``W_mn(iw)`` is evaluated on the imaginary
   quadrature grid (RPA response summed over spin channels).
3. For each requested orbital the Dyson/QP equation

       w - e_p - [ Re Sigma_p(w) + v^x_pp - v^mf_pp ] = 0

   is solved as a nonlinear root with the contour-deformation self-energy
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
from math import isfinite
from numbers import Integral

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from gradscf.integrals.molecular.jk import build_j_from_df, build_jk_from_df
from .freq import scaled_legendre_grid
from .polarizability import rho_response_iw
from .qp import _df_dw_batch, qp_residual_batch, sigma_cd_batch, solve_qp_batch
from .screened import screened_w_imag_axis
from .poles import (
    rpa_resolvent,
    screened_w_imag_resolvent,
)
from .types import GWResult
from .._memory import DEFAULT_MAX_MEMORY_MB, array_storage_bytes


def _requested_orbitals(orbs, nmo):
    indices = tuple(range(nmo)) if orbs is None else tuple(orbs)
    if (
        not indices
        or any(
            not isinstance(p, Integral) or isinstance(p, bool) or not 0 <= p < nmo
            for p in indices
        )
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("orbs must contain distinct in-range integer orbital indices")
    return tuple(map(int, indices))


def _window_indices(indices, allowed, name):
    """Static original-MO indices; an empty correlation window is allowed."""
    allowed = tuple(allowed)
    selected = allowed if indices is None else tuple(indices)
    if any(not isinstance(i, Integral) or isinstance(i, bool) or i not in allowed
           for i in selected) or len(set(selected)) != len(selected):
        raise ValueError(f"{name} must contain distinct indices in {allowed}")
    return tuple(map(int, selected))


def _mo_factors(df_factors: Array, mo_coeff: Array) -> Array:
    """Transform low-rank factors to the MO basis: B_Q[m,n] (naux,nmo,nmo)."""
    return jnp.einsum(
        "Qpq,pm,qn->Qmn",
        jnp.asarray(df_factors),
        jnp.asarray(mo_coeff),
        jnp.asarray(mo_coeff),
        precision=Precision.HIGHEST,
    )


def build_screened_w_restricted(
    *, mo_energy, mo_coeff, nocc, df_factors, nw, eta,
    screening_occupied=None, screening_virtual=None,
):
    """Build dynamic and static W once for a fixed restricted spectrum."""
    energy = jnp.asarray(mo_energy, dtype=jnp.float64)
    coeff = jnp.asarray(mo_coeff, dtype=jnp.float64)
    factors = _mo_factors(df_factors, coeff)
    nmo = energy.shape[0]
    oi = _window_indices(screening_occupied, range(nocc), "screening_occupied")
    va = _window_indices(screening_virtual, range(nocc, nmo), "screening_virtual")
    oi_array = jnp.asarray(oi, dtype=jnp.int32)
    va_array = jnp.asarray(va, dtype=jnp.int32)
    b_ov = factors[:, oi_array[:, None], va_array[None, :]]
    pairs = (energy[oi_array], energy[va_array])
    freqs, _ = scaled_legendre_grid(nw)

    resolvent_data = rpa_resolvent(jnp.concatenate(pairs), b_ov, factors, eta=eta)
    return {
        "wmn": screened_w_imag_resolvent(resolvent_data, freqs),
        "wmn_static": screened_w_imag_resolvent(resolvent_data, jnp.zeros(1))[0],
        "resolvent_data": resolvent_data,
    }


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
    qp_solver: str = "secant",
    conjugate: bool = False,
    del00: Array | None = None,
    delP0: Array | None = None,
    q0: dict | None = None,
    e_mf: Array | None = None,
    linearized: bool = False,
    evaluate_only: bool = False,
    g_orbitals=None,
    return_weight=False,
    return_diagnostics=False,
    wmn_static=None,
    del_w_static=None,
    resolvent_data=None,
    qp_initial=None,
    qp_tol=1e-6,
    prepared=None,
    execution_plan=None,
    transition=None,
) -> tuple[Array, ...]:
    """Solve the QP equation for every orbital in ``orbs`` (one spin channel).

    Roots share nonlinear batch control; screening actions use the execution
    plan's bounded scheduling. ``mo_energy`` is the pole
    spectrum inside G0/W; ``e_mf`` is the base of the QP equation
    (defaults to ``mo_energy``; the two differ under evGW iterations).
    """
    if qp_solver not in {"secant", "newton", "hybrid"}:
        raise ValueError("qp_solver must be secant, newton or hybrid")
    if qp_solver != "secant" and (linearized or evaluate_only):
        raise ValueError("qp_solver requires a nonlinear QP solve")
    mo_energy = jnp.asarray(mo_energy)
    if linearized and evaluate_only:
        raise ValueError("linearized and evaluate_only are mutually exclusive.")
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
    if transition is not None:
        shared["transition"] = transition
    if execution_plan is not None:
        shared["execution_plan"] = execution_plan
    if q0 is not None:
        shared["q0"] = q0
    if del_w_static is not None:
        shared["del_w_static"] = jnp.asarray(del_w_static)
    if resolvent_data is not None:
        shared["resolvent_data"] = resolvent_data
    stacked = prepared if prepared is not None else {
        "wmn_p": wmn[:, :, jnp.asarray(orbs)].transpose(2, 0, 1),  # (norb, nw, nmo)
        "b_pm": b_mn[:, jnp.asarray(orbs), :].transpose(1, 0, 2),  # (norb, naux, nmo)
        "b_mp": b_mn[:, :, jnp.asarray(orbs)].transpose(2, 0, 1),  # (norb, naux, nmo)
        "p_index": jnp.asarray(orbs),
    }
    if wmn_static is not None and prepared is None:
        stacked["wmn_static"] = wmn_static[:, jnp.asarray(orbs)].T
        shared["occupation_sign"] = jnp.where(jnp.arange(mo_energy.size) < nocc, -1., 1.)
    if conjugate:
        # Periodic factors are ordered (intermediate m, external p).
        # The two vertices must be the same pair, contracted as b^dagger W b.
        stacked["b_pm"] = stacked["b_mp"]
    if del00 is not None:
        stacked["del_w"] = del00[None, :] + delP0[jnp.asarray(orbs)]  # (norb, nw)
    if g_orbitals is not None:
        gi = jnp.asarray(g_orbitals, dtype=jnp.int32)
        shared["mo_energy"] = mo_energy[gi]
        shared["g_indices"] = gi
        if wmn_static is not None:
            shared["occupation_sign"] = shared["occupation_sign"][gi]
        if prepared is None:
            for key in ("wmn_p", "b_pm", "b_mp") + (("wmn_static",) if wmn_static is not None else ()):
                stacked[key] = stacked[key][..., gi]
    if prepared is not None and "wmn_static" in prepared:
        signs = jnp.where(jnp.arange(mo_energy.size) < nocc, -1., 1.)
        shared["occupation_sign"] = signs if g_orbitals is None else signs[gi]
    occupied = jnp.asarray([p < nocc for p in orbs])
    omega0 = mo_energy[jnp.asarray(orbs)]
    e_base = e_mf[jnp.asarray(orbs)]
    dv = delta_v[jnp.asarray(orbs)]
    if evaluate_only:
        roots = omega0
    elif linearized:
        # Linearize the Dyson equation at fixed G/W and active residue set:
        #   e_qp = w0 + Z [e_mf + Re Sigma(w0) + delta_v - w0]
        #   Z = 1/(1 - dSigma/dw)
        # evaluated at the current pole energies w0 = mo_energy[p].
        sig0 = sigma_cd_batch(omega0, shared, stacked)
        # Differentiate at fixed G/W and active pole set. A central
        # difference can cross a residue-selection boundary at omega0.
        zfac = 1.0 / _df_dw_batch(omega0, e_base, dv, shared, stacked)
        roots = omega0 + zfac * (e_base + jnp.real(sig0) + dv - omega0)
    else:
        roots, done, diagnostics = solve_qp_batch(
            e_mf[jnp.asarray(orbs)],
            delta_v[jnp.asarray(orbs)],
            shared,
            stacked,
            occupied=occupied,
            diff_mode=diff_mode,
            method=qp_solver,
            tol=qp_tol,
            initial_guess=None if qp_initial is None else jnp.asarray(qp_initial)[jnp.asarray(orbs)],
            return_diagnostics=True,
        )

    # Compute the same observables and status eagerly and under transforms.
    # In particular, sigma_qp is evaluated at the returned QP energies,
    # including when a linearized update moves away from its expansion point.
    sig_roots = sigma_cd_batch(roots, shared, stacked)
    residual = roots - e_base - jnp.real(sig_roots) - dv
    if linearized or evaluate_only:
        done = jnp.isfinite(roots) & (jnp.abs(residual) < qp_tol)
        slope = _df_dw_batch(roots, e_base, dv, shared, stacked)
        done &= jnp.isfinite(slope) & (slope > 1e-10)
        if linearized:
            done = done & jnp.isfinite(zfac)
    done &= jnp.isfinite(sig_roots)
    qp_energy = e_mf.copy()
    sigma_qp = jnp.zeros_like(qp_energy, dtype=jnp.complex128)
    converged_mask = jnp.ones_like(qp_energy, dtype=bool)
    qp_energy = qp_energy.at[jnp.asarray(orbs)].set(roots)
    sigma_qp = sigma_qp.at[jnp.asarray(orbs)].set(sig_roots)
    converged_mask = converged_mask.at[jnp.asarray(orbs)].set(done)
    qp_residual = jnp.zeros_like(qp_energy).at[jnp.asarray(orbs)].set(residual)
    result = (qp_energy, sigma_qp, converged_mask, jnp.all(done), qp_residual)
    if not return_weight and not return_diagnostics:
        return result
    nonlinear = not linearized and not evaluate_only
    if nonlinear:
        local_weight = diagnostics.weight
    else:
        valid_slope = jnp.isfinite(slope) & (jnp.abs(slope) > 1e-10)
        local_weight = jnp.where(valid_slope, 1/jnp.where(slope != 0, slope, 1.), jnp.nan)
    weights = jnp.full_like(qp_energy, jnp.nan).at[jnp.asarray(orbs)].set(local_weight)
    if not return_diagnostics:
        return (*result, weights)
    if nonlinear:
        local_on_shell, linear = diagnostics.on_shell, diagnostics.linearized
    else:
        equation = lambda w: qp_residual_batch(w, e_base, dv, shared, stacked)
        f0, slope0 = jax.jvp(equation, (e_base,), (jnp.ones_like(e_base),))
        local_on_shell = e_base-f0
        linear = jnp.where(jnp.isfinite(slope0) & (jnp.abs(slope0) > 1e-10),
                          e_base-f0/jnp.where(slope0 != 0, slope0, 1.), jnp.nan)
    on_shell = jnp.full_like(qp_energy, jnp.nan).at[jnp.asarray(orbs)].set(local_on_shell)
    linear = jnp.full_like(qp_energy, jnp.nan).at[jnp.asarray(orbs)].set(linear)
    count = jnp.zeros_like(qp_energy, dtype=jnp.int32)
    branch = jnp.zeros_like(qp_energy, dtype=bool)
    if nonlinear:
        count = count.at[jnp.asarray(orbs)].set(diagnostics.candidate_count)
        branch = branch.at[jnp.asarray(orbs)].set(diagnostics.branch_valid)
    else:
        branch = branch.at[jnp.asarray(orbs)].set(jnp.isfinite(slope) & (slope > 1e-10))
    return (*result, weights, dict(qp_on_shell=on_shell, qp_linearized=linear,
                                  qp_candidate_count=count, qp_branch_valid=branch))


def g0w0_cd_restricted(
    *,
    mo_energy: Array,
    mo_coeff: Array,
    nocc: int,
    df_factors: Array | None = None,
    fock_matrix: Array,
    hcore_matrix: Array,
    density_matrix: Array,
    nw: int = 100,
    eta: float = 1e-3,
    orbs: Sequence[int] | None = None,
    diff_mode: str = "implicit",
    qp_solver: str = "secant",
    mo_energy_poles: Array | None = None,
    linearized: bool = False,
    evaluate_only: bool = False,
    screening_energy: Array | None = None,
    g_orbitals: Sequence[int] | None = None,
    screening_occupied: Sequence[int] | None = None,
    screening_virtual: Sequence[int] | None = None,
    wmn_fixed: Array | None = None,
    wmn_static_fixed: Array | None = None,
    resolvent_expansion: bool = False,
    resolvent_data_fixed: dict | None = None,
    qp_initial: Array | None = None,
    qp_tol: float = 1e-6,
    max_memory: float = DEFAULT_MAX_MEMORY_MB,
    screening_config=None,
    v_mf_matrix=None,
    reserved_bytes: int = 0,
    _preparation=None,
    eri=None,
    eri_pair_matrix=None,
    direct_basis=None,
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
        Coulomb factors ``(naux, nao, nao)`` from declared spectral
        factorization or explicit auxiliary-basis 2c/3c RI. The array driver
        does not silently change the integral approximation.
    eri, eri_pair_matrix, direct_basis:
        Alternative real four-center sources (full, packed, or native selected
        AO2MO). Supply exactly one source when df_factors is omitted. This
        conventional path does not build auxiliary or AO-pair spectral factors.
        It retains the full OV Coulomb cache within max_memory; native direct
        coordinates and JAX orbital/projection derivatives stay live. Native
        primitive-parameter derivative limits belong to the integral supplier.
    fock_matrix, hcore_matrix, density_matrix:
        Converged mean-field quantities (AO basis).  The mean-field
        potential is reconstructed as ``v^mf = F - h - J[D]`` which equals
        ``v_xc`` for DFT and ``-K`` for HF starting points.
    nw, eta:
        Imaginary-grid size and real-frequency W broadening. Static Wc(0)
        subtraction treats the imaginary Green denominator in its zero-eta
        limit, including the contour boundary at an intermediate G pole.
    orbs:
        Orbital indices for the self-energy correction; default all.
    diff_mode:
        ``"implicit"`` or ``"unrolled"``; see
        :func:`gradscf.gw.qp.solve_qp_orbital`.
    screening_energy:
        Independent spectrum in W. Defaults to the G pole spectrum; fixing it
        to mean-field energies implements W0 during evGW0 iterations.
    g_orbitals, screening_occupied, screening_virtual:
        Independent static windows in original MO numbering, defaulting to all
        valid orbitals. They restrict correlation sums, never mean-field J/K.
        Empty windows give zero correlation self-energy. The target ``orbs``
        window is independent. Selected direct screening prepares required
        vertices; full/cached RPA references keep full MO-pair data under
        their separate budget preflight.
    qp_solver:
        Nonlinear forward root method: secant, newton or hybrid. Alternative
        methods require a nonlinear solve, not linearized/evaluate-only mode.
    qp_initial:
        Optional real ``(nmo,)`` initial QP energies for a nonlinear solve.
        Selected entries seed the requested roots without altering G/W or the
        mean-field base. Implicit derivatives follow the selected root and do
        not differentiate this numerical seed. This is not a main-peak selector.
    qp_tol:
        Positive absolute root step/residual tolerance (Hartree). The default
        preserves the historical forward precision. Small displacement or
        spectral-residue derivatives can require a tighter converged root.
    linearized:
        Take one Newton update about the pole energies, with G/W fixed.
        The frequency derivative holds the active residue set fixed; this
        update is not an outer self-consistency solver.
    evaluate_only:
        Evaluate self-energy and Dyson residual at the supplied pole
        spectrum without solving or updating QP energies. Used by evGW;
        mutually exclusive with ``linearized``.
    max_memory:
        Known working arrays in decimal MB (shared default 60000), excluding
        arbitrary outer AD/model/compiler work. Selected outputs must fit.
    screening_config:
        Optional shared GMRES configuration. Omit for dense screening;
        explicit direct or cached-RPA combinations are rejected clearly.
    v_mf_matrix:
        Optional actual SCF F-h-J in AO form when GW uses another fit.
    reserved_bytes:
        Known caller-owned arrays outside this numerical argument payload.

    Returns
    -------
    :class:`gradscf.gw.types.GWResult`
    """
    if not isfinite(qp_tol) or qp_tol <= 0:
        raise ValueError("qp_tol must be positive and finite")
    if qp_initial is not None:
        qp_initial = jnp.asarray(qp_initial)
        if qp_initial.shape != jnp.shape(mo_energy) or jnp.iscomplexobj(qp_initial):
            raise ValueError("qp_initial must be a real (nmo,) spectrum")
        if linearized or evaluate_only:
            raise ValueError("qp_initial requires a nonlinear QP solve")
    if df_factors is None:
        if resolvent_expansion or any(x is not None for x in (wmn_fixed, wmn_static_fixed, resolvent_data_fixed)):
            raise NotImplementedError("Conventional four-center GW does not accept factor-based cached/resolvent screening")
        from .conventional import g0w0_conventional
        return g0w0_conventional(mo_energy=mo_energy, mo_coeff=mo_coeff, nocc=nocc,
            eri=eri, eri_pair_matrix=eri_pair_matrix, direct_basis=direct_basis,
            fock_matrix=fock_matrix, hcore_matrix=hcore_matrix, density_matrix=density_matrix,
            nw=nw, eta=eta, orbs=orbs, diff_mode=diff_mode, qp_solver=qp_solver,
            mo_energy_poles=mo_energy_poles, screening_energy=screening_energy,
            g_orbitals=g_orbitals, screening_occupied=screening_occupied,
            screening_virtual=screening_virtual, linearized=linearized,
            evaluate_only=evaluate_only, qp_initial=qp_initial, qp_tol=qp_tol,
            max_memory=max_memory, reserved_bytes=reserved_bytes,
            screening_config=screening_config, v_mf_matrix=v_mf_matrix,
            _preparation=_preparation)
    if any(x is not None for x in (eri, eri_pair_matrix, direct_basis)):
        raise ValueError("Supply exactly one integral representation")
    mo_energy = jnp.asarray(mo_energy, dtype=jnp.float64)
    mo_coeff = jnp.asarray(mo_coeff, dtype=jnp.float64)
    df_factors = jnp.asarray(df_factors)
    nocc = int(nocc)
    nmo = mo_energy.shape[0]
    if screening_config is not None:
        from ..solvers import LinearSolverConfig
        if not isinstance(screening_config, LinearSolverConfig):
            raise TypeError("screening_config must be a shared LinearSolverConfig")
        if screening_config.method != "gmres":
            raise ValueError("Explicit screening_config supports GMRES; omit it for dense screening")
        if resolvent_expansion or resolvent_data_fixed is not None or wmn_fixed is not None or wmn_static_fixed is not None:
            raise ValueError("GMRES screening requires selected vertices, not full cached/resolvent screening")
    orbs = _requested_orbitals(orbs,nmo)

    if not 0 < nocc < nmo:
        raise ValueError("Restricted GW requires occupied and virtual orbitals")
    gi = _window_indices(g_orbitals, range(nmo), "g_orbitals")
    oi = _window_indices(screening_occupied, range(nocc), "screening_occupied")
    va = _window_indices(screening_virtual, range(nocc, nmo), "screening_virtual")
    oi_array, va_array = jnp.asarray(oi, dtype=jnp.int32), jnp.asarray(va, dtype=jnp.int32)
    selected = not (resolvent_expansion or resolvent_data_fixed is not None
                    or wmn_fixed is not None or wmn_static_fixed is not None
                    or jnp.iscomplexobj(df_factors))
    prepared = None
    if selected:
        from .integrals import prepare_restricted, screen_restricted
        data = _preparation if _preparation is not None else prepare_restricted(
            df_factors, mo_coeff, mo_energy, nocc=nocc, orbs=orbs, g_orbitals=gi,
            screening_occupied=oi, screening_virtual=va, nw=nw,
            fock=fock_matrix, hcore=hcore_matrix, density=density_matrix,
            v_mf=v_mf_matrix, max_memory=max_memory, reserved_bytes=reserved_bytes,
            screening_config=screening_config)
        vertices, plan = data["vertices"], data["plan"]
        b_mn, b_ov = None, vertices["b_ov"]
    else:
        from ._memory import full_gw_working_bytes, factor_promotion_bytes
        from .._memory import workspace_bytes, array_device
        available = workspace_bytes(max_memory, resident_bytes=reserved_bytes+
            array_storage_bytes(df_factors, mo_coeff, mo_energy, fock_matrix,
                                hcore_matrix, density_matrix, v_mf_matrix)+
            factor_promotion_bytes(df_factors, jnp.result_type(df_factors, mo_coeff)),
            device=array_device(df_factors, mo_coeff))
        needed = full_gw_working_bytes(df_factors.shape[0], nmo, nw,
            transitions=len(oi)*len(va) if resolvent_expansion or resolvent_data_fixed is not None else None,
            screening_size=len(oi)*len(va))
        if needed > available:
            raise ValueError("Full/resolvent GW forward workspace exceeds max_memory; use selected direct/GMRES screening")
        b_mn = _mo_factors(df_factors, mo_coeff)
        b_ov = b_mn[:, oi_array[:, None], va_array[None, :]]

    if selected:
        delta_v = data["delta_v"]
    else:
        if v_mf_matrix is None:
            v_mf = jnp.asarray(fock_matrix)-jnp.asarray(hcore_matrix)-build_j_from_df(df_factors, density_matrix)
        else:
            v_mf = jnp.asarray(v_mf_matrix)
            if v_mf.shape != fock_matrix.shape:
                raise ValueError("v_mf_matrix must match the AO Fock matrix")
        delta_v = jnp.diag(_exchange_mo(b_mn, nocc)-mo_coeff.T @ v_mf @ mo_coeff)

    poles = (
        mo_energy
        if mo_energy_poles is None
        else jnp.asarray(mo_energy_poles, dtype=jnp.float64)
    )
    screen = poles if screening_energy is None else jnp.asarray(screening_energy, dtype=poles.dtype)
    if poles.shape != mo_energy.shape or screen.shape != mo_energy.shape:
        raise ValueError("G/W energy spectra must have shape (nmo,)")
    from .outer_response import require_occupation_gap
    require_occupation_gap(poles, nocc, 'G0W0 G')
    require_occupation_gap(screen, nocc, 'G0W0 W')
    screen_pairs = (screen[oi_array], screen[va_array])
    ef = 0.5 * (jnp.max(poles[:nocc]) + jnp.min(poles[nocc:]))
    freqs, wts = scaled_legendre_grid(nw)

    def response_fn(omega):
        return rho_response_iw(omega, screen_pairs, b_ov, spin_factor=4.0)

    resolvent_data = resolvent_data_fixed
    if resolvent_expansion and resolvent_data is None:
        resolvent_data = rpa_resolvent(jnp.concatenate(screen_pairs), b_ov, b_mn, eta=eta)
    if selected:
        projected = data.get("projected")
        if projected is None:
            projected = screen_restricted(data, screen_pairs, jnp.concatenate([jnp.zeros(1), freqs]))
        prepared = dict(wmn_p=projected[1:].transpose(1, 0, 2),
            wmn_static=projected[0], b_pm=vertices["b_pm"].transpose(1, 0, 2),
            b_mp=vertices["b_mp"].transpose(1, 0, 2), p_index=jnp.asarray(orbs))
        wmn = wmn_static = None
    elif resolvent_data is not None:
        wmn = (screened_w_imag_resolvent(resolvent_data, freqs)
               if wmn_fixed is None else wmn_fixed)
        wmn_static = (screened_w_imag_resolvent(resolvent_data, jnp.zeros(1))[0]
                      if wmn_static_fixed is None else wmn_static_fixed)
    else:
        wmn = screened_w_imag_axis(b_mn, response_fn, freqs) if wmn_fixed is None else wmn_fixed
        wmn_static = (screened_w_imag_axis(b_mn, response_fn, jnp.zeros(1))[0]
                      if wmn_static_fixed is None else wmn_static_fixed)

    qp_energy, sigma_qp, converged_mask, converged, residual, weight, diagnostics = _qp_loop(
        mo_energy=poles,
        b_mn=b_mn,
        channels=((screen_pairs, b_ov, 2.0),),
        wmn=wmn,
        freqs=freqs,
        wts=wts,
        ef=ef,
        eta=eta,
        delta_v=delta_v,
        nocc=nocc,
        orbs=orbs,
        diff_mode=diff_mode,
        qp_solver=qp_solver,
        e_mf=mo_energy,
        linearized=linearized,
        evaluate_only=evaluate_only,
        g_orbitals=gi,
        return_weight=True,
        return_diagnostics=True,
        wmn_static=wmn_static,
        resolvent_data=resolvent_data,
        qp_initial=qp_initial,
        qp_tol=float(qp_tol),
        prepared=prepared,
        execution_plan=plan if selected else None,
    )
    return GWResult(
        mo_energy=qp_energy,
        mo_coeff=mo_coeff,
        converged=converged,
        sigma_qp=sigma_qp,
        converged_mask=converged_mask,
        nw=int(nw),
        qp_residual=residual,
        qp_computed_mask=jnp.zeros_like(qp_energy,dtype=bool).at[jnp.asarray(orbs)].set(not evaluate_only),
        screening_energy=screen,
        qp_weight=weight,
        g_orbitals=gi,
        screening_occupied=oi,
        screening_virtual=va,
        method="g0w0",
        **diagnostics,
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
    qp_solver: str = "secant",
    mo_energy_poles: tuple[Array, Array] | None = None,
    linearized: bool = False,
    evaluate_only: bool = False,
    screening_energy: tuple[Array, Array] | None = None,
    max_memory: float = DEFAULT_MAX_MEMORY_MB,
    screening_config=None,
    v_mf_matrix=None,
    reserved_bytes: int = 0,
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
    shape ``(2, nmo)`` / ``(2, nao, nmo)``. ``screening_energy`` optionally
    supplies independent alpha/beta spectra for W (evGW0); G still uses
    ``mo_energy_poles``. Both spin channels currently require occupied and
    virtual orbitals for the spin-resolved contour Fermi estimates.
    """
    df_factors = jnp.asarray(df_factors)
    if screening_config is not None:
        from ..solvers import LinearSolverConfig
        if not isinstance(screening_config, LinearSolverConfig):
            raise TypeError("screening_config must be a shared LinearSolverConfig")
        if screening_config.method != "gmres":
            raise ValueError("Explicit screening_config supports GMRES; omit it for dense screening")
    e_a, e_b = (jnp.asarray(e, dtype=jnp.float64) for e in mo_energy)
    p_a, p_b = (
        (e_a, e_b)
        if mo_energy_poles is None
        else (jnp.asarray(p, dtype=jnp.float64) for p in mo_energy_poles)
    )
    s_a, s_b = ((p_a, p_b) if screening_energy is None else
                tuple(jnp.asarray(e, dtype=jnp.float64) for e in screening_energy))
    if any(e.shape != e_a.shape for e in (e_b, p_a, p_b, s_a, s_b)):
        raise ValueError('Unrestricted G/W spectra must have the same (nmo,) shape.')
    c_a, c_b = (jnp.asarray(c, dtype=jnp.float64) for c in mo_coeff)
    nocc_a, nocc_b = int(nocc[0]), int(nocc[1])
    nmo = e_a.shape[0]
    if not (0 < nocc_a < nmo and 0 < nocc_b < nmo):
        raise ValueError('Unrestricted GW currently requires occupied and virtual orbitals in each spin.')
    orbs = _requested_orbitals(orbs,nmo)

    selected = not jnp.iscomplexobj(df_factors)
    prepared_a = prepared_b = None
    if selected:
        from ._memory import plan_gw
        from .integrals import prepare_selected_factors
        from .screened import screened_w_selected
        reserved = reserved_bytes+array_storage_bytes(*jax.tree.leaves(
            (mo_energy, fock_matrix, hcore_matrix, density_matrix, v_mf_matrix)))
        plan = plan_gw(df_factors, c_a, nocc=max(nocc_a, nocc_b),
            n_targets=2*len(orbs), n_g=nmo,
            screening_shape=(1, nocc_a*(nmo-nocc_a)+nocc_b*(nmo-nocc_b)), nw=nw,
            max_memory=max_memory, reserved_bytes=reserved+array_storage_bytes(c_b)+
                2*df_factors.shape[0]*len(orbs)*nmo*jnp.result_type(df_factors, c_a).itemsize,
            screening_config=screening_config)
        def prepare(c, no, extra):
            return prepare_selected_factors(df_factors, c, nocc=no, orbs=orbs,
                g_orbitals=tuple(range(nmo)), screening_occupied=tuple(range(no)),
                screening_virtual=tuple(range(no, nmo)), max_memory=max_memory,
                reserved_bytes=reserved+extra)
        vertices_a = prepare(c_a, nocc_a, array_storage_bytes(c_b))
        vertices_b = prepare(c_b, nocc_b, array_storage_bytes(c_a, *vertices_a.values()))
        b_a = b_b = None
        b_ov_a, b_ov_b = vertices_a["b_ov"], vertices_b["b_ov"]
        exchange_a, exchange_b = vertices_a["exchange"], vertices_b["exchange"]
    else:
        if screening_config is not None:
            raise NotImplementedError("Unrestricted screened-action planning requires real molecular factors")
        b_a, b_b = _mo_factors(df_factors, c_a), _mo_factors(df_factors, c_b)
        b_ov_a, b_ov_b = b_a[:, :nocc_a, nocc_a:], b_b[:, :nocc_b, nocc_b:]
        exchange_a = jnp.diag(_exchange_mo(b_a, nocc_a))[jnp.asarray(orbs)]
        exchange_b = jnp.diag(_exchange_mo(b_b, nocc_b))[jnp.asarray(orbs)]

    if v_mf_matrix is not None:
        potential_a, potential_b = map(jnp.asarray, v_mf_matrix)
    elif fock_matrix is None:
        if density_matrix is None:
            raise ValueError("Unrestricted G0W0 needs Fock matrices or HF spin densities")
        _, k_a = build_jk_from_df(df_factors, jnp.asarray(density_matrix[0]))
        _, k_b = build_jk_from_df(df_factors, jnp.asarray(density_matrix[1]))
        potential_a, potential_b = -k_a, -k_b
    else:
        if hcore_matrix is None or density_matrix is None:
            raise ValueError("fock_matrix, hcore_matrix and density_matrix must be provided together")
        dm_tot = jnp.asarray(density_matrix[0])+jnp.asarray(density_matrix[1])
        j_tot = build_j_from_df(df_factors, dm_tot)
        potential_a = jnp.asarray(fock_matrix[0])-jnp.asarray(hcore_matrix)-j_tot
        potential_b = jnp.asarray(fock_matrix[1])-jnp.asarray(hcore_matrix)-j_tot
    def correction(c, exchange, potential):
        columns = c[:, jnp.asarray(orbs)]
        delta = exchange-jnp.einsum("mp,mn,np->p", columns, potential, columns, precision="highest")
        return jnp.zeros(nmo, dtype=delta.dtype).at[jnp.asarray(orbs)].set(delta)
    delta_v_a = correction(c_a, exchange_a, potential_a)
    delta_v_b = correction(c_b, exchange_b, potential_b)

    from .outer_response import require_occupation_gap
    for spectrum, occupation in ((p_a, nocc_a), (p_b, nocc_b), (s_a, nocc_a), (s_b, nocc_b)):
        require_occupation_gap(spectrum, occupation, 'Unrestricted G0W0')
    ef_a = 0.5 * (jnp.max(p_a[:nocc_a]) + jnp.min(p_a[nocc_a:]))
    ef_b = 0.5 * (jnp.max(p_b[:nocc_b]) + jnp.min(p_b[nocc_b:]))

    freqs, wts = scaled_legendre_grid(nw)

    def response_fn(omega):
        return rho_response_iw(omega, s_a, b_ov_a, spin_factor=2.0) + rho_response_iw(
            omega, s_b, b_ov_b, spin_factor=2.0
        )

    channels = ((s_a, b_ov_a, 1.0), (s_b, b_ov_b, 1.0))
    if selected:
        # One dielectric factor per frequency serves BOTH orbital frames.
        joint = jnp.concatenate((vertices_a["b_mp"], vertices_b["b_mp"]), axis=1)
        projected = screened_w_selected(joint, response_fn,
            jnp.concatenate((jnp.zeros(1), freqs)), plan,
            channels=((s_a, b_ov_a, 2.), (s_b, b_ov_b, 2.)))
        def stack(vertices, values):
            return dict(wmn_p=values[1:].transpose(1, 0, 2), wmn_static=values[0],
                b_pm=vertices["b_pm"].transpose(1, 0, 2),
                b_mp=vertices["b_mp"].transpose(1, 0, 2), p_index=jnp.asarray(orbs))
        prepared_a = stack(vertices_a, projected[:, :len(orbs)])
        prepared_b = stack(vertices_b, projected[:, len(orbs):])
        wmn = wmn_b = static_a = static_b = None
    else:
        wmn = screened_w_imag_axis(b_a, response_fn, freqs)
        wmn_b = screened_w_imag_axis(b_b, response_fn, freqs)
        static_a = screened_w_imag_axis(b_a, response_fn, jnp.zeros(1))[0]
        static_b = screened_w_imag_axis(b_b, response_fn, jnp.zeros(1))[0]

    qp_a, sig_a, mask_a, conv_a, residual_a, weight_a, diagnostics_a = _qp_loop(
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
        qp_solver=qp_solver,
        wmn_static=static_a, prepared=prepared_a, execution_plan=plan if selected else None,
        e_mf=e_a,
        linearized=linearized,
        evaluate_only=evaluate_only,
        return_diagnostics=True,
    )
    qp_b, sig_b, mask_b, conv_b, residual_b, weight_b, diagnostics_b = _qp_loop(
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
        qp_solver=qp_solver,
        wmn_static=static_b, prepared=prepared_b, execution_plan=plan if selected else None,
        e_mf=e_b,
        linearized=linearized,
        evaluate_only=evaluate_only,
        return_diagnostics=True,
    )
    return GWResult(
        mo_energy=jnp.stack([qp_a, qp_b]),
        mo_coeff=jnp.stack([c_a, c_b]),
        converged=conv_a & conv_b,
        sigma_qp=jnp.stack([sig_a, sig_b]),
        converged_mask=jnp.stack([mask_a, mask_b]),
        nw=int(nw),
        qp_residual=jnp.stack([residual_a, residual_b]),
        qp_computed_mask=jnp.zeros((2,nmo),dtype=bool).at[:,jnp.asarray(orbs)].set(not evaluate_only),
        screening_energy=jnp.stack([s_a, s_b]),
        qp_weight=jnp.stack([weight_a, weight_b]),
        method='g0w0',
        **{key: jnp.stack((diagnostics_a[key], diagnostics_b[key])) for key in diagnostics_a},
    )


__all__ = ["g0w0_cd_restricted", "g0w0_cd_unrestricted"]


# Shared by the facade and fixed-orbital outer methods. Physical preparation
# remains a dynamic PyTree; only the declared algorithm/layout is static.
_run_g0w0_numerics = jax.jit(g0w0_cd_restricted, static_argnames=(
    "nocc", "nw", "orbs", "diff_mode", "qp_solver", "linearized", "evaluate_only",
    "g_orbitals", "screening_occupied", "screening_virtual", "resolvent_expansion", "qp_tol",
    "max_memory", "screening_config", "reserved_bytes",
))
