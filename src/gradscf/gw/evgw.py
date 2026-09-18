"""Eigenvalue-only self-consistent GW (evGW) -- Stage 3.

evGW re-injects the quasiparticle energies into both the Green's function
and the RPA screened interaction (orbitals and density frozen) until the
QP spectrum is self-consistent:

    e_{n+1} = G0W0[e_n]   (fixed-point iteration with optional damping)

The drivers simply re-evaluate the G0W0 machinery
(:func:`gradscf.gw.g0w0_cd_restricted` / ``_unrestricted`` /
:func:`gradscf.gw.pbc.krgw.g0w0_cd_gamma`) with ``mo_energy`` replaced by
the current QP spectrum, so every ingredient (response, W, self-energy,
QP solver, AD rules) is inherited unchanged.

References
----------
- M. S. Hybertsen and S. G. Louie, Phys. Rev. B 34, 5390 (1986).
  DOI:10.1103/PhysRevB.34.5390
- M. Shishkin and G. Kresse, Phys. Rev. B 75, 235102 (2007).
  DOI:10.1103/PhysRevB.75.235102 (evGW convergence behavior)
- X. Blase, C. Attaccalite and V. Olevano, Phys. Rev. B 83, 115103 (2011).
  DOI:10.1103/PhysRevB.83.115103
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from jaxtyping import Array

from .g0w0 import g0w0_cd_restricted, g0w0_cd_unrestricted
from .types import GWResult


def _evgw_loop(
    driver: Callable[..., GWResult],
    mo_energy_mf,
    *,
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 0.0,
    **kwargs,
) -> GWResult:
    """Shared fixed-point iteration around a G0W0 driver.

    ``mo_energy_mf`` stays the fixed mean-field base of the QP equation;
    only the pole spectrum (``mo_energy_poles`` of the drivers) is iterated.
    Each iteration uses the smooth Z-factor linearized update (the graphical
    solve jumps branches in pole-dense regions); the converged linearized
    spectrum IS the evGW result (standard Z-update evGW convention; see
    Shishkin & Kresse, PRB 75, 235102 (2007)).
    """
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in [0, 1).")
    e_old = jnp.asarray(mo_energy_mf, dtype=jnp.float64)
    result = None
    for _ in range(int(max_iter)):
        result = driver(mo_energy_poles=e_old, linearized=True, **kwargs)
        e_new = jnp.asarray(result.mo_energy, dtype=jnp.float64)
        if damping > 0.0:
            e_new = (1.0 - damping) * e_new + damping * e_old
        delta = jnp.max(jnp.abs(e_new - e_old))
        e_old = e_new
        if float(delta) < tol:
            return result
    raise ArithmeticError(
        f"evGW did not converge in {max_iter} iterations (last max|de| = "
        f"{float(delta):.3e} > {tol}). Increase max_iter or damping; "
        "no fallback result is returned."
    )


def evgw_cd_restricted(
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
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 0.0,
) -> GWResult:
    """Spin-restricted evGW with contour deformation.

    Parameters are those of :func:`gradscf.gw.g0w0_cd_restricted` plus the
    iteration controls ``max_iter`` / ``tol`` / ``damping``.  Note the evGW
    loop evaluates every orbital each iteration; restricting ``orbs``
    freezes the remaining energies at their mean-field values.
    """
    return _evgw_loop(
        g0w0_cd_restricted,
        mo_energy,
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        nocc=int(nocc),
        df_factors=df_factors,
        fock_matrix=fock_matrix,
        hcore_matrix=hcore_matrix,
        density_matrix=density_matrix,
        nw=int(nw),
        eta=float(eta),
        orbs=orbs,
        diff_mode=diff_mode,
        max_iter=int(max_iter),
        tol=float(tol),
        damping=float(damping),
    )


def evgw_cd_unrestricted(
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
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 0.0,
) -> GWResult:
    """Spin-unrestricted evGW with contour deformation."""
    e0 = jnp.stack([jnp.asarray(mo_energy[0]), jnp.asarray(mo_energy[1])])

    def driver(mo_energy_poles, **kw):
        return g0w0_cd_unrestricted(mo_energy_poles=(mo_energy_poles[0], mo_energy_poles[1]), **kw)

    return _evgw_loop(
        driver,
        e0,
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        nocc=(int(nocc[0]), int(nocc[1])),
        df_factors=df_factors,
        fock_matrix=fock_matrix,
        hcore_matrix=hcore_matrix,
        density_matrix=density_matrix,
        nw=int(nw),
        eta=float(eta),
        orbs=orbs,
        diff_mode=diff_mode,
        max_iter=int(max_iter),
        tol=float(tol),
        damping=float(damping),
    )


__all__ = ["evgw_cd_restricted", "evgw_cd_unrestricted"]
