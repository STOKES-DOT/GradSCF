"""Result container for GW calculations."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..scf._pytree import pytree_dataclass


@pytree_dataclass(static_fields=("converged", "nw"))
@dataclass(frozen=True)
class GWResult:
    """Quasiparticle energies from a (G0W0/evGW/qsGW/scGW) calculation.

    Attributes
    ----------
    mo_energy:
        Quasiparticle energies, shape ``(nmo,)`` (restricted) or
        ``(2, nmo)`` (unrestricted).  Orbitals not included in the
        self-energy evaluation are prefilled with the mean-field values
        (unlike PySCF, which fills them with zeros).
    mo_coeff:
        Orbital coefficients (identical to the mean-field input for G0W0;
        updated for qsGW/scGW).
    converged:
        True when every requested quasiparticle equation converged
        (|dx| < tol of the secant iteration).
    converged_mask:
        Per-orbital convergence flags (False where the secant iteration did
        not meet the step tolerance; the returned energy is then the best
        residual iterate).  Orbitals not requested are marked True.
    sigma_qp:
        Diagonal correlation self-energy at the converged QP energies
        (complex), same shape as ``mo_energy``; ``None`` when not stored.
    nw:
        Number of imaginary-axis quadrature points used.
    """

    mo_energy: jnp.ndarray
    mo_coeff: jnp.ndarray
    converged: bool
    sigma_qp: jnp.ndarray | None = None
    converged_mask: jnp.ndarray | None = None
    nw: int | None = None


__all__ = ["GWResult"]
