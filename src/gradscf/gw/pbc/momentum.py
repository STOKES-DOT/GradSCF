"""Momentum-conservation index tables for k-point GW.

In a periodic GW calculation the density response is evaluated at every
momentum transfer ``q = kL`` from pair transitions (ki, kj) satisfying

    -ki + kj + kL = G_reciprocal   (i.e. kj = ki - kL modulo the k mesh)

and the self-energy at kn contracts orbital m at the inverse partner
``table[km, q] = kn`` (``km = kn + q``). This module precomputes the mapping on a uniform
k mesh, following the table construction of PySCF ``pbc.gw.krgw_ac``
(``kpti_kptj`` / ``kidx``).

References
----------
- J. Deslippe et al., Comput. Phys. Commun. 183, 1269 (2012).
  DOI:10.1016/j.cpc.2011.12.023
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
"""

from __future__ import annotations

import numpy as np


def wrap_to_kmesh(kpts: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Wrap k points into [-0.5, 0.5) in fractional (mesh) coordinates."""
    fractional = np.asarray(kpts, dtype=float)
    wrapped = fractional - np.floor(fractional + 0.5)
    wrapped[np.abs(wrapped + 0.5) < tol] = -0.5
    return wrapped


def momentum_transfer_table(kpts: np.ndarray, *, tol: float = 1e-6) -> np.ndarray:
    """Index table ``kj_of_ki_q[ki, q] = kj`` with ``kj = ki - q`` on the mesh.

    Parameters
    ----------
    kpts:
        ``(nk, 3)`` k points in fractional coordinates of the reciprocal
        lattice (any consistent unit works as long as the mesh is uniform).
    tol:
        Matching tolerance in fractional units.

    Returns
    -------
    Integer array of shape ``(nk, nk)``; ``kj_of_ki_q[ki, q]`` is the index
    of ``kpts[ki] - kpts[q]`` in ``kpts``.  Raises if a partner is missing
    (non-uniform or incomplete mesh).
    """
    kpts = np.asarray(kpts, dtype=float)
    nk = kpts.shape[0]
    wrapped = wrap_to_kmesh(kpts, tol=tol)
    table = np.full((nk, nk), -1, dtype=int)
    for ki in range(nk):
        for q in range(nk):
            target = wrap_to_kmesh(wrapped[ki] - wrapped[q], tol=tol)
            match = np.where(np.all(np.abs(wrapped - target) < tol, axis=1))[0]
            if match.size != 1:
                raise ValueError(
                    f"k mesh is not closed under momentum transfer: "
                    f"kpts[{ki}] - kpts[{q}] matches {match.size} points."
                )
            table[ki, q] = int(match[0])
    return table


__all__ = ["wrap_to_kmesh", "momentum_transfer_table"]
