"""Imaginary-frequency quadrature grids for GW calculations.

The GW self-energy is integrated along the imaginary frequency axis with a
scaled Gauss-Legendre quadrature that maps the [-1, 1] roots onto [0, inf).
The transformation matches the convention used by PySCF
(``pyscf.gw.gw_ac._get_scaled_legendre_roots``) so results are directly
comparable.

References
----------
- X. Ren et al., "Resolution-of-identity approach to Hartree-Fock, hybrid
  density functionals, RPA, MP2 and GW with numeric atom-centered orbital
  basis functions", New J. Phys. 14, 053020 (2012).
  DOI:10.1088/1367-2630/14/5/053020 (quadrature choice, see also
  www.cond-mat.de/events/correl19/manuscripts/ren.pdf)
- L. Hedin, "New Method for Calculating the One-Particle Green's Function
  with Application to the Electron-Gas Problem", Phys. Rev. 139, A796
  (1965). DOI:10.1103/PhysRev.139.A796
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jaxtyping import Array


def scaled_legendre_grid(nw: int, x0: float = 0.5) -> tuple[Array, Array]:
    """Scaled Gauss-Legendre grid on [0, inf) for imaginary-axis integration.

    The nw Legendre roots x in [-1, 1] are mapped via

        omega = x0 * (1 + x) / (1 - x)
        w     = w_leg * 2 * x0 / (1 - x)**2

    Parameters
    ----------
    nw:
        Number of quadrature points (PySCF default is 100).
    x0:
        Scaling parameter (0.5 following the PySCF convention).

    Returns
    -------
    freqs, wts:
        1D float64 arrays of length ``nw``.
    """
    nw = int(nw)
    if nw <= 0:
        raise ValueError("nw must be a positive integer")
    # leggauss is evaluated once on the host; the grid itself is static and
    # carries no AD sensitivity.
    x, w = np.polynomial.legendre.leggauss(nw)
    freqs = x0 * (1.0 + x) / (1.0 - x)
    wts = w * 2.0 * x0 / (1.0 - x) ** 2
    return jnp.asarray(freqs, dtype=jnp.float64), jnp.asarray(wts, dtype=jnp.float64)


__all__ = ["scaled_legendre_grid"]
