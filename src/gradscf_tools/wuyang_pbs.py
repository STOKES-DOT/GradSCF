"""Explicit smooth local potential bases for finite-basis Wu--Yang diagnostics."""
from __future__ import annotations

from typing import Iterable

import jax.numpy as jnp


DEFAULT_EXPONENTS = (0.2, 1.0, 5.0, 25.0)  # bohr^-2
DEFAULT_MONOMIALS = ("1", "dx", "dy", "dz")


def atom_centered_gaussian_pbs(centers_bohr, *, exponents: Iterable[float] = DEFAULT_EXPONENTS,
                               monomials: Iterable[str] = DEFAULT_MONOMIALS):
    """Return ``(basis_callable, settings)`` for a fixed atom-centered PBS.

    The callable has the ``iks.Potential`` signature ``(params, coords)`` and
    returns one unscaled feature for each exponent/monomial/atom.  Coordinates
    and exponents are in Bohr and Bohr^-2.  No density, label, or fitted
    normalization enters this basis.
    """
    centers = jnp.asarray(centers_bohr)
    exponents = tuple(float(value) for value in exponents)
    monomials = tuple(str(value) for value in monomials)
    if centers.ndim != 2 or centers.shape[1] != 3 or not centers.shape[0]:
        raise ValueError("centers_bohr must have shape (natoms, 3)")
    if not exponents or any(value <= 0 for value in exponents):
        raise ValueError("Gaussian exponents must be positive")
    if not monomials or any(value not in {"1", "dx", "dy", "dz"} for value in monomials):
        raise ValueError("monomials must be drawn from 1, dx, dy, dz")

    def basis(_params, coords):
        coords = jnp.asarray(coords)
        delta = coords[:, None, :] - centers[None, :, :]
        radial = jnp.exp(-jnp.sum(delta * delta, axis=-1)[..., None]
                          * jnp.asarray(exponents)[None, None, :])
        channels = {
            "1": jnp.ones_like(delta[..., 0]),
            "dx": delta[..., 0],
            "dy": delta[..., 1],
            "dz": delta[..., 2],
        }
        return jnp.stack(
            [radial[:, atom, index] * channels[monomial][:, atom]
             for atom in range(centers.shape[0])
             for index in range(len(exponents)) for monomial in monomials],
            axis=1,
        )

    settings = {
        "kind": "atom_centered_gaussian",
        "coordinate_units": "bohr",
        "exponent_units": "bohr^-2",
        "exponents": list(exponents),
        "monomials": list(monomials),
        "atoms": int(centers.shape[0]),
        "nparam": int(centers.shape[0] * len(exponents) * len(monomials)),
    }
    return basis, settings


__all__ = ["DEFAULT_EXPONENTS", "DEFAULT_MONOMIALS", "atom_centered_gaussian_pbs"]
