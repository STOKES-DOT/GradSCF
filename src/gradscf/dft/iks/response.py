"""Uncoupled fixed-orbital response; not an inverse-map or SCF derivative."""
from math import isfinite

import jax.numpy as jnp

from .orbital import OrbitalReference
from .types import _check, _components


def orbital_response(reference, fock, gap_floor, *, with_density=True):
    """Return delta D, sum|kappa_ai|^2 and occupation-chart validity.

    kappa_ai=-F_ai/max(epsilon_a-epsilon_i,gap_floor), with gap_floor in Ha.
    All ordered occupied/virtual pairs are retained without data-dependent
    indexing. Intrablock degeneracies do not select individual eigenvectors.
    Fractional occupations need a different response model and are rejected.
    """
    if not isinstance(reference, OrbitalReference):
        raise ValueError('Stationarity/response requires a declared orbital reference')
    if not isfinite(float(gap_floor)) or gap_floor <= 0:
        raise ValueError('gap_floor must be positive and finite in Hartree')
    valid = reference.integer_occupations
    _check(valid, 'The occupied/virtual response requires integer, not fractional occupations')
    c, f, eps = reference.coefficients, reference.occupations, reference.energies
    maximum = 2 if reference.target.layout == 'rks' else 1
    occupied = f > maximum/2
    pairs = (~occupied)[..., :, None] & occupied[..., None, :]
    fock_mo = c.conj().swapaxes(-1, -2)@fock@c
    gaps = jnp.maximum(eps[..., :, None]-eps[..., None, :], gap_floor)
    kappa = jnp.where(pairs, -fock_mo/gaps, 0.)
    delta = None
    if with_density:
        first = kappa*f[..., None, :]
        delta = c@(first+first.conj().swapaxes(-1, -2))@c.conj().swapaxes(-1, -2)
    return delta, jnp.sum(jnp.abs(kappa)**2), valid


def density_norm2(reference, delta, grid):
    """Spin/Pauli-resolved integral of |n_s+delta n-n_target|^2.

    grid is an AOGrid provider or an explicit (real grid AO, weights) pair.
    Density units are bohr^-3, so the squared L2 norm has units bohr^-3.
    No electron, volume or component normalization is implicit.
    """
    if grid is None:
        raise ValueError('Predicted-density assessment requires a grid')
    weights = jnp.asarray(grid[1] if isinstance(grid, tuple) else grid.weights)
    if jnp.iscomplexobj(weights):
        raise ValueError('Density-norm quadrature weights must be real')
    valid_weights = jnp.all(jnp.isfinite(weights)) & jnp.all(weights >= 0) & (jnp.sum(weights) > 0)
    target = reference.target
    difference = reference.density+delta-target.density
    components = _components(difference, target.overlap.shape[0], target.layout)
    baseline = _components(reference.density-target.density, target.overlap.shape[0], target.layout)
    def add(total, ao, deriv, lap, coords, weights):
        rho = jnp.einsum('gi,cij,gj->cg', ao, components, ao).real
        ref_rho = jnp.einsum('gi,cij,gj->cg', ao, baseline, ao).real
        return total+jnp.stack((jnp.sum(weights*rho**2), jnp.sum(weights*ref_rho**2)))
    initial = jnp.zeros(2, dtype=difference.real.dtype)
    if isinstance(grid, tuple):
        ao, weights = map(jnp.asarray, grid)
        if ao.ndim != 2 or ao.shape[1] != target.overlap.shape[0] or weights.shape != ao.shape[:1] or jnp.iscomplexobj(ao):
            raise ValueError('Grid requires real (npoints,nao) AO values and (npoints,) weights')
        value = add(initial, ao, None, None, None, weights)
        return jnp.where(valid_weights, value, jnp.nan)
    if grid.nao != target.overlap.shape[0]:
        raise ValueError('Grid AO dimension differs from the reference')
    return jnp.where(valid_weights, grid.reduce(add, initial), jnp.nan)
