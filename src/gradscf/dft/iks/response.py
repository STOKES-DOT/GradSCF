"""Uncoupled fixed-orbital response; not an inverse-map or SCF derivative."""
from math import isfinite
from typing import NamedTuple

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


def _density_integrals(target, matrices, grid, *, with_l1=False):
    """One blocked AO pass for several density matrices and norm channels."""
    if grid is None:
        raise ValueError('Predicted-density assessment requires a grid')
    weights = jnp.asarray(grid[1] if isinstance(grid, tuple) else grid.weights)
    if jnp.iscomplexobj(weights):
        raise ValueError('Density-norm quadrature weights must be real')
    valid_weights = jnp.all(jnp.isfinite(weights)) & jnp.all(weights >= 0) & (jnp.sum(weights) > 0)
    components = jnp.stack([_components(matrix,target.overlap.shape[0],target.layout)
                            for matrix in matrices])
    def add(total, ao, deriv, lap, coords, weights):
        rho = jnp.einsum('gi,bcij,gj->bcg', ao, components, ao).real
        squared = jnp.sum(weights*rho**2,axis=(-1,-2))
        values = (jnp.stack((jnp.sum(weights*jnp.abs(rho),axis=(-1,-2)),squared))
                  if with_l1 else squared)
        return total+values
    shape = (2,len(matrices)) if with_l1 else (len(matrices),)
    initial = jnp.zeros(shape,dtype=components.real.dtype)
    if isinstance(grid, tuple):
        ao, weights = map(jnp.asarray, grid)
        if ao.ndim != 2 or ao.shape[1] != target.overlap.shape[0] or weights.shape != ao.shape[:1] or jnp.iscomplexobj(ao):
            raise ValueError('Grid requires real (npoints,nao) AO values and (npoints,) weights')
        value = add(initial, ao, None, None, None, weights)
    else:
        if grid.nao != target.overlap.shape[0]:
            raise ValueError('Grid AO dimension differs from the reference')
        value = grid.reduce(add,initial)
    return jnp.where(valid_weights,value,jnp.nan)


def density_norm2(reference, delta, grid, *, relative=False):
    """Squared spin/Pauli L2 errors; optionally divide by target squared norm.

    Absolute squared norms have units bohr^-3. Both relative errors use the
    same quadrature and target norm; no point-count or volume mean is taken.
    """
    target = reference.target
    matrices = [reference.density+delta-target.density,reference.density-target.density]
    if relative: matrices.append(target.density)
    value = _density_integrals(target,matrices,grid)
    return jnp.where(value[2]>0,value[:2]/value[2],jnp.nan) if relative else value


class DensityErrors(NamedTuple):
    l1: object
    l2_squared: object
    valid: object


def density_errors(target, density, grid):
    """Relative integral |delta rho| and relative integral delta rho^2.

    Each denominator is the corresponding target norm on the identical grid.
    For RKS, the L1 denominator equals the quadrature electron count. UKS and
    GKS retain spin/Pauli components. Both metrics are dimensionless.
    """
    moments = _density_integrals(target,[density-target.density,target.density],grid,with_l1=True)
    good = jnp.all(jnp.isfinite(moments)) & jnp.all(moments[:,1]>0)
    values = jnp.where(good,moments[:,0]/moments[:,1],jnp.nan)
    return DensityErrors(values[0],values[1],good)
