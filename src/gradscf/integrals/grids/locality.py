"""Opt-in Cartesian AO locality with dynamic, conservative block envelopes.

Grid permutation is exact and reversible. A positive AO cutoff is instead a
piecewise-smooth approximation: the same current-parameter support is used by
primal and AD evaluations, with no derivative defined at support changes. The
cutoff bounds individual requested AO channels, in their respective atomic
units, and does not bound a density, XC energy, or XC potential error. Even
coefficient-response support is not an arbitrary-order parameter derivative
bound. Use cutoff=0 for exact derivatives outside a validated screened range.

Only topology and fixed capacities are static. Geometry, exponents,
coefficients, coordinates and weights must be passed anew on every call;
active indices and envelopes must never enter a persistent geometry cache.
"""
from __future__ import annotations

import math

import jax.numpy as jnp

from .ao import _evaluate_cartesian_ao_kernel, _primitive_cartesian_norm_array


def spatial_grid_permutation(coords, *, cell_size: float = 4.0):
    """Return spatial order and its inverse, using cubic cells in Bohr.

    The consumer must gather coordinates and weights with the same order and
    restore any grid-indexed output with the inverse before returning it to
    external/global models. Integer sorting leaves all gathered numeric
    coordinates and weights differentiable. Equal points retain input order.
    """
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("Spatial grid cell_size must be finite and positive.")
    coords = jnp.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Spatial grid coordinates must have shape (ngrids,3).")
    original = jnp.arange(coords.shape[0], dtype=jnp.int32)
    cells = jnp.floor(coords / cell_size)
    order = jnp.lexsort((original, coords[:, 2], coords[:, 1], coords[:, 0],
                        cells[:, 2], cells[:, 1], cells[:, 0]))
    inverse = jnp.zeros_like(order).at[order].set(original)
    return order, inverse


def _gaussian_power_bound(lower, upper, exponents, powers):
    """Exact maximum of r**k exp(-alpha r**2) on nonnegative [lo,hi]."""
    # A primitive's sole nonnegative stationary point is sqrt(k/(2*alpha)).
    # k=0 therefore selects the near endpoint, including intervals at zero.
    radius = jnp.clip(jnp.sqrt(powers / (2 * exponents)), lower, upper)
    return radius**powers * jnp.exp(-exponents * radius**2)


def ao_block_bounds(provider, coords, weights, *, deriv: int = 0,
                    coefficient_response: bool = False, point_mask=None):
    """Bound each AO over a block box, including requested spatial channels.

    For deriv=1 the bound covers value/dx/dy/dz; deriv=2 also covers the
    Laplacian. Primitive absolute envelopes are summed, so cancellation of
    contraction coefficients cannot invalidate support. All input points are
    real by default, including zero/negative weights. Only an explicit boolean
    point_mask excludes padding; weight values do not define support. Workspace scales
    with AO x padded primitives, independently of the number of grid points.

    The default bounds actual AO amplitudes and is homogeneous in contraction
    coefficients. Internal screening consumers use coefficient_response=True:
    each active primitive is weighted by max(abs(coefficient), 1), retaining
    its unit coefficient-response envelope even when its current coefficient
    is zero. This also covers coefficient response of the requested spatial
    channels, but does not bound exponent/mixed or higher spatial derivatives,
    nor their density-dependent energy/field contractions.
    """
    if deriv not in (0, 1, 2):
        raise ValueError("AO locality requires deriv in (0,1,2).")
    coords, weights = jnp.asarray(coords), jnp.asarray(weights)
    if coords.ndim != 2 or coords.shape[1] != 3 or weights.shape != coords.shape[:1]:
        raise ValueError("AO locality needs coordinates (ngrids,3) and weights (ngrids,).")
    valid_points = (jnp.ones(weights.shape, dtype=bool) if point_mask is None
                    else jnp.asarray(point_mask))
    if valid_points.shape != weights.shape or valid_points.dtype != jnp.bool_:
        raise ValueError("AO locality point_mask must be boolean with shape (ngrids,).")
    centers = jnp.asarray(provider.centers)
    exponents = jnp.asarray(provider.exponents)
    coefficients = jnp.asarray(provider.coefficients)
    nao = centers.shape[0]
    dtype = jnp.result_type(coords, centers, exponents, coefficients)
    if coords.shape[0] == 0 or nao == 0 or exponents.shape[1] == 0:
        return jnp.zeros((nao,), dtype=dtype)
    coords, centers = coords.astype(dtype), centers.astype(dtype)
    exponents, coefficients = exponents.astype(dtype), coefficients.astype(dtype)
    angulars = jnp.asarray(provider.angulars, dtype=jnp.int32).reshape(nao, 3)
    nprims = jnp.asarray(provider.nprims, dtype=jnp.int32)
    nonempty = jnp.any(valid_points)
    box_min = jnp.min(jnp.where(valid_points[:, None], coords, jnp.inf), axis=0)
    box_max = jnp.max(jnp.where(valid_points[:, None], coords, -jnp.inf), axis=0)
    box_min = jnp.where(nonempty, box_min, jnp.zeros_like(box_min))
    box_max = jnp.where(nonempty, box_max, jnp.zeros_like(box_max))
    relative_min = box_min[None, :] - centers
    relative_max = box_max[None, :] - centers
    lower = jnp.where((relative_min <= 0) & (relative_max >= 0), 0.,
                      jnp.minimum(jnp.abs(relative_min), jnp.abs(relative_max)))[:, None, :]
    upper = jnp.maximum(jnp.abs(relative_min), jnp.abs(relative_max))[:, None, :]

    primitives = jnp.arange(exponents.shape[1])[None, :] < nprims[:, None]
    physical = jnp.isfinite(exponents) & (exponents > 0)
    safe_exponents = jnp.where(primitives & physical, exponents, 1.)
    alpha = safe_exponents[:, :, None]
    powers = angulars[:, None, :]

    def envelope(k):
        return _gaussian_power_bound(lower, upper, alpha, jnp.maximum(k, 0))

    value = envelope(powers)
    normalization = _primitive_cartesian_norm_array(safe_exponents, angulars)
    if coefficient_response:
        # A zero *active* coefficient still has a nonzero basis derivative.
        # Only nprims determines whether a primitive is physical or padding.
        coefficients = jnp.maximum(jnp.abs(coefficients), 1.)
    prefactor = jnp.where(primitives, jnp.abs(coefficients * normalization), 0.)
    value_bound = jnp.sum(prefactor * jnp.prod(value, axis=-1), axis=-1)
    result = value_bound
    if deriv:
        # Products of the untouched directions avoid dividing by an envelope
        # that may vanish at a polynomial node or underflow in a distant tail.
        other_axes = jnp.stack((value[:, :, 1]*value[:, :, 2],
                                value[:, :, 0]*value[:, :, 2],
                                value[:, :, 0]*value[:, :, 1]), axis=-1)
        gradient = powers*envelope(powers-1) + 2*alpha*envelope(powers+1)
        gradient_bound = jnp.sum(prefactor[:, :, None]*gradient*other_axes, axis=1)
        result = jnp.maximum(result, jnp.max(gradient_bound, axis=-1))
        if deriv == 2:
            second = (powers*jnp.maximum(powers-1, 0)*envelope(powers-2)
                      + 2*alpha*(2*powers+1)*value
                      + 4*alpha**2*envelope(powers+2))
            laplacian_bound = jnp.sum(
                prefactor*jnp.sum(second*other_axes, axis=-1), axis=-1)
            result = jnp.maximum(result, laplacian_bound)
    invalid = (jnp.any(primitives & (~physical | ~jnp.isfinite(coefficients)), axis=-1)
               | jnp.any(~jnp.isfinite(centers), axis=-1)
               | jnp.any(valid_points[:, None] & ~jnp.isfinite(coords)))
    # Never prune an invalid/nonfinite envelope and thereby hide its failure.
    result = jnp.where(invalid | ~jnp.isfinite(result), jnp.inf, result)
    return jnp.where(nonempty, result, jnp.zeros_like(result))


def select_local_aos(bounds, *, cutoff: float, capacity: int):
    """Return fixed-capacity AO indices, valid-slot mask and overflow flag.

    cutoff=0 retains every AO, including exact-zero envelopes. The consumer
    must use its exact dense path when overflow is true; silently truncating
    an overflowing selection is incorrect. Indices in unused slots are zero
    and must be masked during both AO evaluation and matrix gather/scatter.
    """
    if not math.isfinite(cutoff) or cutoff < 0 or capacity < 0 or int(capacity) != capacity:
        raise ValueError("AO locality requires a finite nonnegative cutoff and integer capacity.")
    bounds = jnp.asarray(bounds)
    if bounds.ndim != 1:
        raise ValueError("AO locality bounds must be one-dimensional.")
    active = (jnp.ones(bounds.shape, dtype=bool) if cutoff == 0 else
              (bounds > cutoff) | ~jnp.isfinite(bounds))
    count = jnp.sum(active, dtype=jnp.int32)
    indices = jnp.nonzero(active, size=int(capacity), fill_value=0)[0]
    valid = jnp.arange(int(capacity)) < count
    return indices, valid, count > capacity


def evaluate_local_ao(provider, coords, weights, indices, valid, *, deriv: int = 0):
    """Evaluate only gathered AO columns, preserving the usual block tuple.

    The kernel receives capacity AO rows rather than full-basis rows with
    zeroed outputs. Gather density/MO rows with these same indices; scatter
    local matrix contributions with masking of padded duplicate indices.
    """
    if deriv not in (0, 1, 2):
        raise ValueError("AO locality requires deriv in (0,1,2).")
    coords, weights = jnp.asarray(coords), jnp.asarray(weights)
    indices, valid = jnp.asarray(indices), jnp.asarray(valid)
    if indices.ndim != 1 or valid.shape != indices.shape:
        raise ValueError("Local AO indices and valid mask must have matching one-dimensional shapes.")
    capacity = indices.shape[0]
    if capacity == 0:
        dtype = jnp.result_type(coords, provider.exponents, provider.coefficients)
        values = jnp.zeros((coords.shape[0], 0), dtype=dtype)
        derivative = jnp.zeros((4, coords.shape[0], 0), dtype=dtype) if deriv else None
        return values, derivative, values if deriv == 2 else None, coords, weights
    nao = provider.centers.shape[0]
    values = _evaluate_cartesian_ao_kernel(
        coords, jnp.take(provider.centers, indices, axis=0),
        jnp.take(provider.exponents, indices, axis=0),
        jnp.take(provider.coefficients, indices, axis=0),
        jnp.take(jnp.asarray(provider.nprims, dtype=jnp.int32), indices, axis=0),
        jnp.take(jnp.asarray(provider.angulars, dtype=jnp.int32).reshape(nao, 3),
                 indices, axis=0), deriv=deriv,
    )
    values = values * valid
    if deriv == 0:
        return values, None, None, coords, weights
    return values[0], values[:4], values[4] if deriv == 2 else None, coords, weights
