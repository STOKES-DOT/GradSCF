"""Pure objective composition shared by parameter training and agent scoring."""
from math import isfinite

import jax.numpy as jnp

from .assessment import evaluate


def potential_error(predicted, target, weights, *, metric='grid_l2', density=None, gauge='fixed'):
    """Real local-potential metrics, keeping every spin/Pauli component.

    Arrays are (npoints,) or (ncomponents,npoints). grid_l2 is a quadrature-
    weighted point mean, summed over components (Ha^2). density_integral is
    sum_c [integral n_c*(v_c-v_ref,c)]^2 (Ha^2), not pointwise MSE.
    A fixed common gauge is mandatory for the latter. Nonfinite values stay
    nonfinite; this function neither constructs nor infers a grid potential.
    """
    predicted, target, weights = map(jnp.asarray, (predicted, target, weights))
    if (predicted.ndim not in (1, 2) or target.ndim not in (1, 2)
            or weights.ndim != 1 or predicted.shape[-1] != weights.size
            or target.shape[-1] != weights.size
            or any(jnp.iscomplexobj(v) for v in (predicted, target, weights))):
        raise ValueError('Use real component-by-grid potentials and a one-dimensional quadrature')
    if predicted.ndim == target.ndim == 2 and predicted.shape != target.shape:
        raise ValueError('Potential spin/Pauli component counts differ')
    difference = jnp.atleast_2d(predicted)-jnp.atleast_2d(target)
    norm = jnp.sum(weights)
    good_weights = jnp.all(jnp.isfinite(weights)) & jnp.all(weights >= 0) & (norm > 0)
    if gauge == 'weighted_mean':
        if metric != 'grid_l2':
            raise ValueError('A density-integral potential loss requires a fixed gauge')
        difference -= jnp.sum(weights*difference, axis=-1, keepdims=True)/norm
    elif gauge != 'fixed':
        raise ValueError('Use fixed or weighted_mean potential gauge')
    if metric == 'grid_l2':
        result = jnp.sum(weights*difference**2)/norm
    elif metric == 'density_integral':
        if density is None:
            raise ValueError('density_integral requires component-resolved density')
        density = jnp.atleast_2d(jnp.asarray(density))
        if density.shape != difference.shape or jnp.iscomplexobj(density):
            raise ValueError('Density must match the potential component-by-grid shape')
        result = jnp.sum(jnp.sum(weights*density*difference, axis=-1)**2)
    else:
        raise ValueError('Use grid_l2 or density_integral potential metric')
    return jnp.where(good_weights, result, jnp.nan)


_METRICS = {'energy': ('mse', 'mae'), 'operator': ('mse',),
            'xc_potential': ('mse',), 'stationarity': ('norm2',), 'predicted_density': ('norm2',)}


def loss(reference, energy_fn, params=None, *, weights=None, scales=None, grid=None, gap_floor=.05,
         xc_energy_fn=None):
    """Return (dimensionless weighted loss, raw Evaluation), with live AD.

    Scales use Ha for energy/operator, dimensionless rotation for stationarity,
    and bohr^-3/2 for the density L2 norm. Defaults are one in those units.
    Parameters, target values and raw metrics are never normalized in place.
    No reward transformation or candidate SCF is performed.
    """
    weights = {'energy': {'mse': 1.}} if weights is None else weights
    scales = {} if scales is None else scales
    if set(weights)-set(_METRICS) or set(scales)-set(_METRICS):
        raise ValueError('Unknown fixed-reference objective or normalization scale')
    active = {}
    for name, terms in weights.items():
        if set(terms)-set(_METRICS[name]):
            raise ValueError(f'Unsupported {name} metric')
        for term, weight in terms.items():
            if not isfinite(float(weight)) or weight < 0:
                raise ValueError('Objective weights must be finite and nonnegative')
            if weight > 0: active.setdefault(name, {})[term] = float(weight)
    for scale in scales.values():
        if not isfinite(float(scale)) or scale <= 0:
            raise ValueError('Objective scales must be finite and positive')
    if not active:
        raise ValueError('At least one objective must have positive weight')
    if 'energy' in active and reference.target.energy is None:
        raise ValueError('Energy objective requires a total-energy label')
    result = evaluate(reference, energy_fn, params, quantities=tuple(active), grid=grid, gap_floor=gap_floor,
                      xc_energy_fn=xc_energy_fn)
    value = jnp.zeros((), result.total_energy.dtype)
    squared = {'operator': result.operator_mse, 'xc_potential': result.xc_potential_mse,
               'stationarity': result.rotation_norm2,
               'predicted_density': result.predicted_density_norm2}
    for name, terms in active.items():
        scale = scales.get(name, 1.)
        if name == 'energy':
            error = result.energy_error/scale
            value += terms.get('mse', 0.)*error**2+terms.get('mae', 0.)*jnp.abs(error)
        else:
            value += sum(terms.values())*squared[name]/scale**2
    return jnp.where(result.valid, value, jnp.nan), result
