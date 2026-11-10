"""Thin adaptation of the shared iKS objective to existing Trainer metrics."""
from dataclasses import fields, replace

import jax.numpy as jnp


def inverse_loss(params, functional, dataset, config):
    from gradscf.dft import iks
    from .targets import _MOLECULAR_METRIC_KEYS, _SCF_METRIC_KEYS

    if config.mode != 'fixed_density':
        raise ValueError('Inverse-reference samples currently require fixed_density; use ordinary samples for SCF validation')
    allowed = {'e0_total_mse_weight', 'e0_total_mae_weight', 'xc_potential_mse_weight',
               'inverse_operator_mse_weight', 'inverse_stationarity_weight', 'inverse_density_weight'}
    unsupported = [f.name for f in fields(config) if f.name.endswith('_weight')
                   and getattr(config, f.name) and f.name not in allowed]
    if unsupported or config.e0_normalization != 'none':
        raise ValueError('Unsupported inverse objective; use explicit fixed-reference weights and scales')
    weights = {'energy': {'mse': config.e0_total_mse_weight, 'mae': config.e0_total_mae_weight},
               'operator': {'mse': config.inverse_operator_mse_weight},
               'xc_potential': {'mse': config.xc_potential_mse_weight},
               'stationarity': {'norm2': config.inverse_stationarity_weight},
               'predicted_density': {'norm2': config.inverse_density_weight}}
    rows, losses = [], []
    for datum in dataset:
        ref = datum.inverse
        if ref is None:
            raise ValueError('Use a homogeneous inverse-reference dataset')
        if config.xc_potential_mse_weight and datum.target_xc_potential is not None:
            raise ValueError('Put the inverse AO XC teacher on OrbitalReference; Sample grid labels are ambiguous here')
        if config.inverse_density_weight and datum.target_grid_density is not None:
            raise ValueError('Predicted density uses the OrbitalReference target; Sample grid labels are not consumed')
        if datum.target_e0_total_h is not None:
            ref = replace(ref, target=replace(ref.target, energy=datum.target_e0_total_h))
        state = datum.molecule
        energy = iks.energy_function(state, functional, layout=ref.target.layout,
                                     nuclear_repulsion=datum.inverse_nuclear_repulsion)
        xc = (iks.energy_function(state, functional, layout=ref.target.layout, component='xc')
              if config.xc_potential_mse_weight else None)
        grid = getattr(state, 'grid_ao', None)
        if grid is None and config.inverse_density_weight:
            grid = (state.ao, state.grid.weights)
        value, score = iks.loss(ref, energy, params, weights=weights, grid=grid, xc_energy_fn=xc)
        losses.append(value)
        row = {key: jnp.asarray(jnp.nan, value.dtype) for key in _MOLECULAR_METRIC_KEYS}
        error = score.energy_error
        row.update(predicted_e0_total_h=score.total_energy,
                   e0_total_mse=jnp.nan if error is None else error**2,
                   e0_total_mae=jnp.nan if error is None else jnp.abs(error),
                   grid_density_mse=jnp.asarray(jnp.nan, value.dtype),
                   inverse_valid=score.valid)
        for key, metric in (('stationarity_norm2', score.rotation_norm2),
                            ('predicted_density_norm2', score.predicted_density_norm2),
                            ('reference_density_norm2', score.reference_density_norm2),
                            ('operator_mse', score.operator_mse),
                            ('xc_potential_mse', score.xc_potential_mse)):
            row[key] = jnp.asarray(jnp.nan, value.dtype) if metric is None else metric
        rows.append(row)
    metrics = {key: jnp.stack([row[key] for row in rows]) for key in rows[0]}
    metrics.update({key: jnp.array([], dtype=losses[0].dtype) for key in _SCF_METRIC_KEYS})
    metrics['response_converged'] = jnp.array([], dtype=losses[0].dtype)
    sample_weights = jnp.asarray([datum.weight for datum in dataset])
    value = jnp.sum(sample_weights*jnp.stack(losses))/jnp.sum(sample_weights)
    metrics['total_loss'] = value
    return value, metrics
