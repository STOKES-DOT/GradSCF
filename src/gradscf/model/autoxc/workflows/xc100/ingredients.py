"""Lazy fixed-target density fields for candidate-defined expressions."""
from types import SimpleNamespace

import jax.numpy as jnp


def features_for_candidate(case,names):
    """Return scalar grid leaves; no target energy or inverse potential is read."""
    names=set(names)
    if not names <= case.fields:
        raise ValueError('Unknown density fields: ' + ', '.join(sorted(names - case.fields)))
    if not hasattr(case,'_open_inputs'):
        a=case.arrays
        physical=SimpleNamespace(rho=jnp.asarray(a['rho']),grad_rho=jnp.asarray(a['grad_rho']),
                                 laplacian_rho=jnp.asarray(a['laplacian_rho']),weights=jnp.asarray(a['grid_weights']))
        case._open_inputs=case.inputs(physical,case.fields)
    return {name: case._open_inputs[name] for name in names}
