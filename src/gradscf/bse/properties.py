"""Unit-spatial-norm BSE transition moments and length-gauge strengths."""

import jax
import jax.numpy as jnp
from ..solvers.diagnostics import require_converged_derivative


def transition_dipoles(result, dipole_mo, space):
    d = jnp.asarray(dipole_mo)
    if d.shape != (3, space.nmo, space.nmo):
        raise ValueError("dipole_mo must have shape (3,nmo,nmo)")
    if jnp.iscomplexobj(d):
        raise NotImplementedError(
            "Molecular BSE transition moments currently require real inputs"
        )
    d = d[:, jnp.asarray(space.occupied)[:, None], jnp.asarray(space.virtual)[None, :]]
    mu = jnp.sqrt(2.0) * jnp.einsum(
        "xia,sia->sx", d, result.x_amplitudes + result.y_amplitudes
    )
    if not result.singlet:
        mu = jnp.zeros_like(mu)
    valid = result.converged & result.stable
    response = result.response_valid & result.amplitude_response
    if not result.amplitude_response:
        # Keep a live state dependency even though stopped X has no tangent.
        # Otherwise a grad through omega alone could masquerade as full response.
        sentinel = require_converged_derivative(result.excitation_energies, False)
        mu = mu + (sentinel - jax.lax.stop_gradient(sentinel))[:, None]
    mu = require_converged_derivative(mu, response[:, None])
    return jnp.where(valid[:, None], mu, jnp.nan)


def oscillator_strengths(result, dipole_mo, space):
    mu = transition_dipoles(result, dipole_mo, space)
    value = (2 / 3) * result.excitation_energies * jnp.sum(mu**2, axis=1)
    return require_converged_derivative(
        value, result.response_valid & result.amplitude_response
    )
