"""Array-valued convergence policy shared by the SCF iteration families."""
from __future__ import annotations

import jax.numpy as jnp


def convergence_reached(
    energy_change, density_rms, orbital_gradient, *, conv_tol,
    conv_tol_density, conv_tol_grad, energy_only=False, has_prior_cycle=True,
):
    """Require finite metrics and all requested thresholds.

    Energy changes are in Hartree; density RMS is in the solver's AO density
    representation. Each method supplies its physical orbital-gradient norm,
    evaluated using the raw Fock at the candidate density. An explicit energy
    mode skips density/gradient thresholds, never their finiteness checks.
    A zero energy tolerance deliberately disables early convergence.
    """
    de, dd, grad = map(jnp.asarray, (energy_change, density_rms, orbital_gradient))
    finite = jnp.isfinite(de) & jnp.isfinite(dd) & jnp.isfinite(grad)
    stationary = (dd < conv_tol_density) & (grad < conv_tol_grad)
    return (finite & jnp.asarray(has_prior_cycle) & (conv_tol > 0)
            & (jnp.abs(de) < conv_tol)
            & (jnp.asarray(energy_only) | stationary))
