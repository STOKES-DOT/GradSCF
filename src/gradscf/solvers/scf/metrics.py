"""Physical matrix residuals and array-valued SCF convergence criteria."""
import jax.numpy as jnp


def orbital_gradient_norm(fock, mo_coeff, occupations):
    """Occupation-weighted MO commutator, including fractional and spinor states.

    The norm sums over leading spin blocks. It equals 2||F_vo|| for restricted
    integer occupations, and the joint ||F_vo|| norm for single occupations.
    Unequal fractional occupations retain their orbital-rotation response.
    """
    coeff = jnp.asarray(mo_coeff)
    fmo = coeff.conj().swapaxes(-1, -2) @ (jnp.asarray(fock) @ coeff)
    occ = jnp.asarray(occupations)
    difference = occ[..., None, :]-occ[..., :, None]
    return jnp.linalg.norm(fmo*difference)/jnp.sqrt(jnp.asarray(2., dtype=fmo.real.dtype))


def convergence_reached(
    energy_change, density_rms, orbital_gradient, *, conv_tol,
    conv_tol_density, conv_tol_grad, energy_only=False, has_prior_cycle=True,
):
    """Require finite metrics and the requested energy/density/gradient gates.

    Energy is in Hartree; the density metric follows the caller's AO convention.
    Energy-only mode skips the residual thresholds but never finiteness checks.
    A zero energy tolerance deliberately disables early convergence.
    """
    de, dd, grad = map(jnp.asarray, (energy_change, density_rms, orbital_gradient))
    finite = jnp.isfinite(de) & jnp.isfinite(dd) & jnp.isfinite(grad)
    stationary = (dd < conv_tol_density) & (grad < conv_tol_grad)
    return (finite & jnp.asarray(has_prior_cycle) & (conv_tol > 0)
            & (jnp.abs(de) < conv_tol)
            & (jnp.asarray(energy_only) | stationary))
