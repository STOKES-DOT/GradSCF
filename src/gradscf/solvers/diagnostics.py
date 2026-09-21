"""Common true-residual checks and invalid-derivative policy."""
import jax
import jax.numpy as jnp

SUCCESS = 0
NOT_CONVERGED = 1


def linear_residual(operator, solution, rhs, *, rtol, atol=0., converged=True):
    applied = operator(solution)
    norm = jnp.linalg.norm(applied - rhs)
    rhs_norm = jnp.linalg.norm(rhs)
    roundoff = 32 * jnp.finfo(rhs.dtype).eps * (rhs_norm + jnp.linalg.norm(applied))
    valid = (jnp.asarray(converged) & jnp.all(jnp.isfinite(solution))
             & jnp.isfinite(norm) & (norm <= atol + rtol * rhs_norm + roundoff))
    return norm, valid


@jax.custom_jvp
def require_converged_derivative(values, converged):
    """Retain primal diagnostic values; invalid tangents/cotangents become NaN."""
    return values


@require_converged_derivative.defjvp
def _require_converged_jvp(primals, tangents):
    values, converged = primals
    tangent, _ = tangents
    factor = jnp.where(converged, jnp.ones_like(values), jnp.full_like(values, jnp.nan))
    return values, tangent * factor
