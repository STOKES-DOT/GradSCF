"""Public JAX GMRES with scaled right-hand sides and numerical solve margin."""
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres


def gmres_solve(matvec, rhs, *, rtol, atol, maxiter, restart, preconditioner=None):
    # Keep scaling in the opaque numerical solve, not the differentiated map.
    scale = jnp.max(jnp.abs(rhs))
    def nonzero(_):
        scaled_rhs = rhs / scale
        # Leave room for rounding in response operators and scaling recovery.
        # The caller's true-residual acceptance tolerance remains unchanged.
        solve_rtol, solve_atol = .1 * rtol, .1 * atol / scale
        unit, _ = gmres(matvec, scaled_rhs, tol=solve_rtol, atol=solve_atol,
                         restart=restart, maxiter=maxiter, M=preconditioner,
                         solve_method="incremental")
        norm = jnp.linalg.norm(matvec(unit)-scaled_rhs)
        valid = jnp.isfinite(norm) & (norm <= atol/scale + rtol*jnp.linalg.norm(scaled_rhs))

        def retry(_):
            # Retry the independent batched path when the true residual fails.
            candidate, _ = gmres(matvec, scaled_rhs, tol=solve_rtol, atol=solve_atol,
                restart=restart, maxiter=maxiter, M=preconditioner, solve_method="batched")
            candidate_norm = jnp.linalg.norm(matvec(candidate)-scaled_rhs)
            use_candidate = jnp.isfinite(candidate_norm) & ((candidate_norm < norm) | ~jnp.isfinite(norm))
            return jnp.where(use_candidate, candidate, unit)

        # This predicate stays inside checked_linear_solve's opaque numerical
        # solve. Its differentiated operator remains linear, including VJPs.
        unit = jax.lax.cond(valid, lambda _: unit, retry, None)
        return unit * scale
    return jax.lax.cond(scale > 0, nonzero, lambda _: jnp.zeros_like(rhs), None)
