"""Public JAX GMRES with scale-invariant right-hand-side handling."""
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres


def gmres_solve(matvec, rhs, *, rtol, atol, maxiter, restart, preconditioner=None):
    # Keep scaling in the opaque numerical solve, not the differentiated map.
    scale = jnp.max(jnp.abs(rhs))
    def nonzero(_):
        unit, _ = gmres(matvec, rhs / scale, tol=rtol, atol=atol / scale,
                         restart=restart, maxiter=maxiter, M=preconditioner,
                         solve_method="incremental")
        return unit * scale
    return jax.lax.cond(scale > 0, nonzero, lambda _: jnp.zeros_like(rhs), None)
