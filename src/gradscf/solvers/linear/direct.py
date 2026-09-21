"""Bounded dense reference solves."""
import jax
import jax.numpy as jnp


def direct_solve(matvec, rhs, *, max_dense):
    if rhs.size > max_dense:
        raise ValueError("Direct solve exceeds max_dense; use GMRES")
    matrix = jax.vmap(matvec, in_axes=1, out_axes=1)(jnp.eye(rhs.size, dtype=rhs.dtype))
    return jnp.linalg.solve(matrix, rhs)
