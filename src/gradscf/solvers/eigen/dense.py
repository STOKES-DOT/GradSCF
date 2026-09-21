"""Small dense eigenproblem reference path, without unrolling eigenderivatives."""
import jax
import jax.numpy as jnp


def dense_vectors(operator, *, nroots, max_dense):
    n = operator.shape[0]
    if n > max_dense:
        raise ValueError("Dense eigensolve exceeds max_dense; use Davidson")
    matrix = jax.lax.stop_gradient(operator.apply(jnp.eye(n, dtype=operator.dtype)))
    return jnp.linalg.eigh(matrix)[1][:, :nroots]
