"""Numerical left/right Ritz pairs, without an automatic differentiation rule."""

import jax
import jax.numpy as jnp


def paired_eigenvectors(matrix, count):
    values, right = jnp.linalg.eig(matrix)
    left_values, left = jnp.linalg.eig(matrix.T)
    order = jnp.lexsort((values.imag, values.real))
    values, right = values[order], right[:, order]

    def pair(i, carry):
        indices, used = carry
        distances = jnp.where(used, jnp.inf, jnp.abs(left_values - values[i]))
        index = jnp.argmin(distances).astype(jnp.int32)
        return indices.at[i].set(index), used.at[index].set(True)

    indices, _ = jax.lax.fori_loop(
        0,
        count,
        pair,
        (jnp.zeros(count, jnp.int32), jnp.zeros(matrix.shape[0], bool)),
    )
    return values, right[:, :count], left[:, indices]
