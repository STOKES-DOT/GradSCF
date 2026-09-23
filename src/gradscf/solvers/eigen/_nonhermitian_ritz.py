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
    # A real repeated eigenvalue can be represented by tiny-imaginary conjugate
    # vectors in either independent eigensolve. Taking .real loses half the span.
    roundoff = (
        128
        * jnp.finfo(matrix.dtype).eps
        * jnp.maximum(1.0, jnp.max(jnp.sum(jnp.abs(matrix), axis=1)))
    )
    real = jnp.abs(values.imag) <= roundoff
    separated = (
        ~real[1:]
        | ~real[:-1]
        | (jnp.abs(values[1:].real - values[:-1].real) > roundoff)
    )
    labels = jnp.cumsum(jnp.concatenate([jnp.ones(1, bool), separated]))

    def select(index):
        repair = real[index] & (jnp.sum(labels == labels[index]) > 1)
        return jax.lax.cond(
            repair,
            lambda _: _real_cluster_pair(matrix, values, labels, index),
            lambda _: (right[:, index], left[:, indices[index]]),
            None,
        )

    selected_right, selected_left = jax.lax.map(select, jnp.arange(count))
    return values, selected_right.T, selected_left.T


def _real_cluster_pair(matrix, values, labels, index):
    """Full real cluster dual before selecting one column, including cut clusters.

    U/V from a real SVD span the left/right approximate null spaces. Padding the
    cluster overlap with identity leaves excluded eigenspaces out of the solve;
    in particular, an unrelated Jordan block cannot poison an isolated state.
    Actual eigen-equation residuals must still be checked by the caller.
    """
    n = matrix.shape[0]
    members = labels == labels[index]
    multiplicity = jnp.sum(members)
    center = jnp.sum(jnp.where(members, values.real, 0.0)) / multiplicity
    u, _, vh = jnp.linalg.svd(
        matrix - center * jnp.eye(n, dtype=matrix.dtype), full_matrices=False
    )
    v = vh.T
    active = jnp.arange(n) >= n - multiplicity
    overlap = jnp.where(
        active[:, None] & active[None, :], u.T @ v, jnp.eye(n, dtype=matrix.dtype)
    )
    singular = jnp.linalg.svd(overlap, compute_uv=False)
    valid = jnp.all(jnp.isfinite(singular)) & (
        singular[-1] > 64 * jnp.finfo(matrix.dtype).eps * jnp.maximum(1.0, singular[0])
    )
    rank = jnp.sum(members & (jnp.arange(n) < index))
    column = n - 1 - rank
    dual = jnp.linalg.solve(
        jnp.where(valid, overlap.T, jnp.eye(n, dtype=matrix.dtype)),
        jax.nn.one_hot(column, n, dtype=matrix.dtype),
    )
    return v[:, column].astype(values.dtype), (u @ dual).astype(values.dtype)
