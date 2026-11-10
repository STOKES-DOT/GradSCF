"""Orbital-frame whitening, separate from the symmetric matrix function."""
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from ..eigen.spectral import inverse_sqrt


@partial(jax.jit, static_argnames=("eps",))
def orbital_orthogonalizer(overlap, eps=1e-10):
    """Return a real orbital factor X with X.T @ S @ X = I above the cutoff.

    For S > eps*I, X is the inverse transpose of its Cholesky factor and need
    not be symmetric. Otherwise retain the existing clipped symmetric inverse
    root. The cutoff is an algorithmic scalar, not a differentiated parameter;
    derivatives are piecewise smooth away from its boundary. Callers requiring
    the symmetric matrix-function gauge must continue using inverse_sqrt.
    """
    overlap = jnp.asarray(overlap)
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("Orbital orthogonalization requires a square overlap matrix.")
    if jnp.iscomplexobj(overlap):
        raise ValueError("Orbital orthogonalization currently requires real spatial overlap.")
    identity = jnp.eye(overlap.shape[0], dtype=overlap.dtype)
    above_cutoff = jnp.all(jnp.isfinite(jnp.linalg.cholesky(overlap-eps*identity)))

    def cholesky_factor(matrix):
        lower = jnp.linalg.cholesky(matrix)
        return solve_triangular(lower.T, identity, lower=False)

    return jax.lax.cond(above_cutoff, cholesky_factor,
                        lambda matrix: inverse_sqrt(matrix, eps), overlap)
