"""Regularized full-spectrum response and real SPD inverse-square-root response.

The gap broadening preserves the existing orbital derivative convention; it is
not the isolated-root Davidson response policy. Matrix-function response remains
well defined at repeated positive overlap eigenvalues.
"""
from functools import partial
import jax
import jax.numpy as jnp
from jaxtyping import Array

_EIGH_DEGENERACY_TOL = 1e-6
_EIGH_BROADENING = 1e-10


@jax.custom_jvp
def _safe_symmetric_eigh(matrix: Array) -> tuple[Array, Array]:
    values, vectors = jnp.linalg.eigh(matrix)
    return values, vectors


@_safe_symmetric_eigh.defjvp
def _safe_symmetric_eigh_jvp(primals, tangents):
    (matrix,), (tangent,) = primals, tangents
    values, vectors = _safe_symmetric_eigh(matrix)
    value_diff = values[None, :] - values[:, None]
    nondegenerate = jnp.abs(value_diff) >= _EIGH_DEGENERACY_TOL
    # Avoid singular arithmetic even in an inactive branch of the JVP. This
    # matters when the rule itself is differentiated for force supervision.
    regular_gap = 1.0 / jnp.where(nondegenerate, value_diff, 1.0)
    broadened_gap = value_diff / (value_diff * value_diff + _EIGH_BROADENING)
    response = jnp.where(nondegenerate, regular_gap, broadened_gap)
    response = response.at[jnp.diag_indices_from(response)].set(0.0)
    inner = vectors.T @ (.5*(tangent+tangent.T)) @ vectors
    return (values, vectors), (jnp.diag(inner), vectors @ (response*inner))


def _spectral_orthogonalizer(overlap: Array, eps: float) -> Array:
    eigvals, eigvecs = _safe_symmetric_eigh(overlap)
    clipped = jnp.maximum(eigvals, eps)
    return eigvecs @ jnp.diag(clipped ** -0.5) @ eigvecs.T


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _orthogonalizer(overlap: Array, eps: float) -> Array:
    return _spectral_orthogonalizer(overlap, eps)


@_orthogonalizer.defjvp
def _orthogonalizer_jvp(eps, primals, tangents):
    (overlap,), (tangent,) = primals, tangents
    inverse_root = _orthogonalizer(overlap, eps)

    def positive_definite_response(_):
        # Differentiate the matrix function, not its individual eigenvectors:
        # sqrt(S) dX + dX sqrt(S) = -X dS X, X = S**(-1/2).
        # The Sylvester operator stays nonsingular at repeated eigenvalues.
        root = overlap @ inverse_root
        root = .5 * (root + root.T)
        rhs = -inverse_root @ (.5 * (tangent + tangent.T)) @ inverse_root

        def solve(_, value):
            values, vectors = jnp.linalg.eigh(root)
            inner = vectors.T @ value @ vectors
            return vectors @ (inner / (values[:, None] + values[None, :])) @ vectors.T

        return jax.lax.custom_linear_solve(
            lambda value: root @ value + value @ root,
            rhs, solve=solve, symmetric=True,
        )

    # Preserve the existing regularized response when eigenvalue clipping is
    # active. Exact smooth matrix-function derivatives apply above the cutoff.
    response = jax.lax.cond(
        jnp.min(jnp.linalg.eigvalsh(overlap)) > eps,
        positive_definite_response,
        lambda _: jax.jvp(lambda s: _spectral_orthogonalizer(s, eps),
                          (overlap,), (tangent,))[1],
        operand=None,
    )
    return inverse_root, response


regularized_eigh = _safe_symmetric_eigh
inverse_sqrt = _orthogonalizer
