"""Repeated shifted solves with one transpose-symmetric factorization."""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class ShiftedFactorization(NamedTuple):
    matrix: object
    values: object
    vectors: object
    inverse_vectors: object = None
    spectral_valid: object = True


def factor_shifted(matrix):
    """Prepare M for (M-shift I) solves; eigenvectors are primal data only.

    The solve differentiates the original matrix equation, including at
    repeated eigenvalues. No eigenvalue gap division or clipping is used.
    """
    matrix = jnp.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Expected a square matrix')
    if not jnp.issubdtype(matrix.dtype, jnp.inexact):
        raise ValueError('Expected a real or complex floating-point matrix')
    tolerance = 64*jnp.finfo(matrix.dtype).eps*jnp.maximum(1., jnp.linalg.norm(matrix))
    valid = jnp.all(jnp.isfinite(matrix)) & (jnp.linalg.norm(matrix-matrix.T) <= tolerance)
    matrix = jnp.where(valid, (matrix+matrix.T)*.5, jnp.nan)
    primal = jax.lax.stop_gradient(matrix)
    if not jnp.iscomplexobj(matrix):
        values, vectors = jnp.linalg.eigh(primal)
        return ShiftedFactorization(matrix, values, vectors)
    values, vectors = jnp.linalg.eig(primal)
    inverse = jnp.linalg.inv(vectors)
    # Complex symmetric matrices need not be normal or even diagonalizable.
    # Cache an eigenbasis only when its conditioning and reconstruction permit
    # accurate solves; defective/ill-conditioned bases use the dense equation.
    eps = jnp.finfo(matrix.dtype).eps
    condition = jnp.linalg.norm(vectors) * jnp.linalg.norm(inverse)
    error = jnp.linalg.norm((vectors * values) @ inverse - primal)
    spectral_valid = (jnp.isfinite(condition) & (condition < eps**(-.25))
                      & (error <= 128*eps*jnp.maximum(1., jnp.linalg.norm(primal))))
    return ShiftedFactorization(matrix, values, vectors, inverse, spectral_valid)


def solve_shifted(state, rhs, shift):
    """Solve a real/complex scalar shift for a vector or matrix RHS.

    Factors stay local to their matrix. custom_linear_solve supplies JVP/VJP
    from A dx = db - dA x, without differentiating the spectral basis.
    Complex shifts yield a complex symmetric (not Hermitian) operator;
    symmetric=True refers to its non-conjugating transpose, as JAX requires.
    """
    rhs, shift = jnp.asarray(rhs), jnp.asarray(shift)
    if rhs.ndim not in (1, 2) or rhs.shape[0] != state.matrix.shape[0] or shift.ndim:
        raise ValueError('Expected RHS (n,) or (n,k) and a scalar shift')
    dtype = jnp.result_type(state.matrix, rhs, shift)
    rhs = rhs.astype(dtype)
    def action(x):
        return state.matrix @ x - shift*x
    def solve(_, b):
        def spectral(value):
            denominator = state.values-shift
            if value.ndim == 2:
                denominator = denominator[:, None]
            inverse = state.vectors.T if state.inverse_vectors is None else state.inverse_vectors
            return state.vectors @ ((inverse @ value)/denominator)
        if state.inverse_vectors is None:
            return spectral(b)
        return jax.lax.cond(state.spectral_valid, spectral,
            lambda value: jnp.linalg.solve(state.matrix-shift*jnp.eye(state.matrix.shape[0]), value), b)
    return jax.lax.custom_linear_solve(action, rhs, solve=solve, symmetric=True)
