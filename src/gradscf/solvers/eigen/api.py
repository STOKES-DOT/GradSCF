"""Shared checked eigenvalue and eigenvector response API."""
import jax
import jax.numpy as jnp

from ..operators import as_operator, validate_real_square
from ..diagnostics import require_converged_derivative
from ..types import EigenSolverConfig, EigenResult
from .davidson import _davidson_lowest_symmetric
from .dense import dense_vectors
from .response import attach_eigenvector_response


def solve_hermitian(matrix_or_operator, *, config=None):
    """Solve a real symmetric problem with isolated-root first-order AD.

    Matrix-free callers assert symmetry. Dense inputs are checked, never
    silently symmetrized. Failed primal roots keep their Ritz values while
    their derivatives are invalid. No automatic positive-eigenvalue filtering.
    """
    config = EigenSolverConfig() if config is None else config
    op = as_operator(matrix_or_operator)
    validate_real_square(op)
    dim = op.shape[0]
    if config.nroots > dim:
        raise ValueError("nroots exceeds operator dimension")
    if config.method == "dense":
        vectors = dense_vectors(op, nroots=config.nroots, max_dense=config.max_dense)
    else:
        if op.diagonal is None:
            raise ValueError("Davidson requires an operator diagonal approximation")
        _, vectors, _ = _davidson_lowest_symmetric(
            lambda x: jax.lax.stop_gradient(op.apply(x)), nroots=config.nroots,
            size=dim, diag=op.diagonal, tol=config.atol, max_iter=config.maxiter,
            max_subspace=config.max_subspace, positive_eig_threshold=None)
    vectors = jax.lax.stop_gradient(vectors)
    applied = op.apply(vectors)
    values = jnp.sum(vectors * applied, axis=0)
    residuals = jnp.linalg.norm(applied - vectors * values[None, :], axis=0)
    valid = jnp.isfinite(values) & (residuals <= config.atol)
    if not hasattr(matrix_or_operator, 'matvec'):
        matrix = jnp.asarray(matrix_or_operator)
        symmetry_tol = 32*jnp.finfo(matrix.dtype).eps*jnp.maximum(1.,jnp.linalg.norm(matrix))
        valid = valid & (jnp.linalg.norm(matrix-matrix.T) <= symmetry_tol)
    if config.gradient_mode == "implicit_eigenvector":
        def response_apply(x):
            # Guard the operator's response before GMRES transposition. NaN
            # cotangents injected only at returned vectors can be swallowed by
            # iterative stopping logic when propagated through the linear solve.
            return require_converged_derivative(op.apply(x), jnp.all(valid))
        vectors = attach_eigenvector_response(response_apply, values, vectors,
            tol=config.adjoint_tol, max_iter=config.adjoint_maxiter)
    values = require_converged_derivative(values, valid)
    return EigenResult(values, vectors, residuals, valid, jnp.where(valid, 0, 1))
