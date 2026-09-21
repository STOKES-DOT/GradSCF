"""Implicit derivatives of nondegenerate symmetric Ritz vectors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from ..linear import solve_implicit_linear_system
from ..diagnostics import require_converged_derivative
from jaxtyping import Array

from .davidson import (
    DEFAULT_EIGEN_MAXITER,
    DEFAULT_EIGEN_TOL,
    _resolve_symmetric_linear_operator,
    implicit_differential_davidson_lowest_symmetric,
)


EigenGradientMode = Literal["eigenvalue_only", "implicit_eigenvector"]


def attach_eigenvector_response(
    apply: Callable[[Array], Array],
    eigvals: Array,
    eigvecs: Array,
    *,
    tol: float,
    max_iter: int,
) -> Array:
    """Attach root-wise constrained differentials without changing primal vectors.

    The right-hand side is identically zero in the primal computation but has
    tangent ``-Q (dA) v``.  Therefore the augmented solution has zero primal
    value and the operator-variation term ``(dK) x`` also vanishes.
    """

    stopped_values = jax.lax.stop_gradient(eigvals)
    stopped_vectors = jax.lax.stop_gradient(eigvecs)
    dim, nroots = stopped_vectors.shape

    def attach_root(root: int) -> Array:
        omega = stopped_values[root]
        vector = stopped_vectors[:, root]

        def project(values: Array) -> Array:
            return values - vector * jnp.vdot(vector, values).real

        def augmented_matvec(values: Array) -> Array:
            trial = values[:dim]
            multiplier = values[dim]
            applied = apply(trial[:, None])[:, 0]
            top = applied - omega * trial + multiplier * vector
            bottom = jnp.asarray(
                [jnp.vdot(vector, trial).real],
                dtype=trial.dtype,
            )
            return jnp.concatenate([top, bottom])

        applied_vector = apply(vector[:, None])[:, 0]
        live_zero = applied_vector - jax.lax.stop_gradient(applied_vector)
        rhs = jnp.concatenate(
            [-project(live_zero), jnp.zeros((1,), dtype=vector.dtype)]
        )

        correction = solve_implicit_linear_system(
            augmented_matvec, rhs, tol=tol, max_iter=max_iter,
            restart=min(20, dim + 1),
        )
        return vector + correction[:dim]

    return jnp.swapaxes(jax.lax.map(attach_root, jnp.arange(nroots)), 0, 1)


def implicit_differential_davidson_lowest_symmetric_with_eigenvectors(
    matrix_or_matvec: Array | Callable[[Array], Array],
    *,
    nroots: int,
    size: int | None = None,
    diag: Array | None = None,
    tol: float = DEFAULT_EIGEN_TOL,
    max_iter: int = DEFAULT_EIGEN_MAXITER,
    max_subspace: int | None = None,
    collapse_subspace: int | None = None,
    initial_guess_count: int | None = None,
    max_trial_vectors: int | None = None,
    positive_eig_threshold: float | None = None,
    preconditioner_floor: float = 1e-8,
    preconditioner_level_shift: float = 0.0,
    orth_eps: float = 1e-10,
    eigenvector_adjoint_tol: float = 1e-6,
    eigenvector_adjoint_max_iter: int = 64,
) -> tuple[Array, Array, Array]:
    """Return Davidson roots with opt-in implicit Ritz-vector gradients.

    The forward Ritz pairs and eigenvalue derivative are produced by the
    existing eigenvalue-only solver.  Each vector then receives the constrained
    Xie--Liu--Wang differential through a matrix-free augmented GMRES solve.
    Differentiated roots must be converged, nondegenerate, and spectrally
    separated; this function does not regularize degeneracies.
    """

    apply, _, _, _ = _resolve_symmetric_linear_operator(
        matrix_or_matvec,
        size=size,
        diag=diag,
    )
    eigvals, eigvecs, converged = implicit_differential_davidson_lowest_symmetric(
        matrix_or_matvec,
        nroots=nroots,
        size=size,
        diag=diag,
        tol=tol,
        max_iter=max_iter,
        max_subspace=max_subspace,
        collapse_subspace=collapse_subspace,
        initial_guess_count=initial_guess_count,
        max_trial_vectors=max_trial_vectors,
        positive_eig_threshold=positive_eig_threshold,
        preconditioner_floor=preconditioner_floor,
        preconditioner_level_shift=preconditioner_level_shift,
        orth_eps=orth_eps,
    )
    def checked_apply(values):
        return require_converged_derivative(apply(values), converged)

    differentiable_vectors = attach_eigenvector_response(
        checked_apply,
        eigvals,
        eigvecs,
        tol=eigenvector_adjoint_tol,
        max_iter=eigenvector_adjoint_max_iter,
    )
    return eigvals, differentiable_vectors, converged
