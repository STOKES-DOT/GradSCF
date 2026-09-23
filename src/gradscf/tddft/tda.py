from __future__ import annotations

from ..solvers import (
    LinearOperator,
    EigenSolverConfig,
    EigenResponseConfig,
    LinearSolverConfig,
    solve_hermitian,
)

from collections.abc import Callable

import jax.numpy as jnp

from .defaults import PYSCF_TD_DAVIDSON_MAX_CYCLE
from .defaults import PYSCF_TD_DAVIDSON_TOL
from .defaults import PYSCF_TD_POSITIVE_EIG_THRESHOLD
from ..solvers.eigen.response import (
    EigenGradientMode,
)
from .types import TDAResult


def _finalize_tda_result(
    eigvals,
    eigvecs,
    *,
    nroots: int,
    excitation_threshold: float,
    nocc: int,
    nvir: int,
    converged=True,
) -> TDAResult:
    valid = eigvals > excitation_threshold
    order = jnp.argsort(jnp.where(valid, eigvals, jnp.inf))
    keep = order[:nroots]
    mask = valid[keep]
    energies = jnp.where(mask, eigvals[keep], 0.0)
    amplitudes = jnp.sqrt(0.5) * eigvecs[:, keep].T.reshape(-1, nocc, nvir)
    amplitudes = amplitudes * mask[:, None, None]
    return TDAResult(
        excitation_energies=energies,
        amplitudes=amplitudes,
        converged=converged,
    )


def solve_tda_from_operator(
    delta_eps,
    vind_rows: Callable,
    diagonal,
    *,
    nstates: int | None = None,
    excitation_threshold: float = PYSCF_TD_POSITIVE_EIG_THRESHOLD,
    davidson_tol: float = PYSCF_TD_DAVIDSON_TOL,
    davidson_max_iter: int = PYSCF_TD_DAVIDSON_MAX_CYCLE,
    davidson_max_subspace: int | None = None,
    davidson_initial_guess_count: int | None = None,
    davidson_max_trial_vectors: int | None = None,
    tda_gradient_mode: EigenGradientMode = "eigenvalue_only",
    eigenvector_adjoint_tol: float = 1e-6,
    eigenvector_adjoint_max_iter: int = 64,
) -> TDAResult:
    nocc, nvir = delta_eps.shape
    dim = int(nocc * nvir)
    nroots = dim if nstates is None else min(int(nstates), dim)
    if tda_gradient_mode not in {"eigenvalue_only", "implicit_eigenvector"}:
        raise ValueError(f"Unsupported TDA gradient mode {tda_gradient_mode!r}.")

    def matrix_action(vectors):
        return vind_rows(jnp.asarray(vectors).T).T

    diag = jnp.asarray(diagonal).reshape(dim)
    operator = LinearOperator(
        (dim, dim),
        diag.dtype,
        lambda v: matrix_action(v[:, None])[:, 0],
        diagonal=diag,
        matmat=matrix_action,
    )
    solved = solve_hermitian(
        operator,
        config=EigenSolverConfig(
            nroots=nroots,
            atol=davidson_tol,
            maxiter=davidson_max_iter,
            max_subspace=davidson_max_subspace,
            value_min=excitation_threshold,
            initial_guess_count=davidson_initial_guess_count,
            max_trial_vectors=davidson_max_trial_vectors,
        ),
        response=EigenResponseConfig(
            target=(
                "eigenpairs"
                if tda_gradient_mode == "implicit_eigenvector"
                else "eigenvalues"
            ),
            linear_config=LinearSolverConfig(
                rtol=eigenvector_adjoint_tol, maxiter=eigenvector_adjoint_max_iter
            ),
        ),
    )
    eigvals, eigvecs = solved.values, solved.vectors
    converged = jnp.all(solved.converged) & jnp.all(eigvals > excitation_threshold)
    return _finalize_tda_result(
        eigvals,
        eigvecs,
        nroots=nroots,
        excitation_threshold=excitation_threshold,
        nocc=nocc,
        nvir=nvir,
        converged=converged,
    )
