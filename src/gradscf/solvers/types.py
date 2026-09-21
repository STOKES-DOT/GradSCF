"""Static solver configuration and JIT-compatible results."""
from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple

from jaxtyping import Array


@dataclass(frozen=True)
class LinearSolverConfig:
    method: str = "gmres"
    rtol: float = 1e-9
    atol: float = 0.0
    maxiter: int = 100
    restart: int = 20
    max_dense: int = 2048

    def __post_init__(self):
        if self.method not in {"gmres", "direct"}:
            raise ValueError("linear method must be 'gmres' or 'direct'")
        if any(not isfinite(x) or x < 0 for x in (self.rtol, self.atol)):
            raise ValueError("linear tolerances must be finite and nonnegative")
        if self.rtol == self.atol == 0:
            raise ValueError("at least one tolerance must be positive")
        if min(self.maxiter, self.restart, self.max_dense) < 1:
            raise ValueError("solver limits must be positive")


@dataclass(frozen=True)
class EigenSolverConfig:
    method: str = "davidson"
    nroots: int = 1
    atol: float = 1e-9
    maxiter: int = 100
    max_subspace: int | None = None
    gradient_mode: str = "eigenvalue_only"
    adjoint_tol: float = 1e-10
    adjoint_maxiter: int = 100
    max_dense: int = 2048

    def __post_init__(self):
        if self.method not in {"davidson", "dense"}:
            raise ValueError("eigen method must be 'davidson' or 'dense'")
        if not isinstance(self.nroots, int) or self.nroots < 1:
            raise ValueError("nroots must be positive")
        if self.gradient_mode not in {"eigenvalue_only", "implicit_eigenvector"}:
            raise ValueError("Unknown eigen gradient_mode")
        if any(not isfinite(x) or x <= 0 for x in (self.atol, self.adjoint_tol)):
            raise ValueError("eigen tolerances must be finite and positive")
        if min(self.maxiter, self.adjoint_maxiter, self.max_dense) < 1:
            raise ValueError("solver limits must be positive")
        if self.max_subspace is not None and self.max_subspace < self.nroots:
            raise ValueError("max_subspace must be at least nroots")


class LinearResult(NamedTuple):
    solution: Array
    residual_norm: Array
    converged: Array
    status: Array


class EigenResult(NamedTuple):
    values: Array
    vectors: Array
    residual_norms: Array
    converged: Array
    status: Array
