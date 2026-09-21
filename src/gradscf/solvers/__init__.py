"""Shared matrix-free numerical solvers; no electronic-structure dependencies."""
from .operators import LinearOperator
from .types import EigenSolverConfig, LinearSolverConfig, EigenResult, LinearResult
from .eigen import solve_hermitian
from .linear import solve_linear

__all__ = ["LinearOperator", "EigenSolverConfig", "LinearSolverConfig",
           "EigenResult", "LinearResult", "solve_hermitian", "solve_linear"]
