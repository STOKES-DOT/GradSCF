"""Shared matrix-free numerical solvers; no electronic-structure dependencies."""
from .operators import LinearOperator
from .types import EigenResponseConfig, EigenSolverConfig, LinearSolverConfig, EigenResult, LinearResult
from .eigen import solve_hermitian
from .linear import solve_linear, solve_tensor_sum
from .types import TensorSumResult, RPAResult
from .eigen.stable_rpa import solve_stable_rpa
from .eigen.structured_rpa import solve_rpa

__all__ = ["EigenResponseConfig", "LinearOperator", "EigenSolverConfig", "LinearSolverConfig",
           "EigenResult", "LinearResult", "solve_hermitian",
           "solve_linear", "solve_tensor_sum", "TensorSumResult",
           "solve_stable_rpa", "solve_rpa", "RPAResult"]
