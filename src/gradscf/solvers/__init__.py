"""Shared matrix-free numerical solvers; no electronic-structure dependencies."""
from .operators import LinearOperator
from .types import EigenSolverConfig, LinearSolverConfig, EigenResult, LinearResult, SpectralProjectorResult
from .eigen import solve_hermitian, solve_spectral_projector
from .linear import solve_linear, solve_tensor_sum
from .types import TensorSumResult, RPAResult
from .eigen.stable_rpa import solve_stable_rpa

__all__ = ["LinearOperator", "EigenSolverConfig", "LinearSolverConfig",
           "EigenResult", "LinearResult", "SpectralProjectorResult", "solve_hermitian",
           "solve_spectral_projector", "solve_linear", "solve_tensor_sum", "TensorSumResult",
           "solve_stable_rpa", "RPAResult"]
