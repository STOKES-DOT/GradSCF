"""Compatibility exports for shared fixed-point and linear response."""
from ..solvers.nonlinear.fixed_point import ImplicitFixedPointConfig, implicit_fixed_point_solution
from ..solvers.linear import solve_implicit_linear_system

__all__ = ["ImplicitFixedPointConfig", "implicit_fixed_point_solution", "solve_implicit_linear_system"]
