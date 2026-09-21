"""Compatibility aliases for shared real Hermitian eigenvector response."""
from ..solvers.eigen.response import (
    EigenGradientMode as TDAGradientMode,
    attach_eigenvector_response as _attach_implicit_eigenvector_differential,
    implicit_differential_davidson_lowest_symmetric_with_eigenvectors,
)
