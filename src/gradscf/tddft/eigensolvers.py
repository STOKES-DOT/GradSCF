"""Compatibility exports. Numerical forward/backward lives in gradscf.solvers."""
from ..solvers.eigen.davidson import (
    _davidson_search_nroots, _solver_dtype, _davidson_lowest_symmetric,
    implicit_differential_davidson_lowest_symmetric,
)
from ..solvers.eigen.rpa import (
    _davidson_lowest_tdhf, implicit_differential_davidson_lowest_tdhf,
)

# Electronic-structure defaults remain with their callers.
PYSCF_TD_DAVIDSON_TOL = 1e-5
PYSCF_TD_DAVIDSON_MAX_CYCLE = 100
FULL_TDDFT_DAVIDSON_MAX_CYCLE = 512
PYSCF_TD_POSITIVE_EIG_THRESHOLD = 1e-3
__all__ = ["implicit_differential_davidson_lowest_symmetric",
           "implicit_differential_davidson_lowest_tdhf"]
