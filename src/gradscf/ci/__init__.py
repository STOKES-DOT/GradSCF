"""Differentiable real restricted and unrestricted configuration interaction."""
from .types import CIConfig, CIReference, CIResult, CISResult, CISDCorrectionResult
from .space import CISpace, make_ci_space, UCISpace, make_uci_space
from .hamiltonian import hamiltonian_action, hamiltonian_matrix
from .solver import solve_ci, solve_cis, solve_ucis
from .api import CI, CIS, CISD, CISDT, CISDTQ, CIS_D, UCI, UCIS, UCISD, UCISDT, UCISDTQ
from .types import UCISResult
from ..scf.reference import UnrestrictedReference
from .corrections import cis_d_correction

__all__ = ["CIConfig", "CIReference", "CIResult", "CISResult", "CISpace",
           "make_ci_space", "hamiltonian_action", "hamiltonian_matrix", "solve_ci", "solve_cis",
           "CI", "CIS", "CISD", "CISDT", "CISDTQ", "CIS_D", "cis_d_correction",
           "CISDCorrectionResult", "UnrestrictedReference", "UCISpace", "make_uci_space",
           "UCI", "UCIS", "solve_ucis", "UCISResult", "UCISD", "UCISDT", "UCISDTQ"]
