"""Real closed-shell EOM-CCSD: singlet EE and doublet IP/EA sectors."""

from .types import EOMConfig, EOMResult, EOMPrecisionDiagnostics
from .amplitudes import EOMAmplitudeSpace
from .operators import build_eom_operator, run_eom
from .api import EOMEE, EOMIP, EOMEA

__all__ = [
    "EOMPrecisionDiagnostics",
    "EOMConfig",
    "EOMResult",
    "EOMAmplitudeSpace",
    "build_eom_operator",
    "run_eom",
    "EOMEE",
    "EOMIP",
    "EOMEA",
]
