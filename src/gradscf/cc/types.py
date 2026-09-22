"""Real restricted ground-state CC configuration and JAX result containers."""

from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple
from ..scf.reference import RestrictedReference as CCReference


@dataclass(frozen=True)
class CCConfig:
    method: str = "ccsd"
    max_cycle: int = 100
    conv_tol: float = 1e-10
    residual_tol: float = 1e-9
    diis_space: int = 6
    diis_start_cycle: int = 2
    damping: float = 0.0
    level_shift: float = 0.0
    adjoint_tol: float = 1e-10
    adjoint_max_cycle: int = 100
    adjoint_restart: int = 40
    denominator_tol: float = 1e-10

    def __post_init__(self):
        object.__setattr__(self, "method", self.method.lower())
        if self.method not in {"ccs", "ccd", "ccsd", "cc2", "lccd", "lccsd", "qcisd"}:
            raise ValueError(
                "Unsupported CC method; supported: CCS, CCD, CCSD, CC2, LCCD, LCCSD, QCISD"
            )
        if any(
            not isfinite(x) or x <= 0
            for x in (
                self.conv_tol,
                self.residual_tol,
                self.adjoint_tol,
                self.denominator_tol,
            )
        ):
            raise ValueError("CC tolerances must be positive")
        if (
            min(
                self.max_cycle,
                self.diis_space,
                self.diis_start_cycle,
                self.adjoint_max_cycle,
                self.adjoint_restart,
            )
            < 1
        ):
            raise ValueError("CC iteration/history limits must be positive")
        if not isfinite(self.damping) or not 0 <= self.damping < 1:
            raise ValueError("damping must be in [0,1)")
        if not isfinite(self.level_shift) or self.level_shift < 0:
            raise ValueError("level_shift must be finite and nonnegative")


class CCResult(NamedTuple):
    total_energy: object
    correlation_energy: object
    reference_energy: object
    t1: object
    t2: object
    residual_norm: object
    energy_change: object
    iterations: object
    converged: object
    min_abs_denominator: object
    method_id: object


class LambdaResult(NamedTuple):
    l1: object
    l2: object
    adjoint: object
    residual_norm: object
    converged: object


class TriplesResult(NamedTuple):
    """Noniterative correction and Hartree-valued diagnostics; no T3 iteration.

    canonical_error measures the input active Fock off-diagonal entries. A
    semicanonical calculation may be valid with nonzero canonical_error.
    In that mode singles_component includes both T1*V and F_vo*T2 terms, and
    min_abs_denominator covers the full Cartesian tensor-sum spectrum.
    """

    energy: object
    connected_component: object
    singles_component: object
    min_abs_denominator: object
    cc_residual_norm: object
    canonical_error: object
    valid: object
    variant_id: object
