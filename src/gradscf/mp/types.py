"""Canonical molecular MP configuration and JAX result containers."""
from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple


@dataclass(frozen=True)
class MPConfig:
    order: int = 2
    with_t2: bool = True
    denominator_tol: float = 1e-10
    canonical_tol: float = 1e-8

    def __post_init__(self):
        if isinstance(self.order, bool) or self.order not in (2, 3):
            raise ValueError("MP order must be 2 or 3")
        if any(not isfinite(x) or x <= 0
               for x in (self.denominator_tol, self.canonical_tol)):
            raise ValueError("MP tolerances must be positive and finite")


class MPResult(NamedTuple):
    """Hartree-valued MP corrections and input validity diagnostics.

    e2/e3 are individual orders; correlation_energy is their sum. t2 contains
    T2^(1), even for MP3. same_spin_energy/opposite_spin_energy decompose E2
    only. Invalid inputs produce NaN total/correlation energy. Reference values,
    diagnostics and still-valid lower-order components may remain available.
    """
    total_energy: object
    correlation_energy: object
    reference_energy: object
    e2: object
    e3: object
    same_spin_energy: object
    opposite_spin_energy: object
    t2: object
    min_abs_denominator: object
    canonical_error: object
    valid: object
