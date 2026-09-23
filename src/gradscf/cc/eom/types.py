"""Static closed-shell EOM-CCSD options and JAX-compatible results."""

from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple
from jaxtyping import Array


@dataclass(frozen=True)
class EOMConfig:
    sector: str = "ee"
    nroots: int = 1
    solver: str = "dense"
    conv_tol: float = 1e-9
    gap_tol: float = 1e-8
    imaginary_tol: float = 1e-9
    max_condition: float = 1e8
    max_dense: int = 256
    block_size: int = 16
    max_intermediate_elements: int = 20_000_000
    max_cycle: int = 100
    max_space: int = 40
    guard_roots: int = 1
    seed: int = 0
    preconditioner_floor: float = 1e-8

    def __post_init__(self):
        if self.sector not in {"ee", "ip", "ea"}:
            raise ValueError("EOM sector must be ee, ip or ea")
        if self.solver not in {"dense", "davidson"}:
            raise ValueError("EOM solver must be dense or davidson")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        for name in (
            "nroots",
            "max_dense",
            "block_size",
            "max_intermediate_elements",
            "max_cycle",
            "max_space",
            "guard_roots",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if any(
            not isfinite(x) or x <= 0
            for x in (
                self.preconditioner_floor,
                self.conv_tol,
                self.gap_tol,
                self.imaginary_tol,
                self.max_condition,
            )
        ):
            raise ValueError("EOM tolerances must be finite and positive")


class EOMResult(NamedTuple):
    energies: Array
    right_vectors: Array
    left_vectors: Array
    residual_norms: Array
    left_residual_norms: Array
    converged: Array
    response_valid: Array
    condition_numbers: Array
    raw_eigenvalues: Array
    ground_valid: Array
    biorthogonality_error: Array
    spectrum_complete: Array
    iterations: Array
    subspace_dimension: Array
    spectral_gaps: Array
    guard_residual_norms: Array
    restarts: Array
