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

    def __post_init__(self):
        if self.sector not in {"ee", "ip", "ea"}:
            raise ValueError("EOM sector must be ee, ip or ea")
        if self.solver != "dense":
            raise NotImplementedError(
                "Initial EOM-CCSD requires the bounded dense solver"
            )
        for name in ("nroots", "max_dense", "block_size", "max_intermediate_elements"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if any(
            not isfinite(x) or x <= 0
            for x in (
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
