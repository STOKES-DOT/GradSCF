"""Array contracts and iteration controls for the shared SCF solver."""
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import jax


class SCFState(NamedTuple):
    density: jax.Array
    mo_coeff: jax.Array
    mo_energy: jax.Array
    energy: jax.Array
    fock: jax.Array
    aux: Any = ()


@dataclass(frozen=True)
class SCFConfig:
    max_cycle: int = 80
    conv_tol: float = 1e-10
    conv_tol_density: float = 1e-8
    conv_tol_grad: float = 1e-7
    damping: float = 0.
    diis_space: int = 8
    diis_start_cycle: int = 0
    fock_damping_cycles: int = 1
    skip_first_fock_damping: bool = False
    density_damping: Literal["none", "initial", "all"] = "none"
    energy_only: bool = False
    store_iterates: bool = False
    refresh_on_convergence: bool = False

    def __post_init__(self):
        if self.max_cycle < 1 or self.diis_space < 1 or self.diis_start_cycle < 0:
            raise ValueError("SCF requires positive iterations/history and a nonnegative DIIS start.")
        if min(self.conv_tol, self.conv_tol_density, self.conv_tol_grad) < 0:
            raise ValueError("SCF requires nonnegative convergence tolerances.")
        if self.density_damping not in {"none", "initial", "all"}:
            raise ValueError("density_damping must be none, initial, or all.")


class SCFMetrics(NamedTuple):
    energy_delta: jax.Array
    density_rms: jax.Array
    gradient: jax.Array


class SCFHistory(NamedTuple):
    density_rms: jax.Array
    energy_change: jax.Array
    gradient: jax.Array
    converged: jax.Array
    iterates: tuple


class SCFResult(NamedTuple):
    state: SCFState
    best_state: SCFState
    best_cycle: jax.Array
    converged: jax.Array
    cycles: jax.Array
    metrics: SCFMetrics
    history: SCFHistory
    # Caller seed preparation is excluded; main cycles include full-tail advances.
    # Fresh evaluations count only extra builds beyond those main advances.
    advance_evaluations: jax.Array | int = 0
    fresh_evaluations: jax.Array | int = 0
