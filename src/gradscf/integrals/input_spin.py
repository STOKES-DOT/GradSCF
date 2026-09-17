"""Electron-count validation shared by unrestricted input backends."""

from __future__ import annotations

def _unrestricted_spin_electron_counts(
    total_electrons: int,
    spin: int,
) -> tuple[int, int]:
    total_electrons = int(total_electrons)
    spin = int(spin)
    if total_electrons <= 0:
        raise ValueError("Unrestricted references require a positive electron count.")
    if abs(spin) > total_electrons:
        raise ValueError(
            f"Spin={spin} is incompatible with total_electrons={total_electrons}."
        )
    if (total_electrons + spin) % 2 != 0:
        raise ValueError(
            f"Spin={spin} is incompatible with total_electrons={total_electrons}; "
            "N + spin must be even."
        )
    nalpha = (total_electrons + spin) // 2
    nbeta = total_electrons - nalpha
    if nalpha < 0 or nbeta < 0:
        raise ValueError(
            f"Failed to resolve spin electron counts for total_electrons={total_electrons}, spin={spin}."
        )
    return int(nalpha), int(nbeta)


