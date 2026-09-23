"""Explicit host-side restricted SCF branch selection; not an SCF AD rule."""

from dataclasses import dataclass, replace
from numbers import Integral
from typing import TYPE_CHECKING
import numpy as np
from .init_guess import orbital_rotation_guesses

if TYPE_CHECKING:
    from .facade import RKS


@dataclass(frozen=True)
class RestrictedSCFAttempt:
    amplitude: float
    energy: float
    converged: bool
    cycles: int | None
    seed: int | None = None
    stable: bool | None = None


@dataclass(frozen=True)
class RestrictedMultistartResult:
    selected: "RKS | None"
    selected_index: int | None
    attempts: tuple[RestrictedSCFAttempt, ...]
    seed: int
    seeds: tuple[int, ...] = ()

    @property
    def converged(self):
        return self.selected_index is not None


def run_restricted_multistart(
    mean_field,
    *,
    amplitudes=(0.05, 0.15, 0.4),
    seed=20260923,
    seeds=None,
    require_stable=False,
):
    """Run a fresh baseline plus explicit native orbital-rotation density starts.

    The input RKS facade and its settings are preserved. All candidates use those
    settings; tighten convergence/max_cycle on the input when needed. The selected
    facade is the lowest finite converged candidate, not a certified global or
    stable minimum. Failed convergence is retained in attempts. Empty amplitudes
    requests just a baseline; positive amplitudes are additional starts.

    Selection is a discrete host-side workflow. Differentiate a selected branch
    with the existing SCF interfaces, not this multistart operation.
    """
    from .facade import RKS

    if not isinstance(mean_field, RKS):
        raise TypeError("Restricted multistart requires a GradSCF RKS facade")
    scales = tuple(float(a) for a in amplitudes)
    if any(not np.isfinite(a) or a <= 0 for a in scales):
        raise ValueError(
            "Restart amplitudes must be finite and positive; the baseline is included"
        )
    if seeds is not None and seed != 20260923:
        raise ValueError("Specify seed or seeds, not both")
    chosen_seeds = (seed,) if seeds is None else tuple(seeds)
    if (
        not chosen_seeds
        or any(
            not isinstance(s, Integral) or isinstance(s, bool) or s < 0
            for s in chosen_seeds
        )
        or len(set(chosen_seeds)) != len(chosen_seeds)
    ):
        raise ValueError("seeds must contain distinct nonnegative integers")
    chosen_seeds = tuple(map(int, chosen_seeds))
    if not isinstance(require_stable, bool):
        raise ValueError("require_stable must be boolean")

    def summary(amplitude, candidate, trial_seed=None):
        energy = float(candidate.e_tot)
        converged = bool(candidate.converged) and bool(np.isfinite(energy))
        cycles = None if candidate.cycles is None else int(candidate.cycles)
        stable = candidate.stability().stable if require_stable and converged else None
        return RestrictedSCFAttempt(
            amplitude, energy, converged, cycles, trial_seed, stable
        )

    def eligible(attempt):
        return attempt.converged and (not require_stable or attempt.stable is True)

    # kernel clears clone caches/results before computing; the source is not run
    # or mutated, including when the caller has changed its controls since SCF.
    baseline = replace(mean_field).run()
    attempts = [summary(0.0, baseline)]
    best, best_index = (baseline, 0) if eligible(attempts[0]) else (None, None)
    if scales:
        occupations = np.asarray(baseline.mo_occ)
        if occupations.ndim != 1 or not np.all((occupations == 0) | (occupations == 2)):
            raise ValueError("Restricted multistart requires integer 0/2 occupations")
        if np.iscomplexobj(baseline.mo_coeff):
            raise NotImplementedError(
                "Restricted multistart currently requires real orbitals"
            )
        for trial_seed in chosen_seeds:
            for amplitude, coeff in zip(
                scales,
                orbital_rotation_guesses(
                    baseline.mo_coeff, amplitudes=scales, seed=trial_seed
                ),
            ):
                density = (coeff * occupations[None, :]) @ coeff.T
                trial = replace(mean_field, init_guess=density).run()
                attempt = summary(amplitude, trial, trial_seed)
                attempts.append(attempt)
                if eligible(attempt) and (
                    best is None or attempt.energy < float(best.e_tot)
                ):
                    best, best_index = trial, len(attempts) - 1
    return RestrictedMultistartResult(
        best, best_index, tuple(attempts), chosen_seeds[0], chosen_seeds
    )
