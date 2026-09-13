"""Standalone initial density inputs; hcore orbitals are formed by SCF."""
from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

@dataclass(frozen=True)
class RestrictedInitGuess:
    density: Array | None = None

@dataclass(frozen=True)
class UnrestrictedInitGuess:
    density_alpha: Array | None = None
    density_beta: Array | None = None


def _hcore_requested(value):
    if value is None:
        return True
    if isinstance(value, str):
        if value.lower() in {"hcore", "1e"}:
            return True
        raise ValueError(f"Unsupported initial guess {value!r}; use 'hcore'/'1e' or supply a density matrix")
    return False


def _density(value, dtype):
    dm = jnp.asarray(value, dtype=dtype)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError("Initial density must be a square matrix")
    return .5*(dm+dm.T.conj())


def restricted_initial_guess(*, init_guess: Any, dtype: Any):
    if _hcore_requested(init_guess):
        return RestrictedInitGuess()
    return RestrictedInitGuess(_density(init_guess, dtype))


def unrestricted_initial_guess(*, init_guess: Any, dtype: Any):
    if _hcore_requested(init_guess):
        return UnrestrictedInitGuess()
    dm = jnp.asarray(init_guess, dtype=dtype)
    if dm.ndim != 3 or dm.shape[0] != 2:
        raise ValueError("Unrestricted initial density must have shape (2, nao, nao)")
    return UnrestrictedInitGuess(_density(dm[0], dtype), _density(dm[1], dtype))

def orbital_rotation_guesses(mo_coeff, *, amplitudes=(0., .1), seed=20260913):
    """Return reproducible, metric-preserving orbital guesses for explicit restarts.

    This host-side helper rotates a complete S-orthonormal orbital matrix;
    refilling the requested occupations preserves electron counts/idempotency.
    It does not select a root or assert that a converged state is the ground
    state. The caller must inspect convergence and compare candidate energies.
    """
    coeff = np.asarray(mo_coeff)
    scales = tuple(float(a) for a in amplitudes)
    if coeff.ndim != 2 or coeff.shape[0] != coeff.shape[1] or not np.all(np.isfinite(coeff)):
        raise ValueError("Expected a finite, complete square orbital matrix")
    if not scales or any(not np.isfinite(a) or a < 0 for a in scales):
        raise ValueError("Rotation amplitudes must be finite and nonnegative")
    random = np.random.default_rng(seed).normal(size=coeff.shape)
    generator = random-random.T
    # exp(A) from the Hermitian eigendecomposition of i*A, avoiding another
    # runtime dependency and retaining an exactly orthogonal rotation in theory.
    eigenvalues, vectors = np.linalg.eigh(1j*generator)
    guesses = []
    for scale in scales:
        rotation = ((vectors*np.exp(-1j*scale*eigenvalues))@vectors.conj().T).real
        guesses.append(coeff.copy() if scale == 0 else coeff@rotation)
    return tuple(guesses)


__all__ = ["RestrictedInitGuess", "UnrestrictedInitGuess", "restricted_initial_guess", "unrestricted_initial_guess", "orbital_rotation_guesses"]
