"""Physical-layout adapters to the shared SCF numerical solver.

Fock/XC construction stays with the caller. These adapters preserve the
restricted and unrestricted density, level-shift and iterate-selection
conventions without maintaining another iteration or DIIS implementation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..solvers.scf import SCFConfig, SCFState, orbital_gradient_norm, solve_scf
from .core import _build_density_from_occ, _diagonalize_fock, _orthogonalizer


def orthonormal_diis_error(fock, density, overlap, coefficients):
    error = fock @ density @ overlap - overlap @ density @ fock
    if jnp.finfo(fock.real.dtype).bits < 64:
        return error
    return coefficients.conj().T @ error @ coefficients


def apply_level_shift(fock, overlap, density, factor):
    """Shift the virtual space; density is in the restricted 0/2 convention."""
    return fock + factor * (overlap - .5 * overlap @ density @ overlap)


def run_restricted_scf(
    *, h, s, x, energy_and_fock_builder, cfg, mo_occ_fixed, diis_basis,
    skip_first_fock_damping, density, mo_coeff, mo_occ, mo_energy, raw_fock,
    j_mat, k_mat, force_density_damping=False, density_convergence_tol=None,
    eigenvalue_jitter=0., refresh_on_convergence=False, store_iterates=True,
):
    """Adapt restricted Fock data and optional iterate histories to solve_scf."""
    use_fock_damping = jnp.finfo(h.dtype).bits >= 64 and not force_density_damping

    def evaluate(dm, coeff, energies, previous):
        old_density, old_j, old_k = ((None, None, None) if previous is None
                                    else (previous.density, previous.aux[1], previous.aux[2]))
        energy, xc, fock, coulomb, exchange = energy_and_fock_builder(
            dm, coeff, mo_occ, energies, old_density, old_j, old_k)
        return energy, fock, (xc, coulomb, exchange)

    config = SCFConfig(
        max_cycle=int(cfg.max_cycle), conv_tol=float(cfg.conv_tol),
        conv_tol_density=float(cfg.conv_tol_density if density_convergence_tol is None
                               else density_convergence_tol),
        conv_tol_grad=float(cfg.conv_tol_grad), damping=float(cfg.damping),
        diis_start_cycle=1, fock_damping_cycles=1 if use_fock_damping else 0,
        skip_first_fock_damping=bool(skip_first_fock_damping),
        density_damping=("all" if force_density_damping else
                         "none" if use_fock_damping else "initial"),
        energy_only=cfg.convergence_metric == "energy", store_iterates=store_iterates,
        refresh_on_convergence=refresh_on_convergence or cfg.jk_backend == "direct")
    zero = jnp.asarray(0., dtype=h.dtype)
    initial = SCFState(density, mo_coeff, mo_energy, zero, raw_fock, (zero, j_mat, k_mat))
    result = solve_scf(initial, evaluate=evaluate,
        diagonalize=lambda f: _diagonalize_fock(f, x, eigenvalue_jitter=eigenvalue_jitter),
        density_from_coeff=lambda c: _build_density_from_occ(c, mo_occ_fixed),
        diis_error=lambda f, state: orthonormal_diis_error(f, state.density, s, diis_basis),
        orbital_gradient=lambda state: orbital_gradient_norm(state.fock, state.mo_coeff, mo_occ_fixed),
        shift_fock=lambda f, state: apply_level_shift(f, s, state.density, cfg.level_shift),
        config=config, initial_last_fock=h)
    state = result.state
    histories = result.history.iterates if store_iterates else ((), (), ())
    return (result.converged, result.cycles, state.density, state.mo_coeff,
            state.mo_energy, state.energy, state.aux[0], state.fock,
            state.aux[1], state.aux[2], *histories, result.history.density_rms)


def run_unrestricted_scf(
    *, fock_builder, density_spin, mo_coeff_spin, mo_occ_spin, mo_energy_spin,
    overlap, max_cycle, damping, conv_tol=None, conv_tol_density,
    conv_tol_grad=1e-7, orthogonalization_eps, convergence_metric="energy_and_residual",
    level_shift=0., eigenvalue_jitter=0., iterate_selection="final", orthogonalizer=None,
    state_evaluator=None, initial_aux=(), refresh_on_convergence=False,
):
    """Adapt fixed spin occupations; retain the historical eleven-value result.

    Optional state_evaluator(D,C,eps,previous) returns energy/raw Fock/aux.
    A None previous requests a fresh physical build before convergence.
    """
    # DifferentiableSCF retains its symmetric frame by default; built-in
    # forward producers can supply their already prepared orbital factor.
    x = (_orthogonalizer(overlap, orthogonalization_eps) if orthogonalizer is None
         else jnp.asarray(orthogonalizer))
    dtype = jnp.asarray(density_spin).dtype
    occupations = jnp.asarray(mo_occ_spin, dtype=dtype)

    def evaluate(dm, coeff, energies, previous):
        if state_evaluator is not None:
            return state_evaluator(dm, coeff, energies, previous)
        values = fock_builder(dm, coeff, energies)
        energy = values[2] if len(values) == 3 else jnp.asarray(jnp.nan, dtype=dtype)
        return energy, values[1], ()

    effective, raw, *initial_energy = fock_builder(density_spin, mo_coeff_spin, mo_energy_spin)
    energy = initial_energy[0] if initial_energy else jnp.asarray(jnp.nan, dtype=dtype)
    config = SCFConfig(max_cycle=int(max_cycle),
        conv_tol=float(conv_tol_density**2 if conv_tol is None else conv_tol),
        conv_tol_density=float(conv_tol_density), conv_tol_grad=float(conv_tol_grad),
        damping=float(damping), energy_only=convergence_metric == "energy",
        refresh_on_convergence=bool(refresh_on_convergence))
    initial = SCFState(density_spin, mo_coeff_spin, mo_energy_spin, energy, raw, initial_aux)
    result = solve_scf(initial, evaluate=evaluate,
        diagonalize=jax.vmap(lambda f: _diagonalize_fock(f, x, eigenvalue_jitter=eigenvalue_jitter)),
        density_from_coeff=lambda c: jax.vmap(_build_density_from_occ)(c, occupations),
        diis_error=lambda f, state: jax.vmap(
            lambda fs, dm: orthonormal_diis_error(fs, dm, overlap, x))(f, state.density),
        orbital_gradient=lambda state: orbital_gradient_norm(state.fock, state.mo_coeff, occupations),
        shift_fock=lambda f, state: jax.vmap(
            lambda fs, dm: apply_level_shift(fs, overlap, 2*dm, level_shift))(f, state.density),
        config=config, initial_last_fock=effective)
    rms = result.history.density_rms
    finite = jnp.isfinite(rms)
    best_index = jnp.where(jnp.any(finite), jnp.argmin(jnp.where(finite, rms, jnp.inf)),
                           result.best_cycle-1)
    if iterate_selection == "best_rms":
        state, selected_index = result.best_state, best_index
    elif iterate_selection == "first_converged":
        first = jnp.argmax(result.history.converged.astype(jnp.int32))
        state = jax.lax.cond(result.converged, lambda _: result.state,
                             lambda _: result.best_state, None)
        selected_index = jnp.where(result.converged, first, best_index)
    else:
        state, selected_index = result.state, jnp.asarray(int(max_cycle)-1, jnp.int32)
    return (state.density, state.mo_coeff, state.mo_energy, state.fock,
            result.converged, result.cycles, rms, selected_index+1, best_index+1,
            rms[selected_index], rms[best_index])
