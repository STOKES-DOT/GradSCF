"""Shared bounded SCF scan; Fock/energy physics remains with the caller."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..nonlinear.diis import diis_extrapolate
from .metrics import convergence_reached
from .types import SCFConfig, SCFHistory, SCFMetrics, SCFResult, SCFState


class _Carry(NamedTuple):
    state: SCFState
    last_fock: jax.Array
    focks: jax.Array
    errors: jax.Array
    head: jax.Array
    count: jax.Array
    cycles: jax.Array
    converged: jax.Array
    metrics: SCFMetrics
    best_state: SCFState
    best_cycle: jax.Array
    best_rms: jax.Array
    fresh_evaluations: jax.Array
    state_is_fresh: jax.Array
    best_is_fresh: jax.Array


def _pack_error(error):
    error = jnp.asarray(error)
    if jnp.iscomplexobj(error):
        return jnp.concatenate((error.real.reshape(-1), error.imag.reshape(-1)))
    return error.reshape(-1)


def solve_scf(initial_state, *, evaluate, diagonalize, density_from_coeff,
              diis_error, orbital_gradient, config=None, shift_fock=None,
              initial_last_fock=None, energy_change=None):
    """Converge a matrix fixed point with DIIS, preserving explicit AD.

    ``evaluate(D,C,eps,previous)`` returns ``(energy,raw_fock,aux)``; previous
    is a state for incremental builds and None for a fresh convergence check.
    A failed fresh check switches subsequent advances to the full evaluator;
    those advances already supply their physical certificate without a repeat.
    ``diis_error(fock,state)`` may return a real or complex array. Complex errors
    use a real+imaginary metric. ``orbital_gradient(state)`` certifies the raw
    Fock, never the DIIS extrapolate. ``shift_fock(fock,state)`` acts after DIIS.

    All physical arrays and auxiliary PyTrees remain live to JAX. The fixed
    scan supports reverse mode without attaching an implicit rule or stopping
    gradients; callers retain their existing implicit/explicit differentiation.
    Only scalar histories are retained unless store_iterates is requested.
    """
    cfg = SCFConfig() if config is None else config
    dtype = initial_state.fock.real.dtype
    error_shape = jax.eval_shape(lambda f, state: _pack_error(diis_error(f, state)),
                                initial_state.fock, initial_state)
    zero = jnp.asarray(0, dtype=jnp.int32)
    infinity = jnp.asarray(jnp.inf, dtype=dtype)
    initial = _Carry(initial_state,
        initial_state.fock if initial_last_fock is None else initial_last_fock,
        jnp.zeros((cfg.diis_space,)+initial_state.fock.shape, dtype=initial_state.fock.dtype),
        jnp.zeros((cfg.diis_space,)+error_shape.shape, dtype=error_shape.dtype),
        zero, zero, zero, jnp.asarray(False), SCFMetrics(infinity, infinity, infinity),
        initial_state, zero, infinity, zero, jnp.asarray(True), jnp.asarray(True))

    def finite_state(state):
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in
            (state.density, state.mo_coeff, state.mo_energy, state.energy, state.fock)]))

    def assess(old, state, density_rms):
        delta = (state.energy-old.state.energy if energy_change is None
                 else energy_change(old.state, state))
        metrics = SCFMetrics(jnp.abs(delta), density_rms, orbital_gradient(state))
        passed = convergence_reached(*metrics, conv_tol=cfg.conv_tol,
            conv_tol_density=cfg.conv_tol_density, conv_tol_grad=cfg.conv_tol_grad,
            energy_only=cfg.energy_only, has_prior_cycle=old.cycles > 0)
        return metrics, passed & finite_state(state)

    def advance(old):
        damp = ((old.cycles < cfg.fock_damping_cycles)
                & ((old.cycles > 0) | (not cfg.skip_first_fock_damping)))
        factor = jnp.where(damp, jnp.asarray(cfg.damping, dtype=dtype), 0.)
        fock = (1-factor)*old.state.fock+factor*old.last_fock

        def extrapolate(_):
            return diis_extrapolate(fock, _pack_error(diis_error(fock, old.state)),
                                    old.focks, old.errors, old.head, old.count)

        effective, focks, errors, head, count = jax.lax.cond(
            old.cycles >= cfg.diis_start_cycle, extrapolate,
            lambda _: (fock, old.focks, old.errors, old.head, old.count), operand=None)
        shifted = effective if shift_fock is None else shift_fock(effective, old.state)
        energies, coeff = diagonalize(shifted)
        density = density_from_coeff(coeff)
        if cfg.density_damping != "none":
            active = jnp.asarray(True) if cfg.density_damping == "all" else count < 2
            mixing = jnp.where(active, jnp.asarray(cfg.damping, dtype=dtype), 0.)
            density = (1-mixing)*density+mixing*old.state.density
        full_tail = jnp.asarray(False)
        if cfg.refresh_on_convergence and cfg.conv_tol > 0:
            # A prior fresh failure starts a permanent full-evaluator tail.
            # The seed is fresh too, so exclude cycle zero from this transition.
            full_tail = old.state_is_fresh & (old.cycles > 0)
            energy, raw_fock, aux = jax.lax.cond(full_tail,
                lambda _: evaluate(density, coeff, energies, None),
                lambda _: evaluate(density, coeff, energies, old.state), None)
        else:
            energy, raw_fock, aux = evaluate(density, coeff, energies, old.state)
        state = SCFState(density, coeff, energies, energy, raw_fock, aux)
        rms = jnp.sqrt(jnp.mean(jnp.abs(density-old.state.density)**2))
        metrics, passed = assess(old, state, rms)
        refreshed = jnp.asarray(False)
        if cfg.refresh_on_convergence and cfg.conv_tol > 0:
            # A drifting incremental energy must not prevent its own rebase.
            # The full rebuild still has to pass every original convergence gate.
            ready = ((old.cycles > 0) & finite_state(state)
                & jnp.isfinite(metrics.density_rms) & jnp.isfinite(metrics.gradient)
                & (metrics.density_rms < cfg.conv_tol_density)
                & (metrics.gradient < cfg.conv_tol_grad))
            refreshed = ~full_tail & (passed | ready)
            def refresh(_):
                energy, fock, aux = evaluate(density, coeff, energies, None)
                fresh = SCFState(density, coeff, energies, energy, fock, aux)
                fresh_metrics, fresh_passed = assess(old, fresh, rms)
                return fresh, fresh_metrics, fresh_passed
            state, metrics, passed = jax.lax.cond(
                refreshed, refresh, lambda unchanged: unchanged, (state, metrics, passed))
        return old._replace(state=state, last_fock=effective, focks=focks,
            errors=errors, head=head, count=count, cycles=old.cycles+1,
            converged=passed, metrics=metrics,
            fresh_evaluations=old.fresh_evaluations+refreshed.astype(jnp.int32),
            state_is_fresh=full_tail | refreshed | jnp.asarray(not cfg.refresh_on_convergence))

    def step(old, _):
        current = jax.lax.cond(old.converged, lambda value: value, advance, old)
        rms = jnp.where(old.converged, 0., current.metrics.density_rms)
        delta = jnp.where(old.converged, 0., current.metrics.energy_delta)
        better = jnp.isfinite(rms) & (rms < old.best_rms)
        # A fresh rebuild of the identical physical state also replaces stale
        # E/F/aux in a tied-density best record, without changing its cycle/frame.
        same = (jnp.all(current.state.density == old.best_state.density)
                & jnp.all(current.state.mo_coeff == old.best_state.mo_coeff)
                & jnp.all(current.state.mo_energy == old.best_state.mo_energy))
        rebased_best = jnp.asarray(cfg.refresh_on_convergence) & current.state_is_fresh & same
        updated_best = old.best_state._replace(energy=current.state.energy,
            fock=current.state.fock, aux=current.state.aux)
        best = jax.lax.cond(better, lambda _: current.state,
            lambda _: jax.lax.cond(rebased_best, lambda _: updated_best,
                                  lambda _: old.best_state, None), None)
        current = current._replace(best_state=best,
            best_cycle=jnp.where(better, current.cycles, old.best_cycle),
            best_rms=jnp.where(better, rms, old.best_rms),
            best_is_fresh=jnp.where(better, current.state_is_fresh,
                                   rebased_best | old.best_is_fresh))
        iterates = ((current.state.density, current.state.mo_coeff, current.state.mo_energy)
                    if cfg.store_iterates else ())
        return current, SCFHistory(rms, delta, current.metrics.gradient, current.converged, iterates)

    final, history = jax.lax.scan(step, initial, xs=None, length=cfg.max_cycle)
    # A wholly invalid history must expose the failed iterate, not the seed.
    valid_best = jnp.isfinite(final.best_rms)
    best = jax.lax.cond(valid_best, lambda _: final.best_state, lambda _: final.state, operand=None)
    best_cycle = jnp.where(valid_best, final.best_cycle, final.cycles)
    fresh_evaluations = final.fresh_evaluations
    if cfg.refresh_on_convergence and cfg.conv_tol > 0:
        # A consumer may select a different historical best state. Certify its
        # values once before returning it; this is not another main advance.
        def refresh_best(state):
            energy, fock, aux = evaluate(state.density, state.mo_coeff, state.mo_energy, None)
            return state._replace(energy=energy, fock=fock, aux=aux)
        needed = valid_best & ~final.best_is_fresh
        best = jax.lax.cond(needed, refresh_best, lambda state: state, best)
        fresh_evaluations = fresh_evaluations+needed.astype(jnp.int32)
    return SCFResult(final.state, best, best_cycle, final.converged,
                     final.cycles, final.metrics, history, final.cycles, fresh_evaluations)
