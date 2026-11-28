"""One isolated spin-adapted root in fixed active orbitals."""

from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from gradscf._memory import (
    array_device, array_storage_bytes, resolve_memory_mb, validate_reserved_bytes, workspace_bytes,
)
from gradscf.ao2mo import validate_integrals
from gradscf.ao2mo.reference import physical_integrals_valid
from gradscf.solvers import EigenResponseConfig, EigenSolverConfig, LinearOperator
from gradscf.solvers.diagnostics import require_converged_derivative
from gradscf.solvers.eigen import solve_hermitian
from gradscf.solvers.memory import apply_columns, plan_hermitian


class CASCIState(NamedTuple):
    active_energy: jax.Array
    total_energy: jax.Array
    ci_coeff: jax.Array
    ci_residual: jax.Array
    converged: jax.Array
    gap: jax.Array
    iterations: None
    response_valid: jax.Array


class _RitzGuard(NamedTuple):
    vector: jax.Array
    valid: jax.Array


def solve_casci(h1, eri, operators, *, ecore=0., config=None, response=None, ci0=None,
                max_memory=None, reserved_bytes=0, with_response=True, workload_callback=None):
    """Solve the lowest root of an exact fixed-total-spin CAS Hamiltonian.

    Numerical Hamiltonian parameters remain live through the shared Hermitian
    eigenpair response. The graph topology is static. The starting vector, if
    given, supplies a Davidson guess and fixes the local CI phase.
    An optional workload callback receives the shared Davidson diagnostics.
    """
    state, _ = _solve_casci(
        h1, eri, operators, ecore=ecore, config=config, response=response, ci0=ci0,
        max_memory=max_memory, reserved_bytes=reserved_bytes, with_response=with_response,
        workload_callback=workload_callback,
    )
    return state


def _solve_casci(h1, eri, operators, *, ecore=0., config=None, response=None, ci0=None,
                 max_memory=None, reserved_bytes=0, with_response=True, workload_callback=None,
                 guard0=None, retain_guard=False, _use_verified_residual=False):
    h1, eri = validate_integrals(h1, eri)
    if h1.shape[0] != operators.graph.norb:
        raise ValueError("Active integrals must match the CSF orbital count")
    ecore = jnp.asarray(ecore)
    if ecore.ndim != 0 or jnp.iscomplexobj(ecore):
        raise ValueError("ecore must be a real scalar")
    config = EigenSolverConfig(atol=1e-10) if config is None else config
    if config.nroots != 1 or config.value_min is not None:
        raise ValueError("CASCI requires the lowest single root within the selected spin sector")
    size = operators.size
    max_memory = resolve_memory_mb(max_memory)
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    dtype = jnp.result_type(h1, eri)
    physical = (physical_integrals_valid(h1, eri) & jnp.isfinite(ecore)).astype(dtype)
    columns = min(2, size)
    diagonal = None
    if config.method == "davidson":
        diagonal = jnp.where(physical != 0, operators.diagonal(h1, eri), jnp.nan)
    resident = reserved_bytes + operators.storage_bytes + array_storage_bytes(h1, eri, ecore, ci0)
    if guard0 is not None:
        resident += array_storage_bytes(guard0.vector, guard0.valid)
    initial = None
    if ci0 is not None:
        ci0 = jnp.asarray(ci0)
        if ci0.shape != (size,) or jnp.iscomplexobj(ci0):
            raise ValueError("ci0 must be a real vector with one coefficient per CSF")
        if config.method == "davidson":
            initial = ci0[:, None].astype(dtype)
            if size > 1:
                extra = jax.random.normal(jax.random.PRNGKey(config.seed), (size,), dtype=dtype)
                if guard0 is not None:
                    # Keep a small exploratory component. Two unchanged
                    # eigenvectors could otherwise miss a new lowest root
                    # when their span remains invariant after a crossing.
                    mixed = guard0.vector.astype(dtype) + (1.e-3 / size**0.5) * extra
                    extra = jnp.where(guard0.valid, mixed, extra)
                initial = jnp.stack((initial[:, 0], extra), axis=1)
    # Reserve the smallest legal Ritz search before choosing an action tile.
    # The actual solver chooses its optional capacities with the tile charged.
    minimum = config
    if config.method == "davidson":
        roots = min(config.nroots + 1, size)
        smallest_subspace = min(roots + 2, size)
        if config.max_subspace is not None:
            smallest_subspace = min(config.max_subspace, smallest_subspace)
        minimum = replace(config, max_subspace=smallest_subspace, initial_guess_count=roots,
                          max_trial_vectors=roots, collapse_subspace=roots)
    held = resident + array_storage_bytes(diagonal, initial)
    device = array_device(diagonal, h1, eri)
    _, solver_minimum, _ = plan_hermitian(
        size, dtype, minimum, max_memory, resident_bytes=held,
        initial_columns=0 if initial is None else initial.shape[1], device=device,
    )
    available = workspace_bytes(max_memory, resident_bytes=held + solver_minimum, device=device)
    workspace_dtype = "float64" if jax.config.x64_enabled else dtype
    width = operators.symmetric_pair_block_size(workspace_dtype, available, columns=columns)
    resident += operators.action_workspace_bytes(
        workspace_dtype, columns, symmetric=True, pair_block_size=width,
    )

    apply = operators._prepare_symmetric_action(h1, eri, pair_block_size=width)
    action = LinearOperator((size, size), dtype, apply)
    op = LinearOperator(
        action.shape, action.dtype,
        lambda c: jnp.where(physical != 0, action.matvec(c), jnp.nan),
        diagonal=diagonal,
        matmat=lambda c: jnp.where(physical != 0, apply_columns(action, c, columns), jnp.nan),
    )
    if response is None or not with_response:
        response = EigenResponseConfig(target="eigenpairs" if with_response else "eigenvalues")
    elif response.target != "eigenpairs":
        raise ValueError("CASCI exposes an eigenpair response for its single CI root")
    # CASSCF stops its forward-state diagnostics. Standalone CASCI keeps
    # the live residual, even when the CI state itself has no response.
    verified_residual = _use_verified_residual and not with_response
    result = solve_hermitian(
        op, config=config, response=response, initial_vectors=initial,
        max_memory=max_memory, reserved_bytes=resident, workload_callback=workload_callback,
        _return_residual_inf=verified_residual,
    )
    eig, residual_inf = result if verified_residual else (result, None)
    energy = eig.values[0] if with_response else eig.raw_values[0]
    c = eig.vectors[:, 0]
    largest = c[jnp.argmax(jnp.abs(c))]
    phase_metric = largest if ci0 is None else jnp.vdot(ci0, c).real
    phase_metric = jnp.where(phase_metric == 0, largest, phase_metric)
    c = c * jnp.where(phase_metric < 0, -1., 1.)
    residual = (residual_inf[0] if verified_residual
                else jnp.max(jnp.abs(apply(c) - energy * c)))
    valid = eig.response_valid[0] & (physical != 0)
    total = energy + ecore
    if with_response:
        total = require_converged_derivative(total, valid)
    state = CASCIState(energy, total, c, residual, eig.converged[0] & (physical != 0),
                       eig.boundary_gap, None, valid)
    saved_guard = None
    if retain_guard and config.method == "davidson" and size > 1:
        guard = eig.raw_vectors[:, 1]
        guard_valid = (eig.raw_present[1] & (eig.raw_residual_norms[1] <= config.atol)
                       & jnp.all(jnp.isfinite(guard)) & (physical != 0))
        saved_guard = _RitzGuard(guard, guard_valid)
    return state, saved_guard
