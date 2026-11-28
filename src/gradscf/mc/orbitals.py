"""Single-state CASSCF orbital optimization and coupled stationary response."""
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm, solve_triangular
from jaxtyping import Array

from gradscf._memory import DEFAULT_MAX_MEMORY_MB, array_device, array_storage_bytes
from gradscf.ao2mo import transform_eri_block
from gradscf.ao2mo.active import core_potential, orbital_selection, transform_active_blocks
from gradscf.ao2mo.blocks import _eri_block_size
from gradscf.solvers import EigenSolverConfig, LinearOperator, LinearSolverConfig, solve_hermitian
from gradscf.solvers.diagnostics import require_converged_derivative
from gradscf.solvers.memory import plan_hermitian
from gradscf.solvers.nonlinear.root import attach_root

from .casci import _RitzGuard, _solve_casci
from .integrals import transform_active


class CASSCFState(NamedTuple):
    active_energy: Array
    total_energy: Array
    ci_coeff: Array
    ci_residual: Array
    converged: Array
    gap: Array
    iterations: Array
    response_valid: Array
    mo_coeff: Array
    orbital_gradient: Array
    energy_change: Array


class _MacroState(NamedTuple):
    coefficients: Array
    ci: NamedTuple
    guard: _RitzGuard | None
    density: tuple[Array, ...]
    gradient: Array
    energy_change: Array
    cycles: Array
    accepted: Array


def _rotation_pairs(nmo, core, active):
    outside = tuple(i for i in range(nmo) if i not in core and i not in active)
    return (tuple((p, q) for q in core for p in active)
            + tuple((p, q) for q in core for p in outside)
            + tuple((p, q) for p in outside for q in active))


def _rotation_matrix(angles, pairs, nmo, *, skew=False):
    rows, cols = jnp.asarray(pairs, dtype=jnp.int32).reshape(-1, 2).T
    excitation = jnp.zeros((nmo, nmo), dtype=angles.dtype).at[rows, cols].set(angles, unique_indices=True)
    return excitation - excitation.T if skew else excitation


def _orbital_gradient(coefficients, density, pairs, active_hamiltonian):
    """Pull the coefficient gradient back through the rotation at zero."""
    if not pairs:
        return jnp.zeros(0, dtype=coefficients.dtype)
    d1, d2 = density

    def energy(frame):
        h, g, constant = active_hamiltonian(frame)
        return constant + jnp.sum(h * d1) + .5 * jnp.sum(g * d2)

    coefficient_gradient = jax.grad(energy)(coefficients)
    orbital_gradient = coefficients.T @ coefficient_gradient
    row, col = jnp.asarray(pairs, dtype=jnp.int32).T
    return orbital_gradient[row, col] - orbital_gradient[col, row]


def _reorthogonalize(coefficients, overlap):
    lower = jnp.linalg.cholesky(coefficients.T @ overlap @ coefficients)
    return solve_triangular(lower, coefficients.T, lower=True).T


def _metric_action(rdm1, core, active, nmo, *, whiten=False):
    """Return a blockwise singles overlap or its inverse-square-root action."""
    ncore, nact = len(core), len(active)
    nvirt = nmo - ncore - nact
    nca, ncv = ncore * nact, ncore * nvirt
    if whiten:
        occupations, eigenvectors = jnp.linalg.eigh(rdm1)

        def inverse_sqrt(values):
            positive = values > 1e-12
            weights = jnp.where(positive, jax.lax.rsqrt(jnp.where(positive, values, 1.)), 0.)
            return (eigenvectors * weights) @ eigenvectors.T

        virtual_active = inverse_sqrt(occupations)
        active_core = inverse_sqrt(2. - occupations)
        core_virtual = 2. ** -0.5
    else:
        virtual_active = rdm1
        active_core = 2. * jnp.eye(nact, dtype=rdm1.dtype) - rdm1
        core_virtual = 2.

    def action(angles):
        left = angles[:nca].reshape(ncore, nact)
        middle = angles[nca:nca+ncv]
        right = angles[nca+ncv:].reshape(nvirt, nact)
        return jnp.concatenate(((left @ active_core).reshape(-1), middle * core_virtual,
                                (right @ virtual_active).reshape(-1)))

    return action


def _transition_energy(overlap, hcore, coefficients, core, active, pairs, rdm1, rdm2,
                       bra_angles, ket_angles, integral_options):
    """Biorthogonal matrix element in the two infinitesimal singles frames.

    After core projection, the active cross-overlap is I + O(bra * ket).
    Restoring its determinant representation would insert an active operator
    against (H-E) on the reference. That term vanishes at the CASCI root.
    The analogous core determinant prefactor multiplies the vanishing
    reference expectation of (H-E). Normalization therefore gives the exact
    mixed derivative of <bra|(H-E)|ket> at the reference, using only RDM1/2.
    """
    nmo = coefficients.shape[1]
    core_index = jnp.asarray(core, dtype=jnp.int32)
    active_index = jnp.asarray(active, dtype=jnp.int32)

    def frame(angles):
        matrix = jnp.eye(nmo, dtype=angles.dtype) + _rotation_matrix(angles, pairs, nmo)
        return coefficients @ matrix[:, active_index], coefficients @ matrix[:, core_index]

    bra_active, bra_core = frame(bra_angles)
    ket_active, ket_core = frame(ket_angles)
    if core:
        metric_core = bra_core.T @ overlap @ ket_core
        ket_core = jnp.linalg.solve(metric_core.T, ket_core.T).T
        bra_active = bra_active - bra_core @ (ket_core.T @ overlap @ bra_active)
        ket_active = ket_active - ket_core @ (bra_core.T @ overlap @ ket_active)
    metric_active = bra_active.T @ overlap @ ket_active
    ket_active = jnp.linalg.solve(metric_active.T, ket_active.T).T

    g_active = transform_eri_block((bra_active, ket_active, bra_active, ket_active),
                                   **integral_options)
    constant = jnp.zeros((), dtype=hcore.dtype)
    effective = hcore
    if core:
        density = 2. * ket_core @ bra_core.T
        potential = core_potential(density, integral_options, symmetric=False)
        constant = jnp.sum(density * (hcore + 0.5 * potential).T)
        effective = hcore + potential
    h_active = bra_active.T @ effective @ ket_active
    return constant + jnp.sum(h_active * rdm1) + 0.5 * jnp.sum(g_active * rdm2)


def _super_ci_action(overlap, hcore, coefficients, rdm1, rdm2, core, active, pairs,
                     integral_options):
    zero = jnp.zeros(len(pairs), dtype=coefficients.dtype)

    def energy(left, right):
        return _transition_energy(overlap, hcore, coefficients, core, active, pairs,
                                  rdm1, rdm2, left, right, integral_options)

    left_gradient = jax.grad(energy, argnums=0)
    return jax.linearize(lambda right: left_gradient(zero, right), zero)


def _super_ci_model(overlap, hcore, coefficients, rdm1, rdm2, core, active, pairs,
                    **integral_options):
    """Materialize the singles metric for small validation models."""
    identity = jnp.eye(len(pairs), dtype=coefficients.dtype)
    metric_action = _metric_action(rdm1, core, active, coefficients.shape[1])
    metric = jax.vmap(metric_action, in_axes=1, out_axes=1)(identity)
    coupling, action = _super_ci_action(
        overlap, hcore, coefficients, rdm1, rdm2, core, active, pairs, integral_options)
    return metric, coupling, action


def _super_ci_davidson_config(nrot):
    return EigenSolverConfig(method="davidson", nroots=1, max_subspace=min(nrot, 24))


def _lowest_super_ci_vector(coupling, action, diagonal, max_memory, *, reserved_bytes=0):
    """Lowest reference/singles root in the orthonormalized projected space."""
    n = coupling.size
    # Raise the singles a little to regularize the trial step and keep
    # projected-out zero-norm directions above the reference.
    level_shift = jnp.asarray(1e-3, dtype=coupling.dtype)

    def matvec(v):
        excited = v[1:]
        out = action(excited) + level_shift * excited
        return jnp.concatenate((jnp.vdot(coupling, excited)[None],
                                out + v[0] * coupling))

    if n <= 64:
        matrix = jax.vmap(matvec, in_axes=1, out_axes=1)(jnp.eye(n + 1, dtype=coupling.dtype))
        return jnp.linalg.eigh((matrix + matrix.T) * 0.5)[1][:, 0]

    # A diagonal estimate suffices as the Davidson preconditioner. The
    # actual Ritz construction always uses the full projected action.
    diag = jnp.concatenate((jnp.zeros(1, dtype=coupling.dtype), diagonal + level_shift))
    op = LinearOperator((n + 1, n + 1), coupling.dtype, matvec, diagonal=diag)
    solver = _super_ci_davidson_config(n)
    initial = jnp.zeros((n + 1, 2), dtype=coupling.dtype).at[0, 0].set(1.)
    # A second linearly independent vector is needed for the spectral guard.
    initial = initial.at[1:, 1].set(jnp.cos(jnp.arange(n, dtype=coupling.dtype) + 0.3))
    result = solve_hermitian(op, config=solver, initial_vectors=initial,
                             max_memory=max_memory, reserved_bytes=reserved_bytes)
    return jnp.where(result.converged[0], result.vectors[:, 0], jnp.nan)


def _packed_super_ci_resident_bytes(hcore, coefficients, d1, d2, nact, packed):
    packed = jnp.asarray(packed)
    nvirt = coefficients.shape[1] - nact
    itemsize = jnp.result_type(hcore, coefficients, d1, d2, packed).itemsize
    # Two four-index blocks, h_vv, metric and active correction remain live
    # through the matrix-free Ritz solve. The AO source also remains resident.
    return (array_storage_bytes(hcore, coefficients, d1, d2, packed)
            + itemsize * (2 * (nact * nvirt)**2 + nvirt**2 + 2 * nact**2))


def _packed_super_ci_blocks_fit(hcore, coefficients, d1, d2, nact, options):
    """Check the packed contractions before selecting the MO route."""
    packed = jnp.asarray(options["eri_pair_matrix"])
    nao, nmo = coefficients.shape
    nvirt = nmo - nact
    ca = jax.ShapeDtypeStruct((nao, nact), coefficients.dtype)
    cv = jax.ShapeDtypeStruct((nao, nvirt), coefficients.dtype)
    blocks = ((ca,) * 4, (cv, ca, ca, ca), (cv, cv, ca, ca), (cv, ca, ca, cv))
    output = nact**4 + nvirt * nact**3 + 2 * (nact * nvirt)**2
    itemsize = jnp.result_type(coefficients, hcore, packed).itemsize
    # Charge all block outputs and the active/external coefficient gathers.
    reserved = (array_storage_bytes(hcore, coefficients, d1, d2)
                + itemsize * (2 * output + nao * nmo + nmo**2))
    try:
        for spaces in blocks:
            _eri_block_size(packed, spaces, "packed", options["max_memory"],
                            reserved_bytes=reserved)
        nrot = nact * nvirt
        if nrot > 64:
            dtype = jnp.result_type(hcore, coefficients, d1, d2, packed)
            resident = _packed_super_ci_resident_bytes(hcore, coefficients, d1, d2, nact, packed)
            # Charge the diagonal and the two starting vectors as well.
            plan_hermitian(nrot + 1, dtype, _super_ci_davidson_config(nrot), options["max_memory"],
                           resident_bytes=resident + 3 * (nrot + 1) * dtype.itemsize,
                           initial_columns=2, device=array_device(packed, coefficients))
    except ValueError:
        return False
    return True


def _no_core_super_ci_action(hcore, coefficients, rdm1, rdm2, active, integral_options):
    """Exact mixed Super-CI derivative in MO blocks for an S-orthonormal frame.

    With active and external columns A and V, the varied active frames are
    A+VX and A+VY. Their cross metric is I+X.T@Y, whose inverse contributes
    -X.T@Y to the mixed derivative. This yields the subtraction below.
    """
    external = tuple(i for i in range(coefficients.shape[1]) if i not in active)
    ca = coefficients[:, jnp.asarray(active, dtype=jnp.int32)]
    cv = coefficients[:, jnp.asarray(external, dtype=jnp.int32)]
    h_aa, eri_aaaa, _, (eri_vaaa, eri_vvaa, eri_vaav) = transform_active_blocks(
        hcore, coefficients, blocks=("vaaa", "vvaa", "vaav"), core=(), active=active,
        **integral_options,
    )
    h_v = cv.T @ hcore
    coupling = (jnp.einsum("xj,ij->xi", h_v @ ca, rdm1)
                + jnp.einsum("xjkl,ijkl->xi", eri_vaaa, rdm2))
    correction = h_aa.T @ rdm1 + jnp.einsum("pikl,pjkl->ij", eri_aaaa, rdm2)
    h_vv = h_v @ cv

    def action(angles):
        y = angles.reshape(len(external), len(active))
        result = h_vv @ y @ rdm1.T - y @ correction.T
        result += jnp.einsum("xykl,ijkl,yj->xi", eri_vvaa, rdm2, y)
        result += jnp.einsum("xjky,ijkl,yl->xi", eri_vaav, rdm2, y)
        return result.reshape(-1)

    return coupling.reshape(-1), action


def _super_ci_step(overlap, hcore, coefficients, rdm1, rdm2, core, active, pairs,
                   integral_options, step_limit=0.20):
    packed = integral_options.get("eri_pair_matrix")
    reserved = 0
    if (not core and pairs and packed is not None and _packed_super_ci_blocks_fit(
            hcore, coefficients, rdm1, rdm2, len(active), integral_options)):
        coupling, action = _no_core_super_ci_action(
            hcore, coefficients, rdm1, rdm2, active, integral_options,
        )
        if len(pairs) > 64:
            reserved = _packed_super_ci_resident_bytes(
                hcore, coefficients, rdm1, rdm2, len(active), packed,
            )
    else:
        coupling, action = _super_ci_action(
            overlap, hcore, coefficients, rdm1, rdm2, core, active, pairs, integral_options,
        )
    whiten = _metric_action(rdm1, core, active, coefficients.shape[1], whiten=True)
    one_body = jnp.einsum("pi,pq,qi->i", coefficients, hcore, coefficients)
    rows, cols = jnp.asarray(pairs, dtype=jnp.int32).T
    diagonal = jnp.abs(one_body[rows] - one_body[cols])
    vector = _lowest_super_ci_vector(
        whiten(coupling), lambda x: whiten(action(whiten(x))), diagonal,
        integral_options["max_memory"], reserved_bytes=reserved,
    )
    sign = jnp.where(vector[0] >= 0, 1., -1.)
    step = whiten(vector[1:]) * sign / jnp.maximum(jnp.abs(vector[0]), 1e-10)
    step = jnp.where(jnp.all(jnp.isfinite(step)) & (jnp.vdot(step, coupling) < 0.),
                     step, -coupling)
    size = jnp.linalg.norm(step)
    return step * (step_limit / jnp.maximum(size, step_limit))


def solve_casscf(overlap, hcore, mo_coeff, enuc, operators, *, core, active,
                 eri=None, eri_pair_matrix=None, df_factors=None, direct_basis=None,
                 max_memory=DEFAULT_MAX_MEMORY_MB, config=None, response=None,
                 ci0=None, max_cycle=100, conv_tol=1e-10, conv_tol_grad=1e-7,
                 with_response=True, workload_callback=None, _recycle_ci=True):
    """Optimize an isolated real-space CASSCF ground state in a spin sector.

    Forward orbital steps solve the projected Super-CI eigenproblem. The
    differentiable solution obeys the physical stationarity and CI equations
    in an overlap-orthonormal orbital chart. Davidson guesses use the
    previous accepted CI state in the continuous orbital chart. The current
    Hamiltonian always determines the roots and derivative validity.
    An optional workload callback receives CI Davidson diagnostics.
    """
    if isinstance(max_cycle, bool) or not isinstance(max_cycle, int) or max_cycle < 1:
        raise ValueError("max_cycle must be a positive integer")
    if min(conv_tol, conv_tol_grad) <= 0 or not all(map(isfinite, (conv_tol, conv_tol_grad))):
        raise ValueError("CASSCF convergence tolerances must be positive and finite")
    overlap, hcore, mo_coeff, enuc = map(jnp.asarray, (overlap, hcore, mo_coeff, enuc))
    if mo_coeff.ndim != 2 or mo_coeff.shape[0] < mo_coeff.shape[1]:
        raise ValueError("mo_coeff must be a common spatial orbital frame")
    nao, nmo = mo_coeff.shape
    if overlap.shape != (nao, nao) or hcore.shape != (nao, nao):
        raise ValueError("overlap and hcore must match the AO dimension of mo_coeff")
    if any(jnp.iscomplexobj(x) for x in (overlap, hcore, mo_coeff, enuc)):
        raise NotImplementedError("CASSCF requires real orbital and integral data")
    core, active = orbital_selection(nmo, core, active)
    if len(active) != operators.graph.norb:
        raise ValueError("active orbital count must match the CSF operators")
    pairs = _rotation_pairs(nmo, core, active)
    integral_options = dict(eri=eri, eri_pair_matrix=eri_pair_matrix, df_factors=df_factors,
                            direct_basis=direct_basis, max_memory=max_memory)

    def active_hamiltonian(coefficients):
        return transform_active(hcore, coefficients, core=core, active=active,
                                ecore=enuc, **integral_options)

    def rotate(coefficients, angles):
        return coefficients @ expm(_rotation_matrix(angles, pairs, nmo, skew=True))

    def density_and_gradient(coefficients, ci_coeff):
        density = operators.rdm12(ci_coeff) if pairs else ()
        return density, _orbital_gradient(coefficients, density, pairs, active_hamiltonian)

    def evaluate(coefficients, previous, ci_seed, guard_seed=None):
        h, g, constant = active_hamiltonian(coefficients)
        ci, guard = _solve_casci(
            h, g, operators, ecore=constant, config=config, ci0=ci_seed, guard0=guard_seed,
            retain_guard=bool(pairs) and _recycle_ci, response=response, with_response=False,
            max_memory=max_memory, workload_callback=workload_callback,
            _use_verified_residual=True,
        )
        return ci, guard, ci.total_energy - previous

    seed = _reorthogonalize(mo_coeff, overlap)
    ci, seed_guard, _ = evaluate(seed, jnp.asarray(0., dtype=seed.dtype), ci0)
    seed_density, seed_gradient = density_and_gradient(seed, ci.ci_coeff)
    initial = _MacroState(seed, ci, seed_guard, seed_density, seed_gradient,
                          jnp.zeros((), dtype=seed.dtype), jnp.asarray(0), jnp.asarray(True))

    def stationary(state):
        return (state.ci.converged
                & (jnp.max(jnp.abs(state.gradient), initial=0.) <= conv_tol_grad)
                & (jnp.abs(state.energy_change) <= conv_tol))

    def macro_condition(state):
        return (state.cycles < max_cycle) & state.accepted & state.ci.converged & ~stationary(state)

    def macro_iteration(state):
        density1, density2 = state.density
        step = _super_ci_step(overlap, hcore, state.coefficients, density1, density2,
                              core, active, pairs, integral_options)
        previous = state.ci.total_energy
        rounding = 32. * jnp.finfo(previous.dtype).eps * jnp.maximum(1., jnp.abs(previous))

        def attempt(scale):
            coefficients = rotate(state.coefficients, step * scale)
            ci, guard, change = evaluate(coefficients, previous, state.ci.ci_coeff, state.guard)
            accept = ci.converged & jnp.isfinite(change) & (change <= rounding)
            return coefficients, ci, guard, change, accept

        def search_condition(trial):
            count, _, _, _, _, accepted = trial
            return (count < 14) & ~accepted

        def search(trial):
            count, _, _, _, _, _ = trial
            return (count + 1, *attempt(jnp.exp2(-count.astype(previous.dtype))))

        _, coefficients, ci, guard, change, accepted = jax.lax.while_loop(
            search_condition, search,
            (jnp.asarray(0), state.coefficients, state.ci, state.guard, state.energy_change,
             jnp.asarray(False)))
        def accept_step():
            density, gradient = density_and_gradient(coefficients, ci.ci_coeff)
            return _MacroState(coefficients, ci, guard, density, gradient, change,
                               state.cycles + 1, accepted)

        return jax.lax.cond(
            accepted, accept_step,
            lambda: state._replace(cycles=state.cycles + 1, accepted=accepted),
        )

    # Forward iterations choose a root only. The coupled equations below
    # provide every derivative of the returned physical state.
    if pairs:
        primal = jax.lax.while_loop(macro_condition, macro_iteration, initial)
    else:
        primal = initial
    primal = jax.tree.map(jax.lax.stop_gradient, primal)
    valid = stationary(primal) & primal.ci.response_valid
    coeff, ci_coeff, active_energy = (primal.coefficients, primal.ci.ci_coeff, primal.ci.active_energy)
    total_energy = primal.ci.total_energy

    if with_response:
        angles0 = jnp.zeros(len(pairs), dtype=coeff.dtype)
        center = _reorthogonalize(coeff, overlap)
        state0 = jnp.concatenate((angles0, ci_coeff, active_energy[None]))

        def decode(state):
            angles = state[:len(pairs)]
            coefficients = rotate(center, angles) if pairs else center
            c, e = state[len(pairs):-1], state[-1]
            return coefficients, c, e

        def residual(state):
            coefficients, c, e = decode(state)
            h, g, _ = active_hamiltonian(coefficients)
            _, g_orbital = density_and_gradient(coefficients, c)
            g_ci = 2. * (operators.matvec_symmetric(h, g, c) - e * c)
            normalization = 1. - jnp.vdot(c, c)
            return jnp.concatenate((g_orbital, g_ci, normalization[None]))

        linear_config = (LinearSolverConfig(method="direct", rtol=1e-9, atol=1e-12)
                         if state0.size <= 128 else
                         LinearSolverConfig(rtol=1e-9, atol=1e-12, restart=40))
        if response is not None:
            linear_config = response.linear_config
        solution = attach_root(residual, state0, config=linear_config, converged=valid,
                               max_memory=max_memory)
        coeff, ci_coeff, active_energy = decode(solution)
        # The active eigenvalue already contains the optimized CI response.
        # Add the live nonactive orbital and nuclear terms in the same chart.
        _, _, constant = active_hamiltonian(coeff)
        total_energy = active_energy + constant
        active_energy, total_energy, ci_coeff, coeff = jax.tree.map(
            lambda x: require_converged_derivative(x, valid),
            (active_energy, total_energy, ci_coeff, coeff),
        )

    return CASSCFState(active_energy, total_energy, ci_coeff, primal.ci.ci_residual,
                       stationary(primal), primal.ci.gap, primal.cycles, valid,
                       coeff, primal.gradient, primal.energy_change)
