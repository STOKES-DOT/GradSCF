"""Native-only atomic projection and real DF-HF branch diagnostics."""
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import integrals
from gradscf.integrals.basis.data import load_basis_from_snapshot
from gradscf.integrals.molecular.jk import build_jk_from_df
from gradscf.integrals.molecular.density_fitting import unpack_factors
from gradscf.scf.energy import XCContribution, unrestricted_energy, unrestricted_fock
from nnao_core.native import _solver, basis_parameters


def atomic_occupations(charge, shell_shapes):
    """Spherically average neutral main-group populations in minimal radial AOs."""
    shells, counters, offset = [], defaultdict(int), 0
    for momentum, contractions in shell_shapes:
        for _ in range(contractions):
            n = momentum+1+counters[momentum]
            counters[momentum] += 1
            width = 2*momentum+1
            shells.append((n+momentum, n, offset, width))
            offset += width
    result, remaining = np.zeros(offset), int(charge)
    for _, _, start, width in sorted(shells):
        occupied = min(remaining, 2*width)
        result[start:start+width] = occupied/width
        remaining -= occupied
    if remaining or charge <= 0:
        raise ValueError('Minimal all-electron reference cannot represent the atom.')
    return result


def _width(shell):
    return (2*int(shell[0])+1)*(len(shell[1])-1)


def minao_guess(row, atom_shells, overlap, nuclear_charges, *, nelectron, spin):
    """Project native minimal atomic densities; ghosts have no reference electrons.

    This only selects an initial density. The converged Hamiltonian and basis
    are unchanged. Global spin traces are normalized after basis projection.
    """
    combined, target, reference, populations, offset = [], [], [], [], 0
    for symbol, shells, charge in zip(row['symbols'], atom_shells, nuclear_charges):
        minimal = load_basis_from_snapshot('minao', symbol) if charge else []
        count = sum(_width(s) for s in shells)
        target.extend(range(offset, offset+count))
        offset += count
        count = sum(_width(s) for s in minimal)
        reference.extend(range(offset, offset+count))
        offset += count
        if minimal:
            shapes = [(int(s[0]), len(s[1])-1) for s in minimal]
            populations.extend(atomic_occupations(int(charge), shapes))
        combined.append(shells+minimal)
    with jax.default_device(jax.devices('cpu')[0]):
        topology, parameters, _ = basis_parameters(row, combined)
        matrix = np.asarray(integrals.make_plan(topology).evaluate('overlap', parameters))
    s = np.asarray(overlap)
    np.testing.assert_allclose(matrix[np.ix_(target, target)], s, atol=2e-12, rtol=0)
    projected = np.linalg.solve(s, matrix[np.ix_(target, reference)])
    density = (projected*np.asarray(populations))@projected.T
    trace = np.einsum('pq,qp->', density, s)
    if not np.isfinite(trace) or trace <= 0:
        raise ValueError('Projected atomic density has no finite electrons.')
    counts = np.array([(nelectron+spin)/2, (nelectron-spin)/2])
    return np.stack([density*n/trace for n in counts])


def df_curvature(hcore, factors, coefficients, occupation_rows):
    """Exact real spin-coupled orbital gradient/HVP; no four-index ERIs.

    The Hessian acts in occupied-virtual rotations with dC_occ=-C_vir*x.T.
    Precomputed spin MO blocks keep repeated Davidson actions inexpensive.
    """
    c, b = jnp.asarray(coefficients), jnp.asarray(factors)
    occ = np.asarray(occupation_rows)
    if c.ndim != 3 or c.shape[0] != 2 or occ.shape != (2, c.shape[-1]):
        raise ValueError('Expected complete real spin MO matrices and occupations.')
    if np.any((occ != 0) & (occ != 1)) or jnp.iscomplexobj(c):
        raise ValueError('Real DF stability requires integer spin occupations.')
    occupied = [np.flatnonzero(o == 1) for o in occ]
    virtual = [np.flatnonzero(o == 0) for o in occ]
    density = jnp.einsum('spi,si,sqi->spq', c, jnp.asarray(occ), c)
    coulomb, _ = build_jk_from_df(b, density.sum(axis=0))
    exchange = jnp.stack([build_jk_from_df(b, dm)[1] for dm in density])
    fock = hcore[None]+coulomb[None]-exchange
    fmo = jnp.einsum('spi,spq,sqj->sij', c, fock, c)
    mo = jnp.einsum('spi,Qpq,sqj->sQij', c, b, c,
                    precision=jax.lax.Precision.HIGHEST)
    blocks, gradients, diagonals, shapes = [], [], [], []
    for s, (i, a) in enumerate(zip(occupied, virtual)):
        ix = lambda x, left, right: x[..., left[:, None], right]
        oo, ov, vv = ix(mo[s], i, i), ix(mo[s], i, a), ix(mo[s], a, a)
        foo, fov, fvv = ix(fmo[s], i, i), ix(fmo[s], i, a), ix(fmo[s], a, a)
        blocks.append((oo, ov, vv, foo, fvv))
        gradients.append(-2*fov.reshape(-1))
        eps = jnp.diag(fmo[s])
        diagonals.append((2*(eps[a][None, :]-eps[i][:, None])).reshape(-1))
        shapes.append((len(i), len(a)))
    sizes = [int(np.prod(shape)) for shape in shapes]

    def apply(vector):
        xs = [vector[:sizes[0]].reshape(shapes[0]), vector[sizes[0]:].reshape(shapes[1])]
        rho = sum(jnp.einsum('Qia,ia->Q', block[1], x) for block, x in zip(blocks, xs))
        outputs = []
        for (oo, ov, vv, foo, fvv), x in zip(blocks, xs):
            response = -2*jnp.einsum('Qia,Q->ia', ov, rho)
            response += jnp.einsum('Qij,jb,Qba->ia', oo, x, vv)
            response += jnp.einsum('Qib,jb,Qja->ia', ov, x, ov)
            outputs.append((2*(x@fvv-foo@x-response)).reshape(-1))
        return jnp.concatenate(outputs)
    return jnp.concatenate(gradients), jax.jit(apply), jnp.concatenate(diagonals)


def _physical(state, overlap, hcore, factors, enuc, nelectron, spin):
    density = state['density_spin']
    coulomb, _ = build_jk_from_df(factors, density.sum(axis=0))
    exchange = jnp.stack([build_jk_from_df(factors, dm)[1] for dm in density])
    xc = XCContribution(0., 0., 1.)
    fock = unrestricted_fock(hcore, coulomb, exchange, xc)
    energy = unrestricted_energy(density, hcore, coulomb, exchange, xc, nuclear_repulsion=enuc)
    residual = float(jnp.linalg.norm(fock@density@overlap-overlap@density@fock))
    counts = np.asarray(jnp.einsum('spq,qp->s', density, overlap))
    idempotency = float(jnp.linalg.norm(density@overlap@density-density))
    valid = (bool(state['converged']) and np.isfinite(float(energy)) and residual <= 1e-7
             and idempotency <= 1e-7 and np.allclose(counts, [(nelectron+spin)/2, (nelectron-spin)/2], atol=1e-7, rtol=0))
    return valid, dict(energy=energy, fock=fock, orbital_residual=residual,
                      electron_numbers=counts.tolist(), idempotency_residual=idempotency)


def orbital_stability(state, hcore, factors, spin):
    """Use the project's eigensolver and exact real DF-HF Hessian actions."""
    from gradscf.solvers import LinearOperator, EigenSolverConfig, solve_hermitian
    from gradscf.scf.stability import _df_rhf_orbital_curvature
    if spin:
        occupations = tuple(map(tuple, np.asarray(state['occupations'])))
        gradient, apply, diagonal = df_curvature(hcore, factors, state['mo'], occupations)
    else:
        no = int(np.count_nonzero(np.asarray(state['occupations'])))
        gradient, apply = _df_rhf_orbital_curvature(hcore, factors, state['mo'], no)
        apply = jax.jit(apply)
        eps = jnp.diag(state['mo'].T@state['fock'][0]@state['mo'])
        diagonal = (4*(eps[no:][None, :]-eps[:no, None])).reshape(-1)
    dimension = len(gradient)
    norm = float(jnp.linalg.norm(gradient))
    if not dimension:
        return dict(stable=True, resolved=True, minimum_curvature=0., gradient_norm=norm,
                    eigen_residual=0., dimension=0, solver='empty', direction=None)
    bounded_matmat = lambda vectors: jax.lax.map(apply, vectors.T).T
    operator = LinearOperator((dimension, dimension), gradient.dtype, apply,
                              diagonal=diagonal, matmat=bounded_matmat)
    cfg = EigenSolverConfig(method='dense' if dimension <= 128 else 'davidson',
        nroots=min(3, dimension), atol=1e-7, maxiter=150,
        max_subspace=min(60, dimension), initial_guess_count=min(8, dimension))
    solved = solve_hermitian(operator, config=cfg)
    if not bool(jnp.all(solved.converged)) and dimension > 128:
        from dataclasses import replace
        cfg = replace(cfg, maxiter=300, max_subspace=min(80, dimension))
        solved = solve_hermitian(operator, config=cfg)
        if not bool(jnp.all(solved.converged)) and dimension <= 1024:
            cfg = replace(cfg, method='dense', max_dense=1024)
            solved = solve_hermitian(operator, config=cfg)
    resolved = bool(jnp.all(solved.converged)) and norm <= 1e-7
    minimum = float(solved.values[0])
    return dict(stable=resolved and minimum >= -1e-5, resolved=resolved,
                minimum_curvature=minimum, gradient_norm=norm,
                eigen_residual=float(jnp.max(solved.residual_norms)),
                dimension=dimension, solver=cfg.method,
                direction=solved.vectors[:, 0] if resolved and minimum < -1e-5 else None)


def solve_native(overlap, hcore, factors, nelectron, spin, enuc, *, row, atom_shells,
                 nuclear_charges=None, initial_density=None, check_stability=True, max_restarts=3):
    """Bounded native atomic-start/shifted rescue and negative-mode following.

    Shifted solutions never qualify as accepted energies. Every candidate is
    polished on the original problem and checked for density stationarity.
    PySCF supplies neither integrals nor initial densities to this workflow.
    """
    from gradscf.scf.orbital_optimization import _rotation_functions
    if not jax.config.x64_enabled:
        raise ValueError('Native HF retries require float64.')
    if nelectron < 1 or spin < 0 or spin > nelectron or (nelectron-spin) % 2:
        raise ValueError('Invalid electron number/spin for native HF retry.')
    if np.linalg.eigvalsh(overlap)[0] <= 1e-10:
        raise ValueError('Contracted AO overlap violates the original 1e-10 cutoff.')
    charges = row['atomic_numbers'] if nuclear_charges is None else nuclear_charges
    guess = minao_guess(row, atom_shells, overlap, charges, nelectron=nelectron, spin=spin)
    s, h, packed, nuc = map(jnp.asarray, (overlap, hcore, factors, enuc))
    b = unpack_factors(packed, len(s))
    attempts = []

    def evaluate(seed, shift, label):
        state = _solver(nelectron, spin, shift)(s, h, packed, nuc, jnp.asarray(seed))
        valid, physical = _physical(state, s, h, b, nuc, nelectron, spin)
        attempts.append(dict(stage=label, level_shift=shift, converged=bool(state['converged']),
            valid_raw=valid and shift == 0., energy=float(physical['energy']),
            orbital_residual=physical['orbital_residual'], cycles=int(state['cycles'])))
        state = dict(state, **physical)
        return state, valid and shift == 0.

    def converge(seed, label):
        state, valid = evaluate(seed, 0., label)
        for shift in (.2, .5):
            if valid:
                break
            shifted, _ = evaluate(seed, shift, label+':shift')
            state, valid = evaluate(shifted['density_spin'], 0., label+':raw-polish')
        return state, valid

    state, valid = converge(guess, 'native-minao')
    if initial_density is not None:
        candidate, candidate_valid = converge(initial_density, 'native-density-hint')
        if candidate_valid and (not valid or float(candidate['energy']) < float(state['energy'])):
            state, valid = candidate, True
    if not valid:
        raise RuntimeError('Native atomic-start HF failed: '+str(attempts))
    stability = dict(stable=None, resolved=False, minimum_curvature=None, gradient_norm=None,
                     eigen_residual=None, dimension=None, solver='not-requested', direction=None)
    restarts = 0
    if check_stability:
        while True:
            stability = orbital_stability(state, h, b, spin)
            if stability['stable']:
                break
            if not stability['resolved'] or restarts >= max_restarts:
                raise RuntimeError('Native internal stability unresolved: '+str({k:v for k,v in stability.items() if k != 'direction'}))
            if spin:
                occupation_rows = tuple(map(tuple, np.asarray(state['occupations'])))
                base = state['mo']
            else:
                occupation = tuple(np.asarray(state['occupations'])/2)
                occupation_rows, base = (occupation, occupation), state['mo'][None]
            _, rotate, density_from = _rotation_functions('uks' if spin else 'roks', occupation_rows)
            best, best_energy = None, float(state['energy'])
            for step in (.5, -.5, 1., -1.):
                seed = density_from(rotate(step*stability['direction'], base))
                trial, valid = converge(seed, 'negative-mode')
                if valid and float(trial['energy']) < best_energy-1e-10:
                    best, best_energy = trial, float(trial['energy'])
            if best is None:
                raise RuntimeError('No lower converged native state along the resolved negative mode.')
            state, restarts = best, restarts+1
    state = {key: np.asarray(value) if hasattr(value, 'shape') else value for key, value in state.items()}
    state.update(stability={k:v for k,v in stability.items() if k != 'direction'},
        stability_restarts=restarts, attempts=attempts, rescue_used=len(attempts)>1,
        total_scf_cycles=sum(r['cycles'] for r in attempts),
        initial_guess='native MINAO atomic projection; normalized spin populations')
    return state
