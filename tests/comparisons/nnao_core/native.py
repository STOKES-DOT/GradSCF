"""Thin native-integral HF/DFPT adapters for the comparison suite.

SCF, J/K and response iteration remain in GradSCF. Only packed three-center
factors are retained; no four-index ERI is requested or materialized here.
"""
from dataclasses import replace
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import integrals, scf
from gradscf.data.molecule import ANGSTROM_TO_BOHR
from gradscf.integrals.basis import BasisParameters, BasisTopology
from gradscf.integrals.basis.data import load_basis_from_snapshot
from gradscf.integrals.molecular.density_fitting import unpack_factors
from gradscf.integrals.molecular.jk import build_jk_from_df
from .protocols import counterpoise_energies, polarizability_from_response, subset_packed_factors


def traditional_shells(row, basis):
    result = []
    for symbol in row['symbols']:
        base = 'sto-6g' if basis.lower() == 'sto-6g+p' else basis
        shells = load_basis_from_snapshot(base, symbol)
        if basis.lower() == 'sto-6g+p':
            if symbol not in {'H', 'C', 'N', 'O'}:
                raise ValueError('This paper-matched polarization control is defined for CHNO only.')
            l = 1 if symbol == 'H' else 2
            polarization = [s for s in load_basis_from_snapshot('def2-svp', symbol) if s[0] == l]
            if len(polarization) != 1:
                raise ValueError('Expected one original def2-SVP polarization shell.')
            shells = shells+polarization
        result.append(shells)
    return result


def basis_parameters(row, atom_shells):
    if len(atom_shells) != len(row['symbols']):
        raise ValueError('One explicit shell list per atom is required.')
    coordinates = jnp.asarray(row['coords_angstrom'], dtype=jnp.float64)*ANGSTROM_TO_BOHR
    ls, exponents, coefficients, owners, ao_owners = [], [], [], [], []
    for atom, shells in enumerate(atom_shells):
        for block in shells:
            l, values = int(block[0]), jnp.asarray(block[1:], dtype=jnp.float64)
            ls.append(l)
            exponents.append(values[:, 0])
            coefficients.append(values[:, 1:])
            owners.append(atom)
            ao_owners.extend([atom]*((2*l+1)*(values.shape[1]-1)))
    topology = BasisTopology(tuple(ls), tuple(len(x) for x in exponents),
        tuple(x.shape[1] for x in coefficients), tuple(row['atomic_numbers']), False)
    parameters = BasisParameters(tuple(exponents), tuple(coefficients),
                                 coordinates[jnp.asarray(owners)], coordinates)
    return topology, parameters, np.asarray(ao_owners, dtype=int)


def make_inputs(row, atom_shells, *, auxbasis='def2-universal-jkfit'):
    if not jax.config.x64_enabled:
        raise ValueError('The core benchmark requires float64.')
    # Integral preparation is native CPU work; the solver may run on a GPU.
    with jax.default_device(jax.devices('cpu')[0]):
        topology, parameters, ao_owners = basis_parameters(row, atom_shells)
        plan = integrals.make_plan(topology)
        overlap = plan.evaluate('overlap', parameters)
        kinetic = plan.evaluate('kinetic', parameters)
        hcore = kinetic+plan.evaluate('nuclear', parameters)
        dipole = plan.evaluate('dipole', parameters, origin=jnp.zeros(3))
        at, ap = integrals.prepare_basis(list(zip(row['symbols'], row['coords_angstrom'])),
                                        auxbasis, cart=False)
        factors = integrals.make_auxiliary_plan(topology, at).factors(parameters, ap)
        enuc = scf.nuclear_repulsion_energy(parameters.nuclear_coords,
                                           jnp.asarray(topology.nuclear_charges))
    return dict(overlap=np.asarray(overlap), kinetic=np.asarray(kinetic), hcore=np.asarray(hcore),
                dipole=np.asarray(dipole), factors=np.asarray(factors), enuc=float(enuc),
                topology=topology, parameters=parameters, ao_owners=ao_owners,
                auxbasis=auxbasis)


@lru_cache(maxsize=16)
def _solver(nelectron, spin, level_shift):
    from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable
    from gradscf.scf.uks import UKSConfig, run_uks_from_integrals
    cfg = dict(xc_spec='hf', jk_backend='df', max_cycle=200, conv_tol=1e-12,
               conv_tol_density=1e-10, conv_tol_grad=1e-9, level_shift=level_shift,
               damping=.2 if level_shift else 0.)
    nalpha, nbeta = (nelectron+spin)//2, (nelectron-spin)//2

    def calculate(s, h, factors, enuc, guess):
        n = len(s)
        factors = unpack_factors(factors, n)
        common = dict(overlap=s, hcore=h, df_factors=factors, nuclear_repulsion=enuc,
                      ao=jnp.zeros((0, n)), ao_deriv1=jnp.zeros((4, 0, n)), grid_weights=jnp.zeros(0))
        if spin == 0:
            r = run_rks_from_integrals_traceable(**common, eri=None, nelectron=nelectron,
                init_density=None if guess is None else jnp.sum(guess, axis=0), config=RKSConfig(**cfg))
            return dict(energy=r.total_energy, converged=r.converged, cycles=r.cycles,
                density=r.density_matrix, density_spin=jnp.stack([r.density_matrix/2]*2),
                mo=r.mo_coeff, occupations=r.mo_occ, mo_energy=r.mo_energy,
                fock=jnp.stack([r.fock_matrix]*2))
        r = run_uks_from_integrals(**common, eri=jnp.zeros((0,)), nalpha=nalpha, nbeta=nbeta,
            init_density_alpha=None if guess is None else guess[0],
            init_density_beta=None if guess is None else guess[1], config=UKSConfig(**cfg))
        return dict(energy=r.total_energy, converged=r.converged, cycles=r.cycles,
            density=r.density_matrix_alpha+r.density_matrix_beta,
            density_spin=jnp.stack([r.density_matrix_alpha, r.density_matrix_beta]),
            mo=jnp.stack([r.mo_coeff_alpha, r.mo_coeff_beta]),
            occupations=jnp.stack([r.mo_occ_alpha, r.mo_occ_beta]),
            mo_energy=jnp.stack([r.mo_energy_alpha, r.mo_energy_beta]),
            fock=jnp.stack([r.fock_matrix_alpha, r.fock_matrix_beta]))
    # UKS's public entry validates occupations on the host; its scan remains
    # the existing traceable implementation. Do not add an incompatible JIT.
    return jax.jit(calculate) if spin == 0 else calculate


def solve_hf(overlap, hcore, factors, nelectron, spin, enuc):
    if nelectron < 1 or spin < 0 or spin > nelectron or (nelectron-spin) % 2:
        raise ValueError('Electron number and spin do not define valid occupations.')
    s, h, b, nuc = map(jnp.asarray, (overlap, hcore, factors, enuc))
    if np.linalg.eigvalsh(overlap)[0] <= 1e-10:
        raise ValueError('Contracted AO overlap is linearly dependent at the benchmark threshold.')
    state = _solver(nelectron, spin, 0.)(s, h, b, nuc, None)
    rescued = not bool(state['converged'])
    if rescued:
        shifted = _solver(nelectron, spin, .2)(s, h, b, nuc, state['density_spin'])
        state = _solver(nelectron, spin, 0.)(s, h, b, nuc, shifted['density_spin'])
    state = {key: np.asarray(value) for key, value in state.items()}
    residual = np.linalg.norm(state['fock']@state['density_spin']@overlap
                              -overlap@state['density_spin']@state['fock'])
    number = np.einsum('spq,qp->s', state['density_spin'], overlap)
    if not bool(state['converged']) or not np.isfinite(state['energy']) or residual > 1e-7:
        raise RuntimeError(f'HF convergence/residual check failed: {residual}.')
    np.testing.assert_allclose(number, [(nelectron+spin)/2, (nelectron-spin)/2], atol=1e-7, rtol=0)
    state.update(orbital_residual=float(residual), electron_numbers=number.tolist(), rescue_used=rescued)
    return state


@lru_cache(maxsize=16)
def _response(nelectron):
    from gradscf.dfpt.ks import ks_problem
    from gradscf.solvers import LinearSolverConfig
    config = LinearSolverConfig(rtol=1e-10, atol=1e-12, maxiter=300, restart=60)

    @jax.jit
    def calculate(s, h, factors, coefficients, potential):
        factors = unpack_factors(factors, len(s))
        inputs = dict(s=s, h=h, factors=factors)

        def fock(density, a):
            j, k = build_jk_from_df(a['factors'], density[0])
            return (a['h']+j-k/2)[None]

        def energy(density, a):
            j, k = build_jk_from_df(a['factors'], density[0])
            return jnp.sum(density[0]*(a['h']+j/2-k/4))

        problem = ks_problem(coefficients[None], (nelectron//2,), inputs, fock, energy,
            overlap=lambda a: a['s'][None], occupation=2., converged=True, tolerance=1e-7)
        tangent = dict(s=jnp.zeros_like(s), h=potential, factors=jnp.zeros_like(factors))
        response = problem.response(tangent=tangent, config=config)
        return response.density[0], response.residual_norm, response.converged
    return calculate


def static_polarizability(data, state, nelectron):
    if state['mo'].ndim != 2 or nelectron % 2:
        raise ValueError('This polarizability benchmark uses closed-shell RHF states.')
    s, h, b, mo = map(jnp.asarray, (data['overlap'], data['hcore'], data['factors'], state['mo']))
    densities, residuals, converged = [], [], []
    for axis in range(3):
        d1, residual, valid = _response(nelectron)(s, h, b, mo, jnp.asarray(data['dipole'][axis]))
        if not bool(valid):
            raise RuntimeError(f'Implicit field response failed on axis {axis}: residual={residual}.')
        densities.append(np.asarray(d1))
        residuals.append(float(residual))
        converged.append(bool(valid))
    tensor = polarizability_from_response(data['dipole'], densities)
    number_response = np.einsum('xpq,qp->x', densities, data['overlap'])
    np.testing.assert_allclose(number_response, 0., atol=1e-7, rtol=0)
    if np.linalg.eigvalsh(tensor)[0] < -1e-6:
        raise RuntimeError('Negative static polarizability: inspect the reference branch.')
    return dict(tensor_au=tensor.tolist(), isotropic_au=float(np.trace(tensor)/3),
                response_residuals=residuals, converged=converged,
                electron_number_response=number_response.tolist(),
                basis_response='fixed zero-field coefficients; MACE has no field input')


def counterpoise(row, atom_shells, groups, *, auxbasis='def2-universal-jkfit', solver=None):
    if len(groups) != 2 or sorted(groups[0]+groups[1]) != list(range(len(row['symbols']))):
        raise ValueError('Counterpoise requires a disjoint, exhaustive two-fragment partition.')
    data = make_inputs(row, atom_shells, auxbasis=auxbasis)
    charges = np.asarray(row['atomic_numbers'])
    diagnostics = []
    def evaluate(s, h, factors, nelectron, enuc, label, masked, shells, initial_density=None):
        if solver is None:
            return solve_hf(s, h, factors, nelectron, 0, enuc)
        state = solver(s, h, factors, nelectron, 0, enuc, row=row, atom_shells=shells,
                       nuclear_charges=masked, initial_density=initial_density)
        diagnostics.append(dict(label=label, nelectron=nelectron, nao=len(s),
            energy_hartree=float(state['energy']), orbital_residual=state['orbital_residual'],
            stability=state['stability'], attempts=state['attempts']))
        return state
    total = evaluate(data['overlap'], data['hcore'], data['factors'], int(charges.sum()),
                     data['enuc'], 'dimer', charges, atom_shells)
    own, ghost = [], []
    for atoms in groups:
        kept = np.zeros(len(charges), dtype=bool)
        kept[atoms] = True
        masked = np.where(kept, charges, 0)
        top = replace(data['topology'], nuclear_charges=tuple(map(int, masked)))
        with jax.default_device(jax.devices('cpu')[0]):
            h = data['kinetic']+np.asarray(integrals.make_plan(top).evaluate('nuclear', data['parameters']))
            nuc = float(scf.nuclear_repulsion_energy(data['parameters'].nuclear_coords, jnp.asarray(masked)))
        nelectron = int(masked.sum())
        indices = np.flatnonzero(kept[data['ao_owners']])
        factors = subset_packed_factors(data['factors'], indices, nao=len(h))
        own_shells = [shells if active else [] for shells, active in zip(atom_shells, kept)]
        a = evaluate(data['overlap'][np.ix_(indices, indices)], h[np.ix_(indices, indices)],
                     factors, nelectron, nuc, 'own/'+str(len(own)), masked, own_shells)
        embedded = None
        if solver is not None:
            embedded = np.zeros((2, len(h), len(h)))
            for spin in range(2):
                embedded[spin][np.ix_(indices, indices)] = a['density_spin'][spin]
        g = evaluate(data['overlap'], h, data['factors'], nelectron, nuc,
                     'ghost/'+str(len(ghost)), masked, atom_shells, embedded)
        own.append(float(a['energy']))
        ghost.append(float(g['energy']))
    result = dict(counterpoise_energies(float(total['energy']), *own, *ghost),
                dimer_energy_hartree=float(total['energy']), own_energies_hartree=own,
                ghost_energies_hartree=ghost, nao=len(data['overlap']),
                orbital_basis_policy='dimer coefficients frozen in all five calculations',
                auxiliary_policy='one fixed dimer auxiliary basis and metric; fragment factors are AO slices')
    if solver is not None:
        result['scf_diagnostics'] = diagnostics
    return result
