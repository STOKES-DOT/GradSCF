"""S1-S8 VG/HT derivatives from one shared full-TDDFT geometry linearization.

Set GRADSCF_ANTHRACENE_INPUT to input_eight_states.json and
GRADSCF_ANTHRACENE_OUTPUT to .../ad_eight_states/raw. The immutable ground
geometry and 66 modes come from BASE_OUTPUT. Previously converged, signed
displacements under .../eight_states/raw validate four mode directions;
they never supply a derivative used in the spectrum.
"""
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from common import (BASE_OUTPUT, OUTPUT, ROOT, SETTINGS, SETTINGS_PATH, SYMBOLS,
                    event, mean_field, save)
from gradscf import tdscf
from gradscf.integrals._native import register_integrals
from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule
from gradscf_tools.excited import match_transition_states
from gradscf_tools.molecular_ep import NativeRHF
from gradscf_tools.molecular_response import MolecularEnergy

METHOD = 'AD of native geometry, implicit SCF and structured full-TDDFT eigenvectors'
FD_OUTPUT = Path(os.environ.get('GRADSCF_ANTHRACENE_FD_OUTPUT',
                                BASE_OUTPUT.parent/'eight_states/raw'))


def checkpoint(path, **arrays):
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moving_response(model, coordinates, coefficients, count):
    """Keep the metric, SCF, Fock blocks, quadrature and X/Y response live."""
    ground, inputs = model.solve_with_inputs(coordinates, solution=coefficients)
    moving = replace(model.parameters, centers=coordinates[model.owners],
                     nuclear_coords=coordinates)
    dipole = model.plan.evaluate('dipole', moving, origin=jnp.zeros(3))
    fock = jnp.mean(ground.fock_matrix, axis=0)
    energies = jnp.diag(ground.mo_coeff.T@fock@ground.mo_coeff)
    empty = jnp.zeros((0, 0, 0, 0), dtype=coordinates.dtype)
    reference = RestrictedMolecule(
        ao=inputs['ao'], ao_deriv1=inputs['ao_deriv1'],
        grid_ao=inputs.get('grid_ao'), grid=QuadratureGrid(inputs['grid_weights']),
        dipole_integrals=dipole, rep_tensor=empty,
        mo_coeff=jnp.stack((ground.mo_coeff, ground.mo_coeff)), mo_occ=ground.mo_occ,
        mo_energy=jnp.stack((energies, energies)), rdm1=ground.density_matrix,
        h1e=inputs['hcore'], overlap_matrix=inputs['overlap'], fock_matrix=fock,
        nuclear_repulsion=inputs['nuclear_repulsion'], atom_coords=coordinates,
        direct_basis=inputs['direct_basis'], mf_energy=ground.total_energy,
        nocc=int(np.count_nonzero(model.occupations[0])), xc_spec=model.xc,
        exact_exchange_fraction=float(model.xc == 'hf'), max_memory=SETTINGS['max_memory_mb'])
    driver = tdscf.TDDFT(reference, nstates=count,
        davidson_tol=SETTINGS['td_tol'], davidson_max_iter=200,
        tda_gradient_mode='implicit_eigenvector', eigenvector_adjoint_tol=1e-9,
        eigenvector_adjoint_max_iter=200,
        response_kernel_options=dict(max_memory=SETTINGS['max_memory_mb'],
            direct_workers=SETTINGS['direct_workers'], direct_tol=0.,
            integral_cache='none', grid_block_size=2048))
    result = driver.kernel()
    values = jnp.concatenate((result.excitation_energies, driver.transition_dipole().reshape(-1)))
    valid = ground.stationary & jnp.all(result.converged)
    return jnp.where(valid, values, jnp.nan), dict(valid=valid,
        mo_coeff=ground.mo_coeff, x=result.x_amplitudes, y=result.y_amplitudes,
        density=jnp.sum(ground.density_matrix, axis=0), ground_energy=ground.total_energy,
        gradient_norm=ground.gradient_norm)


def validate_directions(coordinates, modes, mass_ratio, gradients, slopes, signature):
    """Check four existing signed, independently re-converged FD directions."""
    step = SETTINGS['mode_step_bohr_sqrt_amu']
    pairs = [(mode, [FD_OUTPUT/'mode_points'/f'mode_{mode:03d}_{sign:+d}_{step:.5f}.npz'
                     for sign in (1, -1)]) for mode in range(modes.shape[-1])]
    pairs = [(mode, paths) for mode, paths in pairs if all(path.exists() for path in paths)]
    if len(pairs) < 4:
        raise RuntimeError('At least four completed signed FD mode pairs are required for AD validation.')
    # Include a low-index mode and select distinct strong gap/dipole directions
    # among the already completed pairs. No new displaced jobs are launched.
    available = [mode for mode, _ in pairs]
    gap_order = sorted(available, key=lambda mode: np.max(abs(gradients[:, mode])), reverse=True)
    mu_order = sorted(available, key=lambda mode: np.max(abs(slopes[:, :, mode])), reverse=True)
    selected = list(dict.fromkeys([available[0], gap_order[0], mu_order[0], *mu_order, *gap_order]))[:4]
    checks = []
    for mode in selected:
        paths = dict(pairs)[mode]
        observed = []
        for sign, path in zip((1, -1), paths):
            with np.load(path) as data:
                if str(data['signature']) != signature:
                    raise RuntimeError(f'The FD validation point has a different anchor: {path.name}')
                np.testing.assert_array_equal(data['coordinates'],
                    coordinates+sign*step*np.sqrt(mass_ratio)*modes[:, :, mode])
                observed.append((data['gap'].copy(), data['dipole'].copy()))
        divisor = 2*step*np.sqrt(mass_ratio)
        gap = (observed[0][0]-observed[1][0])/divisor
        mu = (observed[0][1]-observed[1][1])/divisor
        np.testing.assert_allclose(gradients[:, mode], gap, atol=3e-8, rtol=3e-3)
        np.testing.assert_allclose(slopes[:, :, mode], mu, atol=3e-7, rtol=3e-3)
        check = dict(mode=mode+1, step_bohr_sqrt_amu=step,
            gap_error=np.abs(gap-gradients[:, mode]).tolist(),
            dipole_error=np.max(abs(mu-slopes[:, :, mode]), axis=1).tolist(),
            input_sha256={path.name: file_hash(path) for path in paths})
        checks.append(check)
        event('ad_fd_direction_checked', **check)
    return checks


def main():
    started = perf_counter()
    # Native integral/J-K geometry primitives currently lower on CPU. The one
    # numerical SCF seed uses common.mean_field's explicit GPU execution.
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    count, fd_candidates = SETTINGS['ht_states'], SETTINGS['tracking_roots']
    if (count != 8 or fd_candidates < count+2 or OUTPUT.resolve() in
            (BASE_OUTPUT.resolve(), FD_OUTPUT.resolve())):
        raise ValueError('Use the eight-state input and a separate ad_eight_states/raw output.')
    save('mode_response_status.json', dict(complete=False, stage='starting',
        derivative_method=METHOD, state_count=count, cartesian_component_count=4*count))
    harmonic = json.loads((BASE_OUTPUT/'harmonics_status.json').read_text())
    if not harmonic.get('complete'):
        raise RuntimeError('Validated ground-state normal modes are required.')
    for key in ('atom', 'unit', 'charge', 'spin', 'basis', 'cart', 'xc', 'grid_level'):
        if harmonic['identity']['settings'][key] != SETTINGS[key]:
            raise ValueError(f'The reused harmonic data uses a different {key}.')
    with np.load(BASE_OUTPUT/'harmonics.npz') as data:
        coordinates, modes = data['coordinates'].copy(), data['modes_bohr_per_Q'].copy()
        omega = data['frequencies_cm1'].copy()/219474.6313631965
        mass_ratio = float(data['amu_to_electron_mass'])
    with np.load(BASE_OUTPUT/'ground.npz') as data:
        np.testing.assert_array_equal(coordinates, data['coordinates'])
        density, ground_energy = data['density'].copy(), float(data['energy'])
    if modes.shape != (*coordinates.shape, 66) or np.any(omega <= 0):
        raise RuntimeError('All 66 validated positive normal modes must be retained.')
    anchor_path = FD_OUTPUT/'vertical.npz'
    with np.load(anchor_path) as data:
        anchor = {name: data[name].copy() for name in data.files}
    np.testing.assert_array_equal(coordinates, anchor['coordinates'])
    if anchor['excitation'].shape != (fd_candidates,):
        raise RuntimeError('The independent signed FD anchor has a different candidate space.')
    native = Path(register_integrals()._name).resolve()
    if file_hash(native) != harmonic['identity']['native_sha256']:
        raise RuntimeError('The reused harmonic data uses a different native library.')
    source = hashlib.sha256()
    for path in sorted((ROOT/'src').rglob('*.py')):
        source.update(str(path.relative_to(ROOT)).encode())
        source.update(path.read_bytes())
    files = [BASE_OUTPUT/'ground.npz', BASE_OUTPUT/'harmonics.npz', anchor_path,
             Path(__file__), Path(__file__).with_name('common.py'), SETTINGS_PATH]
    identity = dict(settings={**SETTINGS, 'ht_derivative': METHOD},
        source_sha256=source.hexdigest(), native_sha256=file_hash(native),
        input_sha256={str(path.resolve()): file_hash(path) for path in files},
        harmonic_source_sha256=harmonic['identity']['source_sha256'],
        execution_device=str(jax.devices('cpu')[0]), jax_version=jax.__version__,
        dtype='float64', all_occupied_and_virtual_orbitals=True,
        ad_requested_states=count, shared_rpa_guard_states=1, fd_anchor_states=fd_candidates,
        response_gradient_mode='implicit_eigenvector', eigenvector_adjoint_tol=1e-9,
        eigenvector_adjoint_max_iter=200)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    components = OUTPUT/'cartesian_components'
    components.mkdir(exist_ok=True)
    mf = mean_field(coordinates, density)
    np.testing.assert_allclose(mf.e_tot, ground_energy, atol=1e-8, rtol=0.)
    model = MolecularEnergy(mf, forward='reference', jk_backend='direct',
        gradient_tolerance=max(1e-8, 10*SETTINGS['scf_gradient_tol']),
        max_memory=SETTINGS['max_memory_mb'])
    del mf
    gc.collect()
    # The archived C is a converged numerical origin. solve_with_inputs attaches
    # full moving-metric and implicit orbital response to it at R0.
    r, c = jnp.asarray(coordinates), jnp.asarray(anchor['mo_coeff'])
    evaluate = jax.jit(lambda geometry, solution:
                       moving_response(model, geometry, solution, count))
    event('ad_shared_vjp_start', states=count, shared_rpa_guard_states=1,
          fd_anchor_states=fd_candidates, pullbacks=4*count)
    forward_started = perf_counter()
    values, pullback, auxiliary = jax.vjp(evaluate, r, c, has_aux=True)
    jax.block_until_ready((values, auxiliary))
    values = np.asarray(values)
    auxiliary = jax.tree.map(np.asarray, auxiliary)
    if not bool(auxiliary['valid']) or not np.isfinite(values).all():
        raise RuntimeError('The stationary ground state or structured full TDDFT failed.')
    event('ad_shared_vjp_forward', seconds=perf_counter()-forward_started,
        ground_gradient_norm=float(auxiliary['gradient_norm']))
    engine = NativeRHF(list(zip(SYMBOLS, coordinates)), SETTINGS['basis'], unit='Bohr')
    nocc = int(np.count_nonzero(anchor['mo_occ']))
    matched = match_transition_states(anchor['mo_coeff'], anchor['x'][:count]+anchor['y'][:count],
        auxiliary['mo_coeff'], auxiliary['x']+auxiliary['y'],
        engine.cross_overlap(coordinates, coordinates), values[:count], nocc=nocc)
    roots, phases = matched['roots'], matched['phases']
    if not np.array_equal(np.sort(roots), np.arange(count)):
        raise RuntimeError('The AD roots must be a permutation of the eight physical targets.')
    # The shared RPA solve adds its own ninth guard root and checks the gap to
    # that external state. Only the eight physical targets carry vector AD.
    selected = np.r_[roots, [count+3*root+axis for root in roots for axis in range(3)]]
    signs = np.r_[np.ones(count), np.repeat(phases, 3)]
    output = values[selected]*signs
    np.testing.assert_allclose(output[:count], anchor['excitation'][:count], atol=3e-8, rtol=0.)
    np.testing.assert_allclose(output[count:].reshape(count, 3), anchor['dipole'][:count], atol=2e-6, rtol=1e-6)
    moments = values[count:].reshape(count, 3)[roots]*phases[:, None]
    excitation = values[:count][roots]
    checkpoint(OUTPUT/'vertical.npz', signature=signature, coordinates=coordinates,
        excitation=excitation, dipole=moments,
        oscillator_strength=(2./3)*excitation*np.sum(moments**2, axis=1),
        x=auxiliary['x'][roots]*phases[:, None, None],
        y=auxiliary['y'][roots]*phases[:, None, None],
        mo_coeff=auxiliary['mo_coeff'], mo_occ=anchor['mo_occ'], density=auxiliary['density'],
        target_indices=np.arange(count), source_candidate_roots=roots,
        phases=phases, target_pair_gaps=matched['gaps_hartree'], shared_rpa_guard_states=1)
    # One retained linearization, one compiled pullback, 32 scalar adjoints.
    # Discarding the seed cotangent is valid: C supplies the numerical origin,
    # while the attached stationarity response supplies dC/dR.
    reverse = jax.jit(lambda weight: pullback(weight)[0])
    rows, timings = [], []
    for index, (component, sign) in enumerate(zip(selected, signs)):
        path = components/f'component_{index:02d}.npz'
        if path.exists():
            with np.load(path) as data:
                if str(data['signature']) != signature or int(data['component']) != component:
                    raise RuntimeError(f'Stale AD component checkpoint: {path.name}')
                gradient, seconds = data['gradient'].copy(), float(data['seconds'])
        else:
            row_started = perf_counter()
            weight = jnp.zeros(4*count).at[component].set(sign)
            gradient = np.asarray(reverse(weight))
            seconds = perf_counter()-row_started
            if gradient.shape != coordinates.shape or not np.isfinite(gradient).all():
                raise RuntimeError(f'The full geometry pullback is invalid for component {index}.')
            checkpoint(path, signature=signature, component=component, sign=sign,
                       gradient=gradient, seconds=seconds)
            event('ad_component_saved', component=index+1, components=4*count, seconds=seconds)
        rows.append(gradient)
        timings.append(seconds)
        save('mode_response_status.json', dict(complete=False, stage='cartesian_components',
            signature=signature, identity=identity, derivative_method=METHOD,
            state_count=count, completed_components=index+1, cartesian_component_count=4*count))
    jacobian = np.asarray(rows)
    checkpoint(OUTPUT/'cartesian_jacobian.npz', signature=signature,
        coordinates=coordinates, values=output, jacobian=jacobian, roots=roots, phases=phases)
    translation = np.max(abs(jacobian.sum(axis=1)), axis=1)
    if np.max(translation) > 1e-5:
        raise RuntimeError('The complete AD geometry Jacobian violates translation invariance.')
    save('translation_check.json', dict(complete=True, tolerance=1e-5,
        gap_max_error=float(max(translation[:count])),
        dipole_max_error=float(max(translation[count:]))))
    gradients = np.einsum('sac,ack->sk', jacobian[:count], modes)
    slopes = np.einsum('sxac,ack->sxk', jacobian[count:].reshape(count, 3, *coordinates.shape), modes)
    fd_signature = hashlib.sha256((str(anchor['signature'])+file_hash(anchor_path)).encode()).hexdigest()
    checks = validate_directions(coordinates, modes, mass_ratio, gradients, slopes, fd_signature)
    checkpoint(OUTPUT/'mode_response.npz', signature=signature, frequencies=omega,
        gap_gradient=gradients, dipole=output[count:].reshape(count, 3),
        dipole_derivative=slopes, vertical_energy=output[:count])
    save('mode_response_status.json', dict(complete=True, signature=signature, identity=identity,
        derivative_method=METHOD, target_roots=list(range(1, count+1)),
        source_candidate_roots=(roots+1).tolist(), state_count=count, mode_count=len(omega),
        cartesian_component_count=4*count, component_seconds=timings,
        elapsed_seconds=perf_counter()-started, finite_difference_validation=checks,
        mode_coordinate_units='bohr sqrt(electron mass)',
        gap_cartesian_units='Hartree/bohr', dipole_cartesian_units='electron charge',
        harmonic_model='0 K independent equal-curvature VG',
        state_phase='Constant signed AO transition-density match to independent FD vertical anchor',
        ground_hessian_recomputed=False, displaced_scf_used_for_derivatives=False))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save('mode_response_failure.json', dict(complete=False, derivative_method=METHOD,
             error=traceback.format_exc()))
        raise
