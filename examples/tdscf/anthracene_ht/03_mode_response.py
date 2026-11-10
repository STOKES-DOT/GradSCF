"""Fully re-converged full-TDDFT finite differences for the isolated S1 band.

The X/Y amplitude response is included by solving at every displaced geometry.
Normal-coordinate units are bohr*sqrt(electron mass); no frozen-orbital slope.
The exact saved vertical state fixes the phase anchor for all resumed points.
"""
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter
import traceback

import jax
import numpy as np

from common import OUTPUT, ROOT, SETTINGS, SYMBOLS, event, mean_field, response, save
from gradscf.integrals._native import register_integrals
from gradscf_tools.excited import transition_density_overlaps, select_tracked_root
from gradscf_tools.molecular_ep import NativeRHF


def checkpoint(path, **arrays):
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def main():
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    save('mode_response_status.json', dict(complete=False, stage='starting'))
    if not json.loads((OUTPUT/'harmonics_status.json').read_text()).get('complete'):
        raise RuntimeError('Validated ground-state normal modes are required.')
    with np.load(OUTPUT/'harmonics.npz') as data:
        coordinates, modes = data['coordinates'].copy(), data['modes_bohr_per_Q'].copy()
        omega = data['frequencies_cm1'].copy()/219474.6313631965
        mass_ratio = float(data['amu_to_electron_mass'])
    if (modes.shape[:2] != coordinates.shape or omega.shape != (modes.shape[-1],)
            or not np.isfinite(modes).all() or not np.isfinite(omega).all() or np.any(omega <= 0)):
        raise RuntimeError('Finite positive modes at the declared geometry are required.')
    with np.load(OUTPUT/'ground.npz') as data:
        np.testing.assert_array_equal(data['coordinates'], coordinates)
        density = data['density'].copy()
    source = hashlib.sha256()
    for path in sorted((ROOT/'src').rglob('*.py')):
        source.update(str(path.relative_to(ROOT)).encode()); source.update(path.read_bytes())
    native = Path(register_integrals()._name).resolve()
    identity = dict(settings=SETTINGS, source_sha256=source.hexdigest(),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        common_sha256=hashlib.sha256(Path(__file__).with_name('common.py').read_bytes()).hexdigest(),
        harmonics_sha256=hashlib.sha256((OUTPUT/'harmonics.npz').read_bytes()).hexdigest(),
        ground_sha256=hashlib.sha256((OUTPUT/'ground.npz').read_bytes()).hexdigest())
    input_signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    vertical_path = OUTPUT/'vertical.npz'
    points = OUTPUT/'mode_points'
    points.mkdir(exist_ok=True)
    if vertical_path.exists():
        with np.load(vertical_path) as data:
            if 'signature' not in data or str(data['signature']) != input_signature:
                raise RuntimeError('The saved vertical anchor uses different inputs or source.')
            baseline = {name: data[name].copy() for name in data.files}
        event('vertical_anchor_reused', signature=input_signature)
    else:
        if any(points.glob('*.npz')):
            raise RuntimeError('Mode checkpoints exist without their exact vertical phase anchor.')
        mf = mean_field(coordinates, density)
        td, mu, strengths = response(mf)
        x0, y0 = (np.asarray(value) for value in td.xy)
        baseline = dict(coordinates=coordinates, excitation=np.asarray(td.e), dipole=mu,
            oscillator_strength=strengths, x=x0, y=y0, mo_coeff=np.asarray(mf.mo_coeff),
            mo_occ=np.asarray(mf.mo_occ), density=np.asarray(mf.make_rdm1()))
        checkpoint(vertical_path, signature=input_signature, **baseline)
        del td, mf, x0, y0
        gc.collect()
    np.testing.assert_array_equal(baseline['coordinates'], coordinates)
    if any(not np.isfinite(baseline[name]).all() for name in
           ('excitation', 'dipole', 'x', 'y', 'mo_coeff', 'density')):
        raise RuntimeError('The vertical electronic anchor must be finite.')
    nocc = int(np.count_nonzero(baseline['mo_occ']))
    root = SETTINGS['target_root']
    anchor_c = baseline['mo_coeff']
    anchor_amplitude = baseline['x'][root]+baseline['y'][root]
    gap, dipole = float(baseline['excitation'][root]), baseline['dipole'][root]
    density = baseline['density']
    if gap <= 0 or np.min(abs(np.delete(baseline['excitation'], root)-gap)) < 1e-4:
        raise RuntimeError('The target state is not an isolated positive excitation.')
    # Tie every signed point to the exact cached C, X/Y and dipole gauge.
    anchor_hash = hashlib.sha256(vertical_path.read_bytes()).hexdigest()
    signature = hashlib.sha256((input_signature+anchor_hash).encode()).hexdigest()
    engine = NativeRHF(list(zip(SYMBOLS, coordinates)), SETTINGS['basis'], unit='Bohr')
    if engine.topology.nao != anchor_c.shape[0]:
        raise RuntimeError('Cross-overlap AO convention differs from the TD reference.')
    save('mode_response_status.json', dict(complete=False, stage='mode_points',
        signature=signature, input_signature=input_signature, anchor_sha256=anchor_hash,
        identity=identity, completed_modes=0))

    def point(mode, sign, step):
        path = points/f'mode_{mode:03d}_{sign:+d}_{step:.5f}.npz'
        moving = coordinates+sign*step*np.sqrt(mass_ratio)*modes[:, :, mode]
        if path.exists():
            with np.load(path) as old:
                if str(old['signature']) != signature:
                    raise RuntimeError(f'Stale mode checkpoint: {path}')
                np.testing.assert_array_equal(old['coordinates'], moving)
                return float(old['gap']), old['dipole'].copy()
        started = perf_counter()
        state = mean_field(moving, density)
        result, moments, _ = response(state, nstates=SETTINGS['tracking_roots'])
        amplitudes = sum(np.asarray(value) for value in result.xy)
        cross = np.asarray(engine.cross_overlap(coordinates, moving))
        overlaps = transition_density_overlaps(anchor_c, anchor_amplitude,
            state.mo_coeff, amplitudes, cross, nocc=nocc)
        match = select_tracked_root(overlaps, np.asarray(result.e), minimum_overlap=.90)
        selected = match['root']
        occupied = anchor_c[:, :nocc].T@cross@np.asarray(state.mo_coeff)[:, :nocc]
        virtual = anchor_c[:, nocc:].T@cross@np.asarray(state.mo_coeff)[:, nocc:]
        projected = occupied@amplitudes[selected]@virtual.T
        phase = 1. if np.sum(anchor_amplitude*projected) >= 0 else -1.
        moment, value = phase*moments[selected], float(result.e[selected])
        if value <= 0 or not np.isfinite(moment).all():
            raise RuntimeError('The tracked transition has a nonphysical energy or dipole.')
        checkpoint(path, signature=signature, coordinates=moving, gap=value,
            dipole=moment, root=selected, overlap=match['overlap'], phase=phase,
            state_gap_hartree=match['gap_hartree'], runner_up_overlap=match['runner_up_overlap'])
        event('mode_point_saved', mode=mode+1, sign=sign, step=step,
            seconds=perf_counter()-started, overlap=match['overlap'], root=selected+1)
        del state, result
        gc.collect()
        return value, moment

    def derivative(mode, step):
        plus, minus = point(mode, 1, step), point(mode, -1, step)
        divisor = 2*step*np.sqrt(mass_ratio)
        return (plus[0]-minus[0])/divisor, (plus[1]-minus[1])/divisor

    gradients, slopes = [], []
    for mode in range(modes.shape[-1]):
        gradient, slope = derivative(mode, SETTINGS['mode_step_bohr_sqrt_amu'])
        gradients.append(gradient); slopes.append(slope)
        save('mode_response_status.json', dict(complete=False, stage='mode_points',
             signature=signature, input_signature=input_signature, anchor_sha256=anchor_hash,
             identity=identity, completed_modes=mode+1, mode_count=modes.shape[-1]))
    gradients, slopes = np.array(gradients), np.array(slopes).T
    huang_rhys = gradients**2/(2*omega**3)
    ht_strength = np.sum(slopes**2, axis=0)/(2*omega)
    selected_checks = sorted(set(np.argsort(huang_rhys)[-3:]) | set(np.argsort(ht_strength)[-3:]))
    checks = []
    for mode in selected_checks:
        small_g, small_mu = derivative(int(mode), SETTINGS['mode_check_step_bohr_sqrt_amu'])
        np.testing.assert_allclose(small_g, gradients[mode], atol=3e-8, rtol=3e-3)
        np.testing.assert_allclose(small_mu, slopes[:, mode], atol=3e-7, rtol=3e-3)
        checks.append(dict(mode=int(mode)+1, gap_error=float(abs(small_g-gradients[mode])),
            dipole_error=float(np.max(abs(small_mu-slopes[:, mode])))))
    checkpoint(OUTPUT/'mode_response.npz', frequencies=omega, gap_gradient=gradients,
        dipole=dipole, dipole_derivative=slopes, vertical_energy=gap, signature=signature)
    save('mode_response_status.json', dict(complete=True, signature=signature,
        input_signature=input_signature, anchor_sha256=anchor_hash, identity=identity,
        target_root=root+1, step_checks=checks, mode_count=modes.shape[-1],
        derivative_method='Full displaced SCF/full TDDFT central differences',
        mode_coordinate_units='bohr sqrt(electron mass)', harmonic_model='VG same curvature, 0 K'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save('mode_response_failure.json', dict(complete=False, error=traceback.format_exc()))
        raise
