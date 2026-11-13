"""S1-S8 VG/HT derivatives from shared, fully re-converged displaced solves.

Eight isolated real states are matched jointly against twelve candidate roots.
Ground geometry/modes are immutable inputs; no ground Hessian is recalculated.
"""
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter
import traceback

import jax
import numpy as np

from common import BASE_OUTPUT, OUTPUT, ROOT, SETTINGS, event, mean_field, response, save
from gradscf.integrals._native import register_integrals
from gradscf_tools.excited import match_transition_states
from gradscf_tools.molecular_ep import NativeRHF


def checkpoint(path, **arrays):
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def main():
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    count, candidates = SETTINGS['ht_states'], SETTINGS['tracking_roots']
    if count != 8 or candidates < count+2 or OUTPUT.resolve() == BASE_OUTPUT.resolve():
        raise ValueError('Use input_eight_states.json and a separate eight-state output directory.')
    save('mode_response_status.json',dict(complete=False,stage='starting',state_count=count))
    harmonic = json.loads((BASE_OUTPUT/'harmonics_status.json').read_text())
    if not harmonic.get('complete'):
        raise RuntimeError('Validated ground-state normal modes are required.')
    # Root-count changes do not change the immutable ground-state physics.
    for key in ('atom','unit','charge','spin','basis','cart','xc','grid_level'):
        if harmonic['identity']['settings'][key] != SETTINGS[key]:
            raise ValueError(f'The reused harmonic data uses a different {key}.')
    with np.load(BASE_OUTPUT/'harmonics.npz') as data:
        coordinates,modes=data['coordinates'].copy(),data['modes_bohr_per_Q'].copy()
        omega=data['frequencies_cm1'].copy()/219474.6313631965
        mass_ratio=float(data['amu_to_electron_mass'])
    with np.load(BASE_OUTPUT/'ground.npz') as data:
        np.testing.assert_array_equal(coordinates,data['coordinates'])
        density=data['density'].copy()
    if modes.shape != (*coordinates.shape,66) or np.any(omega <= 0):
        raise RuntimeError('All 66 validated positive normal modes must be retained.')
    symbols=[row.split()[0] for row in SETTINGS['atom'].split(';')]
    engine=NativeRHF(list(zip(symbols,coordinates)),SETTINGS['basis'],unit='Bohr')
    native=Path(register_integrals()._name).resolve()
    source=hashlib.sha256()
    for path in sorted((ROOT/'src').rglob('*.py')):
        source.update(str(path.relative_to(ROOT)).encode());source.update(path.read_bytes())
    files=[BASE_OUTPUT/'ground.npz',BASE_OUTPUT/'harmonics.npz',Path(__file__),
           Path(__file__).with_name('common.py')]
    legacy=BASE_OUTPUT/'vertical.npz'
    if legacy.exists():files.append(legacy)
    identity=dict(settings=SETTINGS,source_sha256=source.hexdigest(),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        harmonic_source_sha256=harmonic['identity']['source_sha256'])
    if identity['native_sha256'] != harmonic['identity']['native_sha256']:
        raise RuntimeError('The reused harmonic data uses a different native library.')
    input_signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    vertical=OUTPUT/'vertical.npz';points=OUTPUT/'mode_points';points.mkdir(exist_ok=True)
    if vertical.exists():
        with np.load(vertical) as data:
            if str(data['signature']) != input_signature:
                raise RuntimeError('The vertical anchor uses different inputs or source.')
            baseline={key:data[key].copy() for key in data.files}
    else:
        if any(points.glob('*.npz')):
            raise RuntimeError('Displaced points cannot be used without their exact phase anchor.')
        mf=mean_field(coordinates,density)
        td,moments,strengths=response(mf,nstates=candidates)
        x,y=(np.array(v) for v in td.xy);moments=np.array(moments)
        energies,coeff=np.asarray(td.e),np.asarray(mf.mo_coeff)
        if energies.shape != (candidates,) or np.any(energies <= 0) or not np.isfinite(moments).all():
            raise RuntimeError('The requested candidate excitation energies/dipoles are invalid.')
        nocc=int(np.count_nonzero(np.asarray(mf.mo_occ)))
        if legacy.exists():
            with np.load(legacy) as old:
                np.testing.assert_array_equal(old['coordinates'],coordinates)
                match=match_transition_states(old['mo_coeff'],old['x'][:count]+old['y'][:count],
                    coeff,x+y,engine.cross_overlap(coordinates,coordinates),energies,nocc=nocc)
                if not np.array_equal(match['roots'],np.arange(count)):
                    raise RuntimeError('The new lowest-eight anchor differs from the previous vertical states.')
                x[:count]*=match['phases'][:,None,None]
                y[:count]*=match['phases'][:,None,None]
                moments[:count]*=match['phases'][:,None]
        self_match=match_transition_states(coeff,x[:count]+y[:count],coeff,x+y,
            engine.cross_overlap(coordinates,coordinates),energies,nocc=nocc)
        checkpoint(vertical,signature=input_signature,coordinates=coordinates,
            excitation=energies,dipole=moments,oscillator_strength=strengths,x=x,y=y,
            mo_coeff=coeff,mo_occ=np.asarray(mf.mo_occ),density=np.asarray(mf.make_rdm1()),
            target_indices=np.arange(count),guard_gaps=self_match['gaps_hartree'])
        baseline=dict(excitation=energies,dipole=moments,x=x,y=y,mo_coeff=coeff,
                      mo_occ=np.asarray(mf.mo_occ),density=np.asarray(mf.make_rdm1()))
        del mf,td;gc.collect()
    anchor_c=baseline['mo_coeff'];anchor_a=baseline['x'][:count]+baseline['y'][:count]
    nocc=int(np.count_nonzero(baseline['mo_occ']));density=baseline['density']
    signature=hashlib.sha256((input_signature+hashlib.sha256(vertical.read_bytes()).hexdigest()).encode()).hexdigest()

    def point(mode,sign,step):
        path=points/f'mode_{mode:03d}_{sign:+d}_{step:.5f}.npz'
        moving=coordinates+sign*step*np.sqrt(mass_ratio)*modes[:,:,mode]
        if path.exists():
            with np.load(path) as data:
                if str(data['signature']) != signature:raise RuntimeError(f'Stale checkpoint: {path}')
                np.testing.assert_array_equal(data['coordinates'],moving)
                return data['gap'].copy(),data['dipole'].copy()
        started=perf_counter();mf=mean_field(moving,density)
        td,moments,_=response(mf,nstates=candidates)
        x,y=(np.asarray(v) for v in td.xy)
        match=match_transition_states(anchor_c,anchor_a,mf.mo_coeff,x+y,
            engine.cross_overlap(coordinates,moving),np.asarray(td.e),nocc=nocc)
        roots,phases=match['roots'],match['phases']
        if max(roots) == candidates-1:
            raise RuntimeError('A target reached the highest candidate; additional guard roots are required.')
        gap=np.asarray(td.e)[roots];dipole=moments[roots]*phases[:,None]
        if gap.shape != (count,) or np.any(gap <= 0) or not np.isfinite(dipole).all():
            raise RuntimeError('The matched target energies/dipoles are invalid.')
        old_path=BASE_OUTPUT/'mode_points'/path.name
        legacy_error=None
        if old_path.exists():
            with np.load(old_path) as old:
                np.testing.assert_array_equal(old['coordinates'],moving)
                np.testing.assert_allclose(gap[0],old['gap'],atol=3e-8,rtol=0.)
                np.testing.assert_allclose(dipole[0],old['dipole'],atol=2e-6,rtol=1e-6)
                legacy_error=float(np.max(abs(dipole[0]-old['dipole'])))
        checkpoint(path,signature=signature,coordinates=moving,gap=gap,dipole=dipole,
            roots=roots,phases=phases,overlaps=match['matched_overlap'],margins=match['margins'],
            state_gaps=match['gaps_hartree'],candidate_energy=np.asarray(td.e),
            candidate_dipole=moments,candidate_x=x,candidate_y=y,mo_coeff=np.asarray(mf.mo_coeff),
            mo_energy=np.asarray(mf.mo_energy),mo_occ=np.asarray(mf.mo_occ),
            density=np.asarray(mf.make_rdm1()),ground_energy=float(mf.e_tot))
        event('multistate_point_saved',mode=mode+1,sign=sign,step=step,states=count,
              seconds=perf_counter()-started,min_overlap=float(min(match['matched_overlap'])),
              roots=(roots+1).tolist(),legacy_s1_dipole_error=legacy_error)
        del mf,td;gc.collect()
        return gap,dipole

    def derivative(mode,step):
        plus,minus=point(mode,1,step),point(mode,-1,step)
        divisor=2*step*np.sqrt(mass_ratio)
        return (plus[0]-minus[0])/divisor,(plus[1]-minus[1])/divisor

    gradients,slopes=[],[]
    for mode in range(len(omega)):
        g,mu=derivative(mode,SETTINGS['mode_step_bohr_sqrt_amu'])
        gradients.append(g);slopes.append(mu)
        save('mode_response_status.json',dict(complete=False,stage='mode_points',signature=signature,
             input_signature=input_signature,identity=identity,completed_modes=mode+1,
             mode_count=len(omega),state_count=count))
    gradients=np.asarray(gradients).T;slopes=np.moveaxis(np.asarray(slopes),0,-1)
    hr=gradients**2/(2*omega**3);ht=np.sum(slopes**2,axis=1)/(2*omega)
    check_modes=sorted(set(np.argmax(hr,axis=1)) | set(np.argmax(ht,axis=1)))
    checks=[]
    for mode in check_modes:
        g,mu=derivative(int(mode),SETTINGS['mode_check_step_bohr_sqrt_amu'])
        np.testing.assert_allclose(g,gradients[:,mode],atol=3e-8,rtol=3e-3)
        np.testing.assert_allclose(mu,slopes[:,:,mode],atol=3e-7,rtol=3e-3)
        checks.append(dict(mode=int(mode)+1,gap_error=np.abs(g-gradients[:,mode]).tolist(),
                           dipole_error=np.max(abs(mu-slopes[:,:,mode]),axis=1).tolist()))
    checkpoint(OUTPUT/'mode_response.npz',frequencies=omega,gap_gradient=gradients,
        dipole=baseline['dipole'][:count],dipole_derivative=slopes,
        vertical_energy=baseline['excitation'][:count],signature=signature)
    save('mode_response_status.json',dict(complete=True,signature=signature,identity=identity,
        target_roots=list(range(1,count+1)),state_count=count,mode_count=len(omega),step_checks=checks,
        derivative_method='Complete re-converged SCF/full TDDFT central differences',
        mode_coordinate_units='bohr sqrt(electron mass)',harmonic_model='0 K independent equal-curvature VG'))


if __name__=='__main__':
    try:main()
    except Exception:
        save('mode_response_failure.json',dict(complete=False,error=traceback.format_exc()))
        raise
