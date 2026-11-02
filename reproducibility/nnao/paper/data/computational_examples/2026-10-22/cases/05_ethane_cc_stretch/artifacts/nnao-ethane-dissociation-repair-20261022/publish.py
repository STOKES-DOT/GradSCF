"""Publish only rechecked stationary endpoints and preserve the earlier release."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

OUT = Path(__file__).resolve().parent
BENCHMARK = OUT.parent / 'nnao-ethane-dissociation-benchmark-20261022'
SOURCE = OUT.parent / 'nnao-ethane-dissociation-20261019'


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    audit = read(OUT / 'archived-native-audit.json')
    repaired = read(OUT / 'native-rerun/scan-summary.json')
    references = read(OUT / 'traditional-complete-results.json')
    independent = read(OUT / 'independent-validation.json')
    assert independent['status'] == 'complete' and independent['passed'] == 112 and independent['failed'] == 0
    assert audit['status'] == references['status'] == 'complete'
    assert len(audit['results']) == 56 and len(references['results']) == 392
    new = {r['index']: r for r in repaired['results']}
    needed = {r['index'] for r in audit['results'] if not r['reusable']}
    if not needed.issubset(new):
        raise RuntimeError('Repair still missing: ' + str(sorted(needed - set(new))))
    original = read(SOURCE / 'scan-summary.json')['points']
    geometries = read(SOURCE / 'structures.json')
    points, corrections = [], []
    final = OUT / 'final-bases'
    final.mkdir(exist_ok=True)
    for checked in sorted(audit['results'], key=lambda r: r['index']):
        i = checked['index']
        options = []
        if checked['reusable']:
            options.append((checked['free_energy_hartree'], checked['gradient_norm'], checked['free_scf'],
                            SOURCE / f'distance-{i:03d}', 'archived_endpoint_rechecked'))
        if i in new:
            r = new[i]
            assert r['status'] == 'passed' and r['gradient_norm'] <= 1e-5
            options.append((r['free_energy_hartree'], r['gradient_norm'], r['free_scf'],
                            OUT / 'native-rerun' / f'distance-{i:03d}', 'reoptimized'))
        energy, norm, info, source, mode = min(options, key=lambda r: r[0])
        assert info['converged'] and info['stable'] and info['fixed_point_residual'] < 1e-7
        assert info['orbital_residual'] < 1e-7 and info['reconstruction_error'] < 1e-9
        assert norm <= 1e-5 and energy <= checked['mace_energy_hartree'] + 1e-8
        point = dict(index=i, distance_angstrom=checked['distance_angstrom'], status='passed',
            free_gradient_converged=True, gradient_norm=norm, nao=44,
            mace_energy_hartree=checked['mace_energy_hartree'], mace_scf=checked['mace_scf'],
            free_energy_hartree=energy, free_scf=info, endpoint_origin=mode,
            coefficients_source=str(source / 'best-coefficients.npz'),
            starts=new[i]['starts'] if mode == 'reoptimized' else original[i]['starts'])
        points.append(point)
        destination = final / f'distance-{i:03d}'
        destination.mkdir(exist_ok=True)
        shutil.copy2(source / 'best-coefficients.npz', destination / 'best-coefficients.npz')
        # MACE outputs are frozen; only the selected restricted electronic solution changes.
        mace = next(json.loads(s) for s in (SOURCE / f'distance-{i:03d}/frozen-bases.jsonl').read_text().splitlines()
                    if json.loads(s)['basis'] == 'MACE-update3351')
        free = next(json.loads(s) for s in (source / 'frozen-bases.jsonl').read_text().splitlines()
                    if json.loads(s)['basis'] == 'free-contraction-best')
        mace['native_energy_hartree'] = point['mace_energy_hartree']
        free['native_energy_hartree'] = energy
        (destination / 'frozen-bases.jsonl').write_text(json.dumps(mace)+'\n'+json.dumps(free)+'\n')
        write(destination / 'structure.json', geometries[i])
        corrections.append(dict(index=i, distance_angstrom=point['distance_angstrom'],
            original_status=original[i]['status'], original_gradient_norm=original[i]['gradient_norm'],
            new_gradient_norm=norm, endpoint_origin=mode,
            mace_energy_lowering_hartree=original[i]['mace_energy_hartree']-point['mace_energy_hartree'],
            free_energy_lowering_hartree=original[i]['free_energy_hartree']-energy))
    snapshot = OUT / 'selected-native-rerun-snapshot.json'
    write(snapshot, repaired)
    payload = dict(status='complete', points=points, checkpoint_update=3351, required_repair_indices=sorted(needed),
        original_scan_sha256=hashlib.sha256((SOURCE / 'scan-summary.json').read_bytes()).hexdigest(),
        branch_search=['hcore', 'neutral_atomic_population'], method='native DF-RHF',
        fixed_core=True, fixed_exponents=True, coefficient_gradient='existing implicit SCF',
        finite_differences=False, corrections=corrections,
        independent_optimization_runs='unpreconditioned critical scan and overlap-whitened production scan')
    write(BENCHMARK / 'nnao-results.json', payload)
    references['identity']['base_runner_sha256'] = references['identity']['script_sha256']
    references['identity']['script_sha256'] = hashlib.sha256((OUT / 'traditional_branches.py').read_bytes()).hexdigest()
    write(BENCHMARK / 'traditional-results.json', references)
    write(OUT / 'publication.json', dict(status='published', nnao_points=56, traditional_points=392,
        reoptimized_points=sum(r['endpoint_origin'] == 'reoptimized' for r in points),
        rechecked_reused_points=sum(r['endpoint_origin'] != 'reoptimized' for r in points),
        required_repair_indices=sorted(needed),
        source_snapshots={name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in
            ('archived-native-audit.json', 'traditional-complete-results.json', 'selected-native-rerun-snapshot.json')}))
    print(json.dumps(dict(status='published', required_repair_indices=sorted(needed),
        critical=[r for r in corrections if 2.7 <= r['distance_angstrom'] <= 2.9])))


if __name__ == '__main__':
    main()
