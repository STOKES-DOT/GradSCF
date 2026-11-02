"""Independently validate every frozen NNAO basis as repaired points arrive."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
from pathlib import Path
import json
import time
import numpy as np
from pyscf import gto, scf
from reference_seeds import atomic_population_seed
from traditional_branches import solve, OUT, SOURCE


def main():
    audit = {r['index']: r for r in json.loads((OUT / 'archived-native-audit.json').read_text())['results']}
    rows = json.loads((SOURCE / 'structures.json').read_text())
    target = OUT / 'independent-validation.json'
    report = json.loads(target.read_text()) if target.exists() else dict(status='running',
        method='independent CPU PySCF DF-RHF', tolerance_hartree=1e-8, results=[], failed=0)
    if report['failed']:
        raise RuntimeError('An energy mismatch needs explicit investigation before resuming.')
    done = {(r['index'], r['basis']) for r in report['results']}
    previous_seconds = report.get('elapsed_seconds', 0.)
    previous = {}
    state_dir = OUT / 'reference-states'
    state_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    for index, row in enumerate(rows):
        folder = SOURCE / f'distance-{index:03d}' if index < 24 else OUT / 'native-rerun' / f'distance-{index:03d}'
        while not (folder / 'frozen-bases.jsonl').exists():
            if time.perf_counter()-started > 3600:
                raise TimeoutError('Native point has not completed within the bounded validation window.')
            time.sleep(10)
        records = [json.loads(s) for s in (folder / 'frozen-bases.jsonl').read_text().splitlines()]
        for frozen in records:
            name = frozen['basis']
            energy = audit[index]['mace_energy_hartree'] if name == 'MACE-update3351' else frozen['native_energy_hartree']
            mol = gto.M(atom=list(zip(frozen['atom_labels'], row['coords_angstrom'])),
                basis=dict(zip(frozen['atom_labels'], frozen['atom_shells'])), unit='Angstrom',
                charge=0, spin=0, cart=False, verbose=0)
            if (index, name) in done:
                state = state_dir / f'{index:03d}-{name}.npz'
                previous[name] = mol, np.load(state)['density']
                continue
            seeds = [('hcore', None), ('minao', None), ('neutral_atomic_population', atomic_population_seed(mol))]
            if name in previous:
                pmol, density = previous[name]
                seeds.append(('projected_previous', scf.addons.project_dm_nr2nr(pmol, density, mol)))
            attempts, solutions = [], []
            for seed, density in seeds:
                try:
                    mf, result = solve(mol, density, seed)
                except Exception as error:
                    attempts.append(dict(start=seed, valid=False, error=repr(error)))
                    continue
                attempts.append(result)
                if result['valid']:
                    solutions.append((result['energy_hartree'], mf, seed))
            if not solutions:
                raise RuntimeError(f'No valid independent solution: {index}, {name}')
            checked, mf, seed = min(solutions, key=lambda r: r[0])
            assert mf._eri is None
            previous[name] = mol, mf.make_rdm1()
            error = float(energy-checked)
            result = dict(index=index, distance_angstrom=row['distance_angstrom'], basis=name,
                native_energy_hartree=energy, independent_energy_hartree=checked,
                error_hartree=error, selected_start=seed, starts=attempts,
                status='passed' if abs(error) <= 1e-8 else 'failed')
            report['results'].append(result)
            report.update(passed=sum(r['status'] == 'passed' for r in report['results']),
                failed=sum(r['status'] != 'passed' for r in report['results']),
                elapsed_seconds=previous_seconds+time.perf_counter()-started,
                failed_seed_count=sum(not s['valid'] for r in report['results'] for s in r['starts']),
                max_abs_error_hartree=max(abs(r['error_hartree']) for r in report['results']))
            temporary = target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(report, indent=2)+'\n')
            temporary.replace(target)
            np.savez(state_dir / f'{index:03d}-{name}.npz', density=mf.make_rdm1(),
                     mo_coeff=mf.mo_coeff, mo_energy=mf.mo_energy, mo_occ=mf.mo_occ)
            print(index, row['distance_angstrom'], name, error, result['status'], flush=True)
            if result['status'] != 'passed':
                raise RuntimeError(f'Independent energy mismatch: {index}, {name}, {error}')
    report['status'] = 'complete'
    target.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
