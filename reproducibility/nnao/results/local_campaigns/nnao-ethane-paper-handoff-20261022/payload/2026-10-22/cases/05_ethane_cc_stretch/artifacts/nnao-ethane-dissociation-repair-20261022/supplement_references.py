"""Add the same neutral-population seed to the conventional reference search."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
from pathlib import Path
import hashlib
import json
import time
from pyscf import gto
from traditional_branches import solve, OUT, BENCHMARK, SOURCE
from reference_seeds import atomic_population_seed


def main():
    report = json.loads((OUT / 'traditional-branch-results.json').read_text())
    library = json.loads((BENCHMARK / 'basis-library.json').read_text())
    rows = json.loads((SOURCE / 'structures.json').read_text())
    report['supplement_runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['identity']['branch_policy'] += ' + neutral_atomic_population'
    report['status'] = 'running'
    report['results'] = []
    previous = json.loads((OUT / 'traditional-branch-results.json').read_text())['results']
    started = time.perf_counter()
    for old in previous:
        row = rows[old['index']]
        mol = gto.M(atom=list(zip(row['symbols'], row['coords_angstrom'])), basis=library[old['basis']],
                    unit='Angstrom', charge=0, spin=0, cart=False, verbose=0)
        mf, check = solve(mol, atomic_population_seed(mol), 'neutral_atomic_population')
        record = dict(old, branch_attempts=old['branch_attempts']+[check])
        if check['valid'] and check['energy_hartree'] < old['energy_hartree']:
            record.update(energy_hartree=check['energy_hartree'], selected_start='neutral_atomic_population',
                orbital_residual=check['orbital_residual'], converged=True, internal_stable=True,
                energy_lowering_hartree=old['archived_hcore_energy_hartree']-check['energy_hartree'])
        report['results'].append(record)
        report['supplement_seconds'] = time.perf_counter()-started
        target = OUT / 'traditional-complete-results.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(target)
        print(old['basis'], old['distance_angstrom'], record['energy_hartree'], record['selected_start'], flush=True)
    report['status'] = 'complete'
    target.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
