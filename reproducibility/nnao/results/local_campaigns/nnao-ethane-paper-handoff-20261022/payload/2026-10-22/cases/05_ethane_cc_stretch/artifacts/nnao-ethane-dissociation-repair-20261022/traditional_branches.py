"""Audit conventional restricted-HF branches instead of accepting hcore alone."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from pyscf import gto, scf

OUT = Path(__file__).resolve().parent
BENCHMARK = OUT.parent / 'nnao-ethane-dissociation-benchmark-20261022'
SOURCE = OUT.parent / 'nnao-ethane-dissociation-20261019'


def solve(mol, density, name):
    mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit')
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.chkfile = None
    mf.kernel(dm0=density if density is not None else mf.get_init_guess(mol, key=name))
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    stable = False
    for _ in range(5):
        if not mf.converged:
            break
        internal, _, stable, _ = mf.stability(external=False, return_status=True)
        if stable:
            break
        mf.kernel(dm0=mf.make_rdm1(internal, mf.mo_occ))
    p, s = mf.make_rdm1(), mf.get_ovlp()
    f = mf.get_hcore() + mf.get_veff(mol, p)
    residual = float(np.linalg.norm(f @ p @ s - s @ p @ f))
    return mf, dict(start=name, energy_hartree=float(mf.e_tot), converged=bool(mf.converged),
        internal_stable=bool(stable), orbital_residual=residual,
        valid=bool(mf.converged and stable and residual < 1e-7))


def main():
    old = json.loads((BENCHMARK / 'traditional-results.json').read_text())
    rows = json.loads((SOURCE / 'structures.json').read_text())
    library = json.loads((BENCHMARK / 'basis-library.json').read_text())
    old_points = {(r['basis'], r['index']): r for r in old['results']}
    report = dict(identity=dict(old['identity'], branch_policy='archived_hcore + minao + projected_previous'),
        hardware=old['hardware'], requested_cases=392, results=[],
        original_results_sha256=hashlib.sha256((BENCHMARK / 'traditional-results.json').read_bytes()).hexdigest())
    started = time.perf_counter()
    for name, basis in library.items():
        previous = None
        for index, row in enumerate(rows):
            tick = time.perf_counter()
            mol = gto.M(atom=list(zip(row['symbols'], row['coords_angstrom'])), basis=basis,
                        unit='Angstrom', charge=0, spin=0, cart=False, verbose=0)
            guesses = [('minao', None)]
            if previous is not None:
                pmol, density = previous
                guesses.append(('projected_previous', scf.addons.project_dm_nr2nr(pmol, density, mol)))
            attempts, solutions = [], []
            for start, density in guesses:
                mf, result = solve(mol, density, start)
                attempts.append(result)
                if result['valid']:
                    solutions.append((result['energy_hartree'], mf, start, result))
            original = old_points[name, index]
            if not solutions or original['energy_hartree'] < min(r[0] for r in solutions) - 1e-8:
                mf, result = solve(mol, None, 'hcore')
                attempts.append(result)
                if result['valid']:
                    solutions.append((result['energy_hartree'], mf, 'hcore', result))
            if not solutions:
                raise RuntimeError(f'No valid restricted reference: {name}, {index}')
            energy, selected, seed, info = min(solutions, key=lambda r: r[0])
            assert selected._eri is None
            previous = mol, selected.make_rdm1()
            record = dict(original, energy_hartree=energy, orbital_residual=info['orbital_residual'],
                converged=True, internal_stable=True, status='passed', selected_start=seed,
                archived_hcore_energy_hartree=original['energy_hartree'],
                energy_lowering_hartree=original['energy_hartree']-energy,
                branch_attempts=attempts, seconds=time.perf_counter()-tick)
            report['results'].append(record)
            report.update(status='running', elapsed_seconds=time.perf_counter()-started,
                          passed=len(report['results']), failed=0)
            path = OUT / 'traditional-branch-results.json'
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(report, indent=2)+'\n')
            temporary.replace(path)
            print(name, index, row['distance_angstrom'], energy, record['energy_lowering_hartree'], flush=True)
    report['status'] = 'complete'
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
