"""Comparison-only conventional DF-RHF on the archived rigid C-C scan."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '4'

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pyscf
from pyscf import gto, lib, scf

OUT = Path(__file__).resolve().parent
ARTIFACTS = OUT.parent
SOURCE = ARTIFACTS / 'nnao-ethane-dissociation-20261019'
BASES = ('STO-3G', 'STO-6G', 'STO-6G + C(d), H(p)', '3-21G',
         '6-31G(d,p)', 'cc-pVDZ', 'cc-pVTZ')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def calculate(row, basis):
    mol = gto.M(atom=list(zip(row['symbols'], row['coords_angstrom'])), basis=basis,
                unit='Angstrom', charge=row['charge'], spin=row['spin'], cart=False, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit')
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.chkfile = None
    mf.max_memory = 4000
    cycles = []
    mf.callback = lambda env: cycles.append(int(env['cycle']))
    mf.kernel(dm0=mf.get_init_guess(mol, key='hcore'))
    rescue = False
    if not mf.converged:
        rescue = True
        mf.level_shift = .2
        mf.kernel(dm0=mf.make_rdm1())
        density = mf.make_rdm1()
        mf.level_shift = 0.
        mf.kernel(dm0=density)
    if not mf.converged:
        rescue = True
        mf = mf.newton()
        mf.kernel()
    restarts = 0
    stable = False
    for attempt in range(5):
        if not mf.converged:
            break
        internal, _, stable, _ = mf.stability(internal=True, external=False, return_status=True)
        if stable:
            break
        if attempt < 4:
            restarts += 1
            mf.kernel(dm0=mf.make_rdm1(internal, mf.mo_occ))
    density = mf.make_rdm1()
    overlap = mf.get_ovlp()
    fock = mf.get_hcore() + mf.get_veff(mol, density)
    residual = float(np.linalg.norm(fock @ density @ overlap - overlap @ density @ fock))
    energy = float(mf.e_tot)
    passed = bool(mf.converged and stable and np.isfinite(energy) and residual < 1e-7)
    if mf._eri is not None:
        raise RuntimeError('Unexpected full four-index ERI in this DF comparison.')
    return dict(energy_hartree=energy, nao=mol.nao_nr(), status='passed' if passed else 'failed',
                converged=bool(mf.converged), internal_stable=bool(stable), orbital_residual=residual,
                min_overlap_eigenvalue=float(np.linalg.eigvalsh(overlap)[0]),
                stability_restarts=restarts, rescue_used=rescue, scf_cycle_callbacks=len(cycles))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Three distances per basis, reused by the full run.')
    args = parser.parse_args()
    lib.num_threads(4)
    rows = read(SOURCE / 'structures.json')
    assert len(rows) == 56 and all(r['charge'] == r['spin'] == 0 for r in rows)
    torsion = ARTIFACTS / 'nnao-ethane-torsion-20261018'
    library = read(torsion / 'traditional-basis-library.json')
    controls = read(ARTIFACTS / 'nnao-ethane-sto6g-polarized-20261020/basis-library.json')['bases']
    bases = {name: controls[name] if name in controls else library[name.lower()] for name in BASES}
    basis_path = OUT / 'basis-library.json'
    if basis_path.exists():
        assert read(basis_path) == bases
    else:
        write(basis_path, bases)
    identity = dict(geometry_sha256=digest(SOURCE / 'structures.json'),
                    basis_sha256=digest(basis_path), script_sha256=digest(Path(__file__)),
                    method='DF-RHF', auxbasis='def2-universal-jkfit', dtype='float64',
                    conv_tol=1e-12, conv_tol_grad=1e-9, max_cycle=200,
                    required_ao_commutator_norm=1e-7, internal_stability='PySCF RHF internal analysis',
                    restricted_branch=True, full_eri=False, threads=4, pyscf=pyscf.__version__)
    path = OUT / 'traditional-results.json'
    report = read(path) if path.exists() else dict(identity=identity, results=[],
        hardware=dict(host=platform.node(), platform=platform.platform(), processor=platform.processor(),
                      logical_cpus=os.cpu_count(), backend='CPU', numpy=np.__version__))
    if report['identity'] != identity:
        raise ValueError('Inputs or runner changed; do not silently reuse old calculations.')
    selected = {0, min(range(len(rows)), key=lambda i: abs(rows[i]['distance_angstrom'] - 2.8)), 55}
    done = {(r['basis'], r['index']) for r in report['results']}
    started = time.perf_counter()
    previous_seconds = report.get('elapsed_seconds', 0.)
    for name in BASES:
        for index, row in enumerate(rows):
            if (args.smoke and index not in selected) or (name, index) in done:
                continue
            tick = time.perf_counter()
            result = dict(basis=name, index=index, structure_id=row['structure_id'],
                          distance_angstrom=row['distance_angstrom'])
            try:
                result.update(calculate(row, bases[name]))
            except Exception as error:
                result.update(status='failed', error=repr(error))
            result['seconds'] = time.perf_counter() - tick
            report['results'].append(result)
            report.update(status='running', elapsed_seconds=previous_seconds + time.perf_counter() - started,
                          requested_cases=len(BASES) * len(rows))
            write(path, report)
            print(json.dumps(result), flush=True)
    report.update(status='complete' if len(report['results']) == len(BASES) * len(rows) else 'partial',
                  passed=sum(r['status'] == 'passed' for r in report['results']),
                  failed=sum(r['status'] != 'passed' for r in report['results']))
    write(path, report)
    print('SUMMARY', json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
