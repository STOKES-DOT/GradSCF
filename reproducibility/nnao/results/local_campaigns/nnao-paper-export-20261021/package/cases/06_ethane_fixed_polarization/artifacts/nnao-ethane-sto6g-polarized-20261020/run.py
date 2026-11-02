"""Add explicit fixed STO-6G polarization controls to the frozen torsion grid.

This is comparison-only code. PySCF/GPU4PySCF are never production dependencies.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import os
import time

import cupy as cp
import numpy as np
import pyscf
import gpu4pyscf
from pyscf import gto, lib
from gpu4pyscf.scf import hf

OUT = Path(__file__).resolve().parent
OLD = OUT.parent/'nnao-ethane-torsion-20261018'


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ao_count(basis, symbols):
    return sum(sum((2*shell[0]+1)*(len(shell[1])-1) for shell in basis[symbol])
               for symbol in symbols)


def main():
    lib.num_threads(32)
    geometry = OLD/'structures-dense.json'
    rows = json.loads(geometry.read_text())
    np.testing.assert_array_equal([row['angle_deg'] for row in rows], np.arange(0, 121, 5))
    reference_path = OLD/'traditional-basis-library.json'
    reference = json.loads(reference_path.read_text())['6-31g(d,p)']
    carbon_d = [shell for shell in reference['C'] if shell[0] == 2]
    hydrogen_p = [shell for shell in reference['H'] if shell[0] == 1]
    assert carbon_d == [[2, [0.8, 1.0]]] and hydrogen_p == [[1, [1.1, 1.0]]]
    minimal = {symbol: gto.basis.load('sto-6g', symbol) for symbol in ['H', 'C']}
    assert [shell[0] for shell in minimal['C']] == [0, 0, 1]
    assert [shell[0] for shell in minimal['H']] == [0]
    assert all(len(shell)-1 == 6 for shells in minimal.values() for shell in shells)
    heavy = deepcopy(minimal)
    heavy['C'] += deepcopy(carbon_d)
    both = deepcopy(heavy)
    both['H'] += deepcopy(hydrogen_p)
    bases = {'STO-6G': minimal, 'STO-6G + C(d)': heavy,
             'STO-6G + C(d), H(p)': both}
    expected = dict(zip(bases, [16, 26, 44]))
    for name, basis in bases.items():
        assert ao_count(basis, rows[0]['symbols']) == expected[name]
    atomic_json(OUT/'basis-library.json', dict(bases=bases,
        interpretation='Explicit custom additions, not a claim of a standardized STO-6G** basis.',
        minimal_source='PySCF bundled STO-6G library; all primitives and coefficients unchanged.',
        polarization_source=str(reference_path), polarization_source_sha256=file_hash(reference_path),
        polarization_units='exponent: Bohr^-2; raw coefficient: dimensionless',
        polarization=dict(carbon_d=carbon_d, hydrogen_p=hydrogen_p)))
    report = dict(status='running', job_id=os.environ.get('SLURM_JOB_ID'), method='DF-RHF',
        auxbasis='def2-universal-jkfit', dtype='float64', conv_tol=1e-12, conv_tol_grad=1e-9,
        max_cycle=200, cpus=32, device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        versions=dict(pyscf=pyscf.__version__, gpu4pyscf=getattr(gpu4pyscf, '__version__', 'unavailable'),
                      cupy=cp.__version__, numpy=np.__version__),
        geometry_file=str(geometry), geometry_sha256=file_hash(geometry),
        basis_library_sha256=file_hash(OUT/'basis-library.json'), results=[])
    atomic_json(OUT/'results.json', report)
    started = time.perf_counter()
    for name, basis in bases.items():
        for row in rows:
            tick = time.perf_counter()
            mol = gto.M(atom=list(zip(row['symbols'], row['coords_angstrom'])),
                        basis=basis, unit='Angstrom', charge=0, spin=0, cart=False, verbose=0)
            assert mol.nao_nr() == expected[name]
            mf = hf.RHF(mol).density_fit(auxbasis=report['auxbasis'])
            mf.conv_tol = report['conv_tol']
            mf.conv_tol_grad = report['conv_tol_grad']
            mf.max_cycle = report['max_cycle']
            mf.chkfile = None
            mf.level_shift = 0.
            initial = cp.asarray(mf.get_init_guess(mol, key='hcore'), dtype=cp.float64)
            mf.kernel(dm0=initial)
            rescued = not mf.converged
            if rescued:
                mf.level_shift = .2
                mf.kernel(dm0=initial)
                density = mf.make_rdm1()
                mf.level_shift = 0.
                mf.kernel(dm0=density)
            assert mf.converged and mf.level_shift == 0., (name, row['angle_deg'])
            density = mf.make_rdm1()
            overlap = mf.get_ovlp()
            fock = mf.get_hcore()+mf.get_veff(mol, density)
            residual = float(cp.linalg.norm(fock@density@overlap-overlap@density@fock).get())
            energy = float(mf.e_tot)
            assert np.isfinite(energy) and residual < 1e-7, (name, row['angle_deg'], residual)
            result = dict(basis=name, angle_deg=row['angle_deg'], energy_hartree=energy,
                nao=mol.nao_nr(), status='passed', orbital_residual=residual,
                rescue_used=rescued, seconds=time.perf_counter()-tick)
            report['results'].append(result)
            report['elapsed_seconds'] = time.perf_counter()-started
            atomic_json(OUT/'results.json', report)
            print(json.dumps(result), flush=True)
            del mf, initial, density, overlap, fock
            cp.get_default_memory_pool().free_all_blocks()
    assert len(report['results']) == 75
    report.update(status='complete', passed=75, elapsed_seconds=time.perf_counter()-started)
    atomic_json(OUT/'results.json', report)
    print('COMPLETE', json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
