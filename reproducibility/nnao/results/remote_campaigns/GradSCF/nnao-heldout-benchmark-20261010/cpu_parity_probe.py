"""CPU-only same-basis DF-RHF parity probe; this is not a GPU performance result."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys
import time
import traceback

from gpu4_reference import (CUSTOM_BASES, SETTINGS, atomic_json, file_hash,
                            frozen_basis, jsonl, native_energy_error)


def calculate(structure, frozen, np, gto, scf):
    start = time.perf_counter()
    labels, basis = frozen_basis(structure, frozen)
    mol = gto.M(atom=list(zip(labels, structure['coords_angstrom'])), basis=basis,
                unit='Angstrom', charge=structure['charge'], spin=structure['spin'],
                cart=False, verbose=0)
    if frozen.get('nao', mol.nao_nr()) != mol.nao_nr():
        raise ValueError('Frozen native and CPU PySCF AO counts differ.')
    mf = scf.RHF(mol).density_fit(auxbasis=SETTINGS['auxbasis'])
    mf.conv_tol = SETTINGS['conv_tol']; mf.conv_tol_grad = SETTINGS['conv_tol_grad']
    mf.max_cycle = SETTINGS['max_cycle']; mf.init_guess = 'hcore'
    mf.level_shift = 0.; mf.chkfile = None
    density0 = np.asarray(mf.get_init_guess(mol, key='hcore'), dtype=np.float64)
    setup_seconds = time.perf_counter() - start

    def kernel(density, shift):
        cycles = [0]
        mf.callback = lambda env: cycles.__setitem__(0, max(cycles[0], int(env['cycle']) + 1))
        mf.level_shift = shift
        tick = time.perf_counter()
        energy = float(mf.kernel(dm0=density.copy()))
        return dict(level_shift=shift, converged=bool(mf.converged), scf_cycles=cycles[0],
                    elapsed_seconds=time.perf_counter() - tick, energy_hartree=energy)

    passes = [kernel(density0, 0.)]
    if not mf.converged:
        passes.append(kernel(density0, SETTINGS['rescue_level_shift']))
        if mf.converged:
            passes.append(kernel(np.asarray(mf.make_rdm1()), 0.))
    if not mf.converged or mf.level_shift != 0.:
        raise RuntimeError(f'Unshifted CPU SCF did not converge: {passes}')
    density = np.asarray(mf.make_rdm1()); overlap = np.asarray(mf.get_ovlp(mol))
    hcore = np.asarray(mf.get_hcore(mol)); veff = np.asarray(mf.get_veff(mol, density))
    if any(array.dtype != np.float64 for array in (density, overlap, hcore, veff)):
        raise RuntimeError('CPU SCF arrays must all use float64.')
    energy = float(mf.e_tot); fock = hcore + veff
    diagnostics = dict(
        min_overlap_eigenvalue=float(np.linalg.eigvalsh(overlap)[0]),
        orbital_residual=float(np.linalg.norm(fock @ density @ overlap - overlap @ density @ fock)),
        energy_reconstruction_error=float(abs(np.sum(density * hcore) + .5 * np.sum(density * veff)
                                             + mol.energy_nuc() - energy)))
    if (not np.isfinite([energy, *diagnostics.values()]).all()
            or diagnostics['min_overlap_eigenvalue'] < 1e-8
            or diagnostics['orbital_residual'] > 1e-7
            or diagnostics['energy_reconstruction_error'] > 1e-9):
        raise RuntimeError(f'Invalid CPU SCF diagnostics: {diagnostics}')
    error = native_energy_error(energy, frozen)
    return dict(status='passed', structure_id=structure['structure_id'], basis=frozen['basis'],
        nao=mol.nao_nr(), hamiltonian_dimension=mol.nao_nr(), nelectron=mol.nelectron,
        scf_converged=True, setup_seconds=setup_seconds,
        scf_seconds=sum(item['elapsed_seconds'] for item in passes),
        scf_cycles=sum(item['scf_cycles'] for item in passes), rescue_used=len(passes) > 1,
        passes=passes, elapsed_seconds=time.perf_counter() - start, energy_hartree=energy,
        native_energy_hartree=frozen.get('native_energy_hartree', frozen.get('energy_hartree')),
        native_energy_error_hartree=error, **diagnostics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--frozen-bases', type=Path, required=True)
    parser.add_argument('--id', default='W4-11/n2o')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive.')
    rows = jsonl(args.structures)
    if len({row['structure_id'] for row in rows}) != len(rows):
        parser.error('Duplicate structure IDs.')
    structures = [row for row in rows if row['structure_id'] == args.id]
    if len(structures) != 1:
        parser.error('Requested structure ID is absent or duplicated.')
    structure = structures[0]
    if structure.get('charge') != 0 or structure.get('spin') != 0:
        parser.error('This CPU probe requires a neutral closed-shell structure.')
    frozen_rows = jsonl(args.frozen_bases)
    frozen = {(row['structure_id'], row['basis']): row for row in frozen_rows}
    if len(frozen) != len(frozen_rows):
        parser.error('Duplicate frozen structure/basis pairs.')
    for name in CUSTOM_BASES:
        if (args.id, name) not in frozen:
            parser.error(f'Missing frozen custom basis: {name}')
        frozen_basis(structure, frozen[(args.id, name)])
    if args.output.exists():
        parser.error('Refusing to overwrite an existing CPU parity probe report.')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    start = time.perf_counter()
    import numpy as np
    import pyscf
    from pyscf import gto, scf
    pyscf.lib.num_threads(args.threads)
    imports = {name: dict(path=str(Path(module.__file__).resolve()), sha256=file_hash(module.__file__))
               for name, module in (('numpy', np), ('pyscf', pyscf), ('pyscf.gto', gto),
                                    ('pyscf.scf', scf))}
    report = dict(status='running', backend='CPU PySCF density fitting', structure_id=args.id,
        benchmark_scope='CPU same-basis physical parity only; not GPU4PySCF performance.',
        settings=SETTINGS, host=platform.node(), cpu_machine=platform.machine(),
        cpu_processor=platform.processor(), cpu_threads=pyscf.lib.num_threads(),
        thread_environment={name: os.environ[name] for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), python=platform.python_version(),
        python_executable=str(Path(sys.executable).resolve()),
        versions=dict(numpy=np.__version__, pyscf=pyscf.__version__), imported_sources=imports,
        input_hashes=dict(structures=file_hash(args.structures), frozen_bases=file_hash(args.frozen_bases),
                          script=file_hash(__file__), shared_guards=file_hash(Path(__file__).with_name('gpu4_reference.py'))),
        requested_four_center_eri=False, performed_gradients=False, initialized_gpu=False,
        import_seconds=time.perf_counter() - start, results=[])
    atomic_json(args.output, report)
    for name in CUSTOM_BASES:
        try:
            result = calculate(structure, frozen[(args.id, name)], np, gto, scf)
        except Exception as exc:
            result = dict(status='failed', structure_id=args.id, basis=name,
                          error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        report['results'].append(result)
        atomic_json(args.output, report)
        print(f"{args.id} {name}: {result['status']}", flush=True)
    report.update(status='passed' if all(row['status'] == 'passed' for row in report['results']) else 'failed',
                  elapsed_seconds=time.perf_counter() - start)
    atomic_json(args.output, report)
    print(f"CPU parity probe: {report['status']}; {report['elapsed_seconds']:.3f} seconds", flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
