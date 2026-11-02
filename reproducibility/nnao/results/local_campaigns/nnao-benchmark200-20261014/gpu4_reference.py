"""Independent GPU4PySCF DF-RHF comparison of frozen held-out NNAO bases.

This artifact is a reference benchmark, not production GradSCF code. Run after
the native forward process exits so the two CUDA memory pools do not coexist.
Only DF three-index factors are requested; no four-index ERI is constructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import socket
import time
import traceback


STANDARD_BASES = ('minao', 'sto-3g', '3-21g', '6-31g', '6-31g(d)', '6-31g(d,p)',
                  'def2-svp', 'cc-pvdz', '6-311g(d,p)', 'def2-tzvp',
                  'def2-tzvpp', 'cc-pvtz')
CUSTOM_BASES = ('NNAO-reference', 'NNAO-epoch41', 'NNAO-pilot-best975', 'NNAO-200-epoch594', 'NNAO-200-initial')
DEFAULT_BASES = STANDARD_BASES + CUSTOM_BASES[:2]
SETTINGS = dict(method='DF-RHF', cartesian=False, dtype='float64',
                auxbasis='def2-universal-jkfit', conv_tol=1e-12,
                conv_tol_grad=1e-9, max_cycle=200, init_guess='hcore',
                rescue_level_shift=.2, native_energy_atol_hartree=1e-7)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def frozen_profiles(rows):
    """Record weight provenance separately for each frozen basis variant."""
    fields = ('checkpoint_profile', 'checkpoint_epoch', 'model_training_structure_count',
              'checkpoint_sha256', 'native_source_sha256')
    profiles = {}
    for row in rows:
        basis = row['basis']
        profile = {key: row[key] for key in fields if key in row}
        if basis == 'NNAO-pilot-best975' and (
                profile.get('checkpoint_profile') != 'pilot-best975'
                or profile.get('checkpoint_epoch') != 975
                or profile.get('model_training_structure_count') != 24):
            raise ValueError('Frozen pilot-best975 profile must identify epoch 975 trained on 24 structures.')
        if basis in profiles and profiles[basis] != profile:
            raise ValueError(f'{basis}: frozen model provenance differs between structures.')
        profiles[basis] = profile
    return profiles


def profile_note(profiles):
    descriptions = []
    for basis, profile in profiles.items():
        if basis == 'NNAO-reference':
            descriptions.append('NNAO-reference: initial contractions, not a trained checkpoint')
        else:
            descriptions.append(f'{basis}: profile={profile.get("checkpoint_profile", basis)}, '
                f'checkpoint epoch={profile.get("checkpoint_epoch", "unknown")}, '
                f'training structures={profile.get("model_training_structure_count", "unknown")}')
    return '; '.join(descriptions) + '; frozen forward evaluation, no fitting here.'


def frozen_basis(structure, frozen):
    """Keep atom-specific coefficients separate even for equal elements."""
    if frozen['structure_id'] != structure['structure_id']:
        raise ValueError('Frozen basis structure ID differs from geometry.')
    labels = frozen['atom_labels']; shells = frozen['atom_shells']
    if len(labels) != len(set(labels)):
        raise ValueError('Frozen atom labels must be unique.')
    if len(labels) != len(structure['symbols']) or len(shells) != len(labels):
        raise ValueError('Frozen basis atom count differs from geometry.')
    if 'symbols' in frozen and frozen['symbols'] != structure['symbols']:
        raise ValueError('Frozen basis symbols differ from geometry.')
    if ('coords_angstrom' in frozen and
            frozen['coords_angstrom'] != structure['coords_angstrom']):
        raise ValueError('Frozen basis coordinates differ from geometry.')
    for symbol, label, atom_shells in zip(structure['symbols'], labels, shells):
        element = re.match(r'[A-Za-z]+', label)
        if element is None or element.group() != symbol:
            raise ValueError('Frozen atom label element differs from geometry.')
        if not atom_shells:
            raise ValueError('Empty frozen atomic basis.')
        for block in atom_shells:
            if (not isinstance(block[0], int) or block[0] < 0 or len(block) < 2
                    or any(len(row) != len(block[1]) or len(row) < 2 or row[0] <= 0
                           or not all(math.isfinite(float(value)) for value in row)
                           for row in block[1:])):
                raise ValueError('Invalid frozen primitive exponent/coefficient.')
            if any(all(row[column] == 0. for row in block[1:])
                   for column in range(1, len(block[1]))):
                raise ValueError('Frozen contraction vector is zero.')
    return labels, dict(zip(labels, shells))


def reusable_case(saved, signature):
    return (saved.get('status') == 'passed' and saved.get('signature') == signature
            and len(saved.get('timings', [])) == 2)


def run_case(structure, basis_name, frozen, operation):
    """Missing native evidence fails one custom row, preserving standard runs."""
    if basis_name in CUSTOM_BASES and frozen is None:
        return dict(structure_id=structure['structure_id'], basis=basis_name,
            status='failed', error_type='MissingNativeBasis',
            error='Native forward did not publish a valid frozen basis; independent same-basis validation cannot be performed.')
    return operation()


def native_energy_error(energy, frozen):
    reference = frozen.get('native_energy_hartree', frozen.get('energy_hartree'))
    if reference is None or not math.isfinite(energy) or not math.isfinite(reference):
        raise ValueError('Finite native and independent GPU4PySCF energy are required.')
    error = abs(energy - reference)
    if error > SETTINGS['native_energy_atol_hartree']:
        raise ValueError(f'Native/GPU4PySCF energy error {error:.6g} Ha exceeds 1e-7 Ha.')
    return error


def add_reference_differences(rows):
    anchors = {row['structure_id']: row['energy_hartree'] for row in rows
               if row['basis'] == 'cc-pvtz' and row['status'] == 'passed'}
    for row in rows:
        if row['status'] == 'passed' and row['structure_id'] in anchors:
            row['above_cc_pvtz_millihartree'] = 1000 * (
                row['energy_hartree'] - anchors[row['structure_id']])


def calculate(structure, basis_name, frozen, cp, np, gto, hf):
    started = time.perf_counter()
    labels = structure['symbols']; basis = basis_name
    if frozen is not None:
        labels, basis = frozen_basis(structure, frozen)
    mol = gto.M(atom=list(zip(labels, structure['coords_angstrom'])), basis=basis,
                unit='Angstrom', charge=structure['charge'], spin=structure['spin'],
                cart=False, verbose=0)
    if frozen is not None and frozen.get('nao', mol.nao_nr()) != mol.nao_nr():
        raise ValueError('Frozen native and GPU4PySCF AO counts differ.')
    mf = hf.RHF(mol).density_fit(auxbasis=SETTINGS['auxbasis'])
    mf.conv_tol = SETTINGS['conv_tol']; mf.conv_tol_grad = SETTINGS['conv_tol_grad']
    mf.max_cycle = SETTINGS['max_cycle']; mf.init_guess = 'hcore'
    mf.level_shift = 0.; mf.chkfile = None
    cp.cuda.get_current_stream().synchronize()
    initial_density = cp.asarray(mf.get_init_guess(mol, key='hcore'), dtype=cp.float64)
    row = dict(structure_id=structure['structure_id'], basis=basis_name,
               nao=mol.nao_nr(), hamiltonian_dimension=mol.nao_nr(),
               fock_dimension=mol.nao_nr(), fock_elements=mol.nao_nr() ** 2,
               primitive_nao=sum((2 * mol.bas_angular(i) + 1) * mol.bas_nprim(i)
                                 for i in range(mol.nbas)),
               nelectron=mol.nelectron, occupied_orbitals=mol.nelectron // 2,
               virtual_orbitals=mol.nao_nr() - mol.nelectron // 2,
               setup_seconds=time.perf_counter() - started, timings=[])

    def host(array):
        return np.asarray(array.get() if hasattr(array, 'get') else array)

    def kernel(density, level_shift):
        cycles = [0]
        mf.callback = lambda env: cycles.__setitem__(0, max(cycles[0], int(env['cycle']) + 1))
        mf.level_shift = level_shift
        cp.cuda.get_current_stream().synchronize(); tick = time.perf_counter()
        energy = mf.kernel(dm0=density.copy())
        cp.cuda.get_current_stream().synchronize()
        return dict(level_shift=level_shift, converged=bool(mf.converged),
                    scf_cycles=cycles[0], elapsed_seconds=time.perf_counter() - tick,
                    energy_hartree=float(energy))

    for index in range(2):
        passes = [kernel(initial_density, 0.)]
        if not mf.converged:
            passes.append(kernel(initial_density, SETTINGS['rescue_level_shift']))
            if mf.converged:
                passes.append(kernel(cp.asarray(mf.make_rdm1()), 0.))
        if not mf.converged or mf.level_shift != 0.:
            raise RuntimeError(f'{basis_name}: unshifted SCF did not converge: {passes}')
        density_gpu = mf.make_rdm1(); hcore_gpu = mf.get_hcore(mol)
        veff_gpu = mf.get_veff(mol, density_gpu)
        overlap = host(mf.get_ovlp(mol)); density = host(density_gpu)
        hcore = host(hcore_gpu); veff = host(veff_gpu); fock = hcore + veff
        if any(array.dtype != np.float64 for array in (overlap, density, hcore, veff)):
            raise RuntimeError('SCF arrays must all use float64.')
        energy = float(mf.e_tot)
        diagnostic = dict(converged=True,
            min_overlap_eigenvalue=float(np.linalg.eigvalsh(overlap)[0]),
            orbital_residual=float(np.linalg.norm(fock @ density @ overlap - overlap @ density @ fock)),
            reconstruction_error=float(abs(np.sum(density * hcore)
                                           + .5 * np.sum(density * veff) + mol.energy_nuc() - energy)))
        if (not np.isfinite([energy, *diagnostic.values()]).all()
                or diagnostic['min_overlap_eigenvalue'] < 1e-8
                or diagnostic['orbital_residual'] > 1e-7
                or diagnostic['reconstruction_error'] > 1e-9):
            raise RuntimeError(f'{basis_name}: invalid SCF diagnostics: {diagnostic}')
        if frozen is not None:
            diagnostic['native_energy_error_hartree'] = native_energy_error(energy, frozen)
        row['timings'].append(dict(index=index, temperature='cold' if index == 0 else 'warm',
            energy_hartree=energy, scf_cycles=passes[-1]['scf_cycles'],
            total_scf_cycles=sum(item['scf_cycles'] for item in passes),
            scf_seconds=sum(item['elapsed_seconds'] for item in passes),
            rescue_used=len(passes) > 1, passes=passes, **diagnostic))
    if abs(row['timings'][0]['energy_hartree'] - row['timings'][1]['energy_hartree']) > 1e-8:
        raise RuntimeError('Cold and warm hcore-start energies differ by more than 1e-8 Ha.')
    row.update(status='passed', energy_hartree=row['timings'][1]['energy_hartree'],
               elapsed_seconds=time.perf_counter() - started,
               memory_pool_used_bytes=cp.get_default_memory_pool().used_bytes(),
               memory_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes())
    if frozen is not None:
        row.update(native_energy_hartree=frozen.get('native_energy_hartree', frozen.get('energy_hartree')),
                   native_energy_error_hartree=max(item['native_energy_error_hartree']
                                                   for item in row['timings']),
                   model_profile=frozen_profiles([frozen])[basis_name])
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--frozen-bases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bases', nargs='+', choices=STANDARD_BASES + CUSTOM_BASES,
                        default=DEFAULT_BASES)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    structures = jsonl(args.structures); frozen_rows = jsonl(args.frozen_bases)
    identifiers = [row['structure_id'] for row in structures]
    if not structures or len(identifiers) != len(set(identifiers)):
        parser.error('Structures must be nonempty and have unique IDs.')
    if args.ids:
        if set(args.ids) - set(identifiers):
            parser.error('Requested structure IDs are absent.')
        structures = [row for row in structures if row['structure_id'] in args.ids]
    if len(set(args.bases)) != len(args.bases):
        parser.error('Requested bases must be unique.')
    frozen = {(row['structure_id'], row['basis']): row for row in frozen_rows}
    if len(frozen) != len(frozen_rows):
        parser.error('Duplicate frozen structure/basis pairs.')
    model_profiles = frozen_profiles([row for row in frozen_rows if row['basis'] in args.bases])
    for row in structures:
        if row.get('charge') != 0 or row.get('spin') != 0:
            parser.error('This benchmark requires neutral closed-shell structures.')
        for basis_name in args.bases:
            if basis_name in CUSTOM_BASES:
                key = (row['structure_id'], basis_name)
                if key in frozen:
                    frozen_basis(row, frozen[key])
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((args.output / 'cases').glob('*.json')):
        parser.error('Output has case records; use --resume or a new output directory.')
    threads = os.environ.get('SLURM_CPUS_PER_TASK', '8')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = threads
    import cupy as cp
    import numpy as np
    import gpu4pyscf
    import pyscf
    from pyscf import gto
    from gpu4pyscf.scf import hf
    pyscf.lib.num_threads(int(threads))
    device = cp.cuda.Device(); properties = cp.cuda.runtime.getDeviceProperties(device.id)
    device_name = properties['name']
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    provenance = dict(structures_sha256=file_hash(args.structures),
        frozen_bases_sha256=file_hash(args.frozen_bases), script_sha256=file_hash(__file__),
        python=platform.python_version(),
        versions=dict(pyscf=pyscf.__version__, gpu4pyscf=gpu4pyscf.__version__, cupy=cp.__version__),
        gpu_name=device_name, cuda_runtime_version=cp.cuda.runtime.runtimeGetVersion(),
        cuda_driver_version=cp.cuda.runtime.driverGetVersion(),
        imported_sources={name: dict(path=str(Path(module.__file__).resolve()),
            sha256=file_hash(module.__file__)) for name, module in
            (('pyscf', pyscf), ('gpu4pyscf', gpu4pyscf), ('gpu4pyscf.scf.hf', hf))})
    report = dict(status='running', backend='GPU4PySCF density fitting', settings=SETTINGS,
        energy_reference='cc-pVTZ finite-basis DF-RHF, not CBS or GMTKN55 reference labels',
        nnao_note=profile_note(model_profiles), model_profiles=model_profiles,
        performed_gradients=False, requested_four_center_eri=False,
        gpu_memory_note='Pool used/reserved snapshots are not measured GPU peak memory.',
        cold_warm_note='Both kernels restart from the same hcore density; warm reuses fixed-basis DF setup.',
        host=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        cpu_threads=pyscf.lib.num_threads(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_total_memory_bytes=int(properties['totalGlobalMem']),
        gpu_initial_free_memory_bytes=int(device.mem_info[0]),
        provenance=provenance, target_cases=len(structures) * len(args.bases), results=[])
    start = time.perf_counter()
    atomic_json(args.output / 'metadata.json', report)
    results = report['results']
    for structure in structures:
        for basis_name in args.bases:
            key = (structure['structure_id'], basis_name)
            frozen_row = frozen.get(key) if basis_name in CUSTOM_BASES else None
            signature = hashlib.sha256(json.dumps(dict(structure=structure, basis=basis_name,
                frozen=frozen_row, settings=SETTINGS, provenance=provenance),
                sort_keys=True, allow_nan=False).encode()).hexdigest()
            case_path = args.output / 'cases' / (hashlib.sha256(
                f'{key[0]}\0{key[1]}'.encode()).hexdigest()[:20] + '.json')
            previous = json.loads(case_path.read_text()) if args.resume and case_path.is_file() else None
            if previous and reusable_case(previous, signature):
                result = previous
            else:
                try:
                    result = run_case(structure, basis_name, frozen_row,
                        lambda: calculate(structure, basis_name, frozen_row, cp, np, gto, hf))
                except Exception as error:
                    result = dict(structure_id=key[0], basis=basis_name, status='failed',
                        error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                result['signature'] = signature
                atomic_json(case_path, result)
            results.append(result); add_reference_differences(results)
            report.update(passed=sum(row['status'] == 'passed' for row in results),
                failed=sum(row['status'] != 'passed' for row in results),
                completed_cases=len(results), elapsed_seconds=time.perf_counter() - start)
            atomic_json(args.output / 'progress.json', report)
            print(f'{key[0]} {basis_name}: {result["status"]} '
                  f'{result.get("energy_hartree", result.get("error"))}', flush=True)
            cp.get_default_memory_pool().free_all_blocks()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report.update(status='passed' if report['failed'] == 0 else 'failed',
        host_max_rss_bytes=int(rss if platform.system() == 'Darwin' else rss * 1024))
    atomic_json(args.output / 'comparison.json', report)
    atomic_json(args.output / 'progress.json', report)
    return int(report['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
