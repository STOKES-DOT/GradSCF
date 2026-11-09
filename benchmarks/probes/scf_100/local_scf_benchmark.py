"""Complete GradSCF PBE0/direct SCF acceptance, cold plus fresh-input warm.

Run only in the authorized CPU64/single-GPU Slurm allocation. Root owns the
external 6000 s / 200 GB RSS guard. No solver callback or reference export.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.probes.scf_solver.profile_stages import plain, peak_rss_bytes, synchronize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'baseline_summary', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--device', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--screen-tol', type=float, default=1e-14)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID') or len(os.sched_getaffinity(0)) < 64:
        parser.error('Run in the authorized CPU64 Slurm allocation')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:64])
    os.environ.update(JAX_PLATFORMS='cpu' if args.device == 'cpu' else 'cuda,cpu',
        JAX_ENABLE_X64='1', JAX_ENABLE_COMPILATION_CACHE='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', GRADSCF_NATIVE_THREADS='64')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    args.output.mkdir(parents=True, exist_ok=False)

    import jax
    import numpy as np
    import gradscf
    expected_source = os.getenv('GRADSCF_EXPECTED_SRC')
    if expected_source and Path(gradscf.__file__).resolve().parents[1] != Path(expected_source).resolve():
        raise RuntimeError('The imported GradSCF source does not match this benchmark variant')
    from gradscf import gto
    from gradscf.integrals._native import register_integrals
    from gradscf.scf.inputs import assembly
    from gradscf.scf.rks import RKSConfig, run_rks_from_integrals
    from gradscf.solvers.scf import orbital_gradient_norm
    jax.config.update('jax_enable_x64', True)
    cpu, target = jax.devices('cpu')[0], jax.devices(args.device)[0]
    if args.device == 'gpu' and len(jax.devices('gpu')) != 1:
        raise RuntimeError('Expose only the allocated GPU')
    if not np.isfinite(args.screen_tol) or args.screen_tol < 0:
        raise ValueError('screen-tol must be finite and nonnegative')
    original = json.loads(args.config.read_text())
    config = dict(original)
    reference = json.loads(args.baseline_summary.read_text())
    if (config['xc'].lower(), config['representation'], config['cart'], config['charge'], config['spin'],
            config['direct_scf_tol'], config['max_memory']) != ('pbe0', 'direct', True, 0, 0, 1e-12, 180000):
        raise ValueError('Frozen PBE0/direct/cart/CPU64/cutoff1e-12/max_memory180000 controls differ')
    if args.smoke:
        config.update(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587', basis='sto-3g', grid_level=0, natom=3,
            id='water-smoke',case_id='water-sto3g-pbe0-locality-smoke',geometry_sha256=None,protocol_sha256=None)
    elif config['basis'].lower() != 'def2-svp' or config['grid_level'] != 3:
        raise ValueError('Large case requires Cartesian def2-SVP and grid level 3')
    cfg = RKSConfig(xc_spec='pbe0', max_cycle=config['max_cycle'], conv_tol=config['conv_tol'],
        conv_tol_density=config['density_tol'], conv_tol_grad=config['gradient_tol'],
        jk_backend='direct', direct_scf_tol=config['direct_scf_tol'], direct_scf_workers=64,
        max_memory=config['max_memory'],
        **({'grid_screen_tol': args.screen_tol} if args.screen_tol > 0 else {}))

    def save(name, value):
        (args.output/name).write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')
    def memory():
        try:
            stats = target.memory_stats() or {}
        except (RuntimeError, NotImplementedError):
            stats = {}
        return dict(peak_rss_bytes=peak_rss_bytes(), gpu_allocator={key: stats.get(key) for key in
            ('bytes_in_use', 'peak_bytes_in_use', 'bytes_limit')})
    def emit(stage, **values):
        row = dict(stage=stage, **memory())
        row.update(values)
        with (args.output/'measurements.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(row), allow_nan=False)+'\n')
        print(json.dumps(plain(row), allow_nan=False), flush=True)
    save('input.json', original)
    save('effective_input.json', config)
    package = Path(gradscf.__file__).resolve().parent
    native = Path(register_integrals()._name).resolve()
    save('runtime.json', dict(command=sys.argv, job=os.environ['SLURM_JOB_ID'], host=platform.node(),
        affinity=sorted(os.sched_getaffinity(0)), device=str(target), device_kind=target.device_kind,
        native_workers=64, blas_threads=1, memory_allocation_mb=os.getenv('SLURM_MEM_PER_NODE'),
        python=sys.executable, jax=jax.__version__, numpy=np.__version__, module=gradscf.__file__,
        dtype='float64', smoke=args.smoke, rks_config=asdict(cfg), script_sha256=digest(__file__),
        config_sha256=digest(args.config), baseline_sha256=digest(args.baseline_summary),
        effective_input_sha256=digest(args.output/'effective_input.json'), native_sha256=digest(native),
        source_sha256={str(path.relative_to(package)): digest(path) for folder in
            ('scf', 'solvers/scf', 'integrals', 'dft/xc', 'tools') for path in sorted((package/folder).rglob('*.py'))},
        cache_policy='Fresh molecule/input and atom guess each call; normal same-geometry grid cache and JIT retained; no converged-state reuse',
        memory_scope='RSS and allocator peaks are cumulative process high-water',
        saved_state_scope='PBE0 orbitals; later MGGA fixed-density use is not an MGGA SCF result'))

    def solve(device, repeat):
        whole = start = time.perf_counter()
        with jax.default_device(cpu):
            mol = gto.M(**{key: config[key] for key in ('atom', 'basis', 'unit', 'charge', 'spin', 'cart')})
            spec = mol.to_spec()
            if not args.smoke and Counter(spec.symbols) != {'C': 40, 'H': 56}:
                raise ValueError('Large geometry must be C40H56')
            inputs = assembly.build_rks_integral_inputs(atom=spec, basis=mol.basis, config=cfg,
                xc_spec='pbe0', grids_level=config['grid_level'], unit=mol.unit, cart=mol.cart,
                charge=mol.charge, spin=mol.spin, init_guess='atom', include_dipole_integrals=False)
            kwargs = inputs.as_rks_kwargs()
            synchronize(kwargs, jax)
        prepare = time.perf_counter()-start
        n = inputs.overlap.shape[0]
        if inputs.overlap.dtype != np.dtype('float64'):
            raise RuntimeError('SCF input dtype must remain float64')
        if not args.smoke and (n, inputs.grid_weights.size, inputs.nelectron) != (880, 1113968, 296):
            raise ValueError('Large AO/grid/electron dimensions differ from the frozen case')
        if any(kwargs[key] is not None for key in ('eri', 'eri_pair_matrix', 'df_factors')):
            raise RuntimeError('Direct SCF must not create AO ERIs or RI factors')
        emit('prepared', repeat=repeat, seconds=prepare, nao=n, ngrid=int(inputs.grid_weights.size))
        start = time.perf_counter()
        with jax.default_device(device):
            staged = jax.tree.map(lambda value: jax.device_put(value, device)
                if isinstance(value, (jax.Array, np.ndarray)) else value, kwargs)
            synchronize(staged, jax)
        stage = time.perf_counter()-start
        if not isinstance(staged['nelectron'], int):
            raise RuntimeError('Structural nelectron must remain a Python integer')
        emit('staged', repeat=repeat, seconds=stage)
        start = time.perf_counter()
        with jax.default_device(device):
            result = run_rks_from_integrals(**staged, config=cfg)
            synchronize(result, jax)
        solver, total = time.perf_counter()-start, time.perf_counter()-whole
        with jax.default_device(device):
            residual = float(orbital_gradient_norm(result.fock_matrix, result.mo_coeff, result.mo_occ))
        finite = all(np.isfinite(np.asarray(value)).all() for value in
            (result.total_energy, residual, result.mo_coeff, result.mo_occ, result.density_matrix, result.fock_matrix))
        full_space = result.mo_coeff.shape == (n, n)
        return result, dict(repeat=repeat, cpu_prepare_seconds=prepare, stage_seconds=stage,
            solver_seconds=solver, total_seconds=total, energy_hartree=float(result.total_energy),
            orbital_residual_hartree=residual, finite=bool(finite), full_physical_space=full_space,
            converged=bool(result.converged), cycles=int(result.cycles), nao=n,
            ngrid=int(inputs.grid_weights.size), nelectron=inputs.nelectron,
            local_pass=bool(result.converged and finite and full_space and residual < 1e-7), **memory())

    baseline = None
    reference_kind = 'saved accepted GradSCF PBE0'
    if not args.smoke:
        if not reference.get('scientific_pass') or reference.get('engine', 'gradscf') != 'gradscf':
            raise ValueError('Large case needs an accepted GradSCF baseline summary')
        baseline = float(reference['energy_hartree'])
    elif reference.get('smoke') and reference.get('device') == 'cpu' and reference.get('scientific_pass'):
        baseline, reference_kind = float(reference['energy_hartree']), 'saved CPU water smoke'
    elif args.device == 'gpu':
        cpu_result, cpu_row = solve(cpu, 'cpu_smoke_reference')
        if not cpu_row['local_pass']:
            raise RuntimeError('CPU water reference failed convergence')
        baseline, reference_kind = cpu_row['energy_hartree'], 'independent CPU water smoke in this process'
        emit('cpu_smoke_reference', **cpu_row)
        del cpu_result
    rows = []
    for repeat in range(2 if args.smoke else 4):
        result, row = solve(target, repeat)
        if baseline is None and args.smoke and args.device == 'cpu' and row['local_pass']:
            baseline, reference_kind = row['energy_hartree'], 'first accepted CPU water smoke'
        delta = None if baseline is None else row['energy_hartree']-baseline
        row.update(energy_delta_hartree=delta, reference_energy_hartree=baseline,
            reference_kind=reference_kind, scientific_pass=row['local_pass'] and delta is not None and abs(delta) < 1e-7,
            cache_state='cold' if repeat == 0 else 'fresh-input warm')
        rows.append(row)
        emit('scf_result', **row)
        save('summary.json', dict(smoke=args.smoke, device=args.device, engine='gradscf',
            energy_hartree=row['energy_hartree'], scientific_pass=len(rows)==(2 if args.smoke else 4)
            and all(item['scientific_pass'] for item in rows), records=rows,
            warm_total_median_seconds=statistics.median(item['total_seconds'] for item in rows[1:]) if repeat else None))
        if not row['scientific_pass']:
            raise RuntimeError('Complete SCF acceptance failed; remaining repeats skipped')
        if repeat == (1 if args.smoke else 3):
            np.savez(args.output/'final_state.npz', density=np.asarray(result.density_matrix),
                mo_coeff=np.asarray(result.mo_coeff), mo_occ=np.asarray(result.mo_occ),
                energy=np.asarray(result.total_energy))
        del result
        gc.collect()


if __name__ == '__main__':
    main()
