"""Whole r2SCAN XC at a converged PBE0 density: this is NOT r2SCAN SCF.

Compare the same production operator/provider without and with actual saved
C/occ hints. CPU/GPU use their own states; cross-device speed ratios are invalid.
"""
from __future__ import annotations
import argparse
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
from benchmarks.probes.scf_solver.profile_stages import plain, peak_rss_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'saved_state', 'saved_grid', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--device', choices=('cpu', 'gpu'), required=True)
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID') or len(os.sched_getaffinity(0)) < 64:
        parser.error('Run inside the authorized CPU64 Slurm allocation')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:64])
    os.environ.update(JAX_PLATFORMS='cpu' if args.device == 'cpu' else 'cuda,cpu',
        JAX_ENABLE_X64='1', JAX_ENABLE_COMPILATION_CACHE='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', GRADSCF_NATIVE_THREADS='64')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import jax
    import jax.numpy as jnp
    import gradscf
    from gradscf import gto, integrals
    from gradscf._memory import tree_storage_bytes
    from gradscf.integrals.basis import basis_from_molecule_spec
    from gradscf.integrals.backends.native.jk import NativeDirectBasis
    from gradscf.integrals.backends.native.workspace import bind_jk_workspace
    from gradscf.integrals.grids.blocks import AOGrid, ao_point_bytes
    from gradscf.scf.inputs.grid import prepare_scf_grid, _scf_operator_reservation
    from gradscf.scf.rks import RKSConfig, _restricted_xc_energy_and_matrix
    jax.config.update('jax_enable_x64', True)
    target, sync = jax.devices(args.device)[0], jax.block_until_ready
    if args.device == 'gpu' and len(jax.devices('gpu')) != 1:
        raise RuntimeError('Expose only the allocated GPU')
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
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
        row = dict(stage=stage, **values, **memory())
        with (args.output/'measurements.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(row), allow_nan=False)+'\n')
        print(json.dumps(plain(row), allow_nan=False), flush=True)
    config = json.loads(args.config.read_text())
    if config['xc'].lower() != 'pbe0':
        raise ValueError('Input orbitals must be from the paired PBE0 SCF')
    package = Path(gradscf.__file__).resolve().parent
    save('manifest.json', dict(job=os.environ['SLURM_JOB_ID'], host=platform.node(), command=sys.argv,
        affinity=sorted(os.sched_getaffinity(0)), device=str(target), device_kind=target.device_kind,
        jax=jax.__version__, numpy=np.__version__, dtype='float64', native_workers=64, blas_threads=1,
        script_sha256=digest(__file__), config_sha256=digest(args.config),
        state_sha256=digest(args.saved_state), grid_sha256=digest(args.saved_grid),
        source_sha256={str(p.relative_to(package)): digest(p) for folder in ('scf', 'integrals', 'dft/xc', 'tools')
                      for p in sorted((package/folder).rglob('*.py'))},
        gates=dict(energy_hartree=1e-8, potential_maxabs_hartree=1e-9),
        scope=__doc__, memory_scope='Cumulative process RSS/allocator peaks; XLA temp is an estimate'))
    with jax.default_device(target):
        start = time.perf_counter()
        spec = gto.M(**{key: config[key] for key in ('atom', 'basis', 'unit', 'cart', 'charge', 'spin')}).to_spec()
        basis = basis_from_molecule_spec(spec, basis=config['basis'])
        with np.load(args.saved_state, allow_pickle=False) as state:
            density, coeff, occ = (jnp.asarray(state[key]) for key in ('density', 'mo_coeff', 'mo_occ'))
        with np.load(args.saved_grid, allow_pickle=False) as grid:
            raw = AOGrid.from_basis(basis, jnp.asarray(grid['coords']), jnp.asarray(grid['weights']),
                                    block_size=4096, adaptive=args.device == 'cpu')
        sync((density, coeff, occ, raw))
        if raw.nao != 880 or len(spec.charges) != 96 or coeff.shape != density.shape or density.shape != (880, 880):
            raise ValueError('Expected the complete 96-atom/880-AO paired PBE0 state')
        if any(value.dtype != jnp.float64 for value in (density, coeff, occ, raw.coords, raw.weights)):
            raise ValueError('Saved state/grid must retain float64')
        nocc = int(np.count_nonzero(np.asarray(occ)))
        guard = jax.jit(lambda d, c, o: jnp.all(d == jnp.einsum('pi,i,qi->pq', c, o, c,
            precision=jax.lax.Precision.HIGHEST)) & jnp.all(o[nocc:] == 0)
            & jnp.all(jnp.isfinite(c)) & jnp.all(jnp.isfinite(o)))
        matches = bool(sync(guard(density, coeff, occ)))
        emit('basis_state_preparation', seconds=time.perf_counter()-start, exact_guard_matches=matches, nocc=nocc)
        start = time.perf_counter()
        topology, parameters = integrals.prepare_basis(spec, config['basis'], cart=config['cart'])
        direct = NativeDirectBasis(integrals.make_plan(topology), parameters)
        allowance = 32*density.nbytes
        direct = bind_jk_workspace(direct, workers=64, max_memory=config['max_memory'],
            resident_bytes=tree_storage_bytes(raw, density, coeff, occ, direct)+allowance
                +raw.block_size*ao_point_bytes(raw, deriv=1), cutoff=config['direct_scf_tol'])
        device_reserve, host_reserve = _scf_operator_reservation(direct_basis=direct,
            with_k=True, density_dtype=density.dtype, workers=64, cutoff=config['direct_scf_tol'])
        provider = sync(prepare_scf_grid(raw, max_memory=config['max_memory'], deriv=1, expected_passes=8,
            resident_bytes=tree_storage_bytes(density, coeff, occ, direct)+allowance+device_reserve,
            host_workspace_bytes=host_reserve)) if args.device == 'cpu' else raw
        emit('provider_preparation', seconds=time.perf_counter()-start, provider=type(provider).__name__,
            block_size=provider.block_size, cache_bytes=getattr(getattr(provider, 'values', None), 'nbytes', 0),
            bound_private_output_bytes=direct.private_output_bytes, host_workspace_bytes=host_reserve,
            hint_eligible=provider.nao >= 128 and provider.block_size >= 256 and 0 < nocc < provider.nao,
            orbital_branch_expected=matches and provider.nao >= 128 and provider.block_size >= 256 and 0 < nocc < provider.nao)
        cfg = RKSConfig(xc_spec='r2scan', max_memory=config['max_memory'])
        values, compiled, records = (density, coeff, occ, provider), [], []
        for hinted in (False, True):
            def action(d, c, o, g):
                hints = dict(mo_coeff=c, mo_occ=o, occupied_columns=nocc) if hinted else {}
                return _restricted_xc_energy_and_matrix(density=d, ao=jnp.zeros((0, g.nao)),
                    ao_deriv1=jnp.zeros((4, 0, g.nao)), weights=g.weights, cfg=cfg, xc_kind='MGGA', grid_ao=g, **hints)
            start = time.perf_counter(); lowered = jax.jit(action).lower(*values); lower = time.perf_counter()-start
            start = time.perf_counter(); executable = lowered.compile(); compile_time = time.perf_counter()-start
            start = time.perf_counter(); result = sync(executable(*values)); first = time.perf_counter()-start
            memory_plan = executable.memory_analysis()
            row = dict(hinted=hinted, lower_seconds=lower, compile_seconds=compile_time, first_seconds=first,
                warm_seconds=[], xla_temp_bytes=getattr(memory_plan, 'temp_size_in_bytes', None))
            emit('first_execution', **row)
            records.append(row); compiled.append(executable)
            if not hinted:
                reference = tuple(np.asarray(value) for value in result)
            else:
                observed = tuple(np.asarray(value) for value in result)
        energy_error, matrix_error = float(abs(observed[0]-reference[0])), float(np.max(abs(observed[1]-reference[1])))
        passed = bool(np.isfinite(energy_error+matrix_error) and energy_error <= 1e-8 and matrix_error <= 1e-9)
        emit('accuracy', energy_error_hartree=energy_error, potential_maxabs_error_hartree=matrix_error, passed=passed)
        if passed:
            for repeat in range(3):
                for index in (0, 1) if repeat % 2 == 0 else (1, 0):
                    start = time.perf_counter(); sync(compiled[index](*values)); elapsed = time.perf_counter()-start
                    records[index]['warm_seconds'].append(elapsed)
                    emit('warm_execution', hinted=bool(index), repeat=repeat, seconds=elapsed)
        for row in records:
            row['warm_median_seconds'] = statistics.median(row['warm_seconds']) if passed else None
        save('summary.json', dict(accuracy_pass=passed, exact_guard_matches=matches, records=records,
            energy_error_hartree=energy_error, potential_maxabs_error_hartree=matrix_error,
            whole_xc_warm_speedup=records[0]['warm_median_seconds']/records[1]['warm_median_seconds'] if passed else None,
            scope=__doc__, **memory()))
        if not passed:
            raise RuntimeError('Fixed whole-XC accuracy gate failed; warm repetitions skipped')


if __name__ == '__main__':
    main()
