"""Fixed-density PBE0 semilocal XC on the saved GradSCF 96-atom input.

Raw stream versus screened local streams, optionally versus an admitted CPU
AO cache. No SCF, J/K, quadrature regeneration, or PySCF calculation is run.
The fixed gates are |dE| <= 1e-8 Ha and max|dVxc| <= 1e-9 Ha.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'saved_density', 'saved_grid', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--device', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--cpu-cache', action='store_true', help='Also materialize a CPU cache only if the production budget admits it')
    parser.add_argument('--selected-only', action='store_true', help='Revalidate only the retained 4096/1e-14 local policy')
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        parser.error('Run inside the authorized Slurm allocation')
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 64:
        parser.error('CPU64 is required')
    os.sched_setaffinity(0, affinity[:64])
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
    from gradscf.integrals.backends.native.jk import NativeDirectBasis, _DIRECT_JK_PRIVATE_MAX_BATCH
    from gradscf.integrals.backends.native.workspace import bind_jk_workspace
    from gradscf.integrals.grids.blocks import AOGrid, _LocalAOGrid, ao_point_bytes, grid_slice
    from gradscf.integrals.grids.locality import ao_block_bounds, select_local_aos
    from gradscf.scf.inputs.grid import prepare_scf_grid, _scf_operator_reservation
    from gradscf.scf.rks import RKSConfig, _restricted_xc_energy_and_matrix

    jax.config.update('jax_enable_x64', True)
    target = jax.devices(args.device)[0]
    if args.device == 'gpu' and len(jax.devices('gpu')) != 1:
        raise RuntimeError('Expose only the allocated GPU')
    sync = jax.block_until_ready
    def allocator():
        try:
            stats = target.memory_stats() or {}
        except (RuntimeError, NotImplementedError):
            stats = {}
        return {key: stats.get(key) for key in ('bytes_in_use', 'peak_bytes_in_use', 'bytes_limit')}

    def save(name, value):
        (args.output/name).write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')

    def emit(stage, **values):
        row = dict(stage=stage, peak_rss_bytes=peak_rss_bytes(), gpu_allocator=allocator(), **values)
        with (args.output/'measurements.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(row), allow_nan=False)+'\n')
        print(json.dumps(plain(row), allow_nan=False), flush=True)

    config = json.loads(args.config.read_text())
    if config['xc'].lower() != 'pbe0' or config['representation'] != 'direct':
        raise ValueError('Use the frozen GradSCF PBE0/direct config')
    package = Path(gradscf.__file__).resolve().parent
    cpu_model = next((line.split(':', 1)[1].strip() for line in
        Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), platform.processor())
    manifest = dict(job=os.environ['SLURM_JOB_ID'], host=platform.node(), command=sys.argv, cpu_model=cpu_model,
        memory_allocation_mb=os.getenv('SLURM_MEM_PER_NODE'), native_workers=64, blas_threads=1,
        device=str(target), device_kind=target.device_kind, affinity=sorted(os.sched_getaffinity(0)),
        jax=jax.__version__, numpy=np.__version__, python=sys.executable, dtype='float64',
        script_sha256=digest(__file__), config_sha256=digest(args.config),
        saved_density_sha256=digest(args.saved_density), saved_grid_sha256=digest(args.saved_grid),
        module=gradscf.__file__, config=config, cpu_cache_requested=args.cpu_cache,
        selected_only=args.selected_only,
        gates=dict(energy_abs_hartree=1e-8, potential_maxabs_hartree=1e-9),
        source_sha256={str(path.relative_to(package)): digest(path)
            for folder in ('scf', 'integrals', 'dft/xc', 'tools')
            for path in sorted((package/folder).rglob('*.py'))},
        memory_scope='XLA temp is an estimate; RSS and allocator peak are cumulative process high-water, not per-variant peaks.',
        scope='Same saved GradSCF density/grid; semilocal operator only, excludes exact exchange and SCF.')
    save('manifest.json', manifest)
    with jax.default_device(target):
        start = time.perf_counter()
        spec = gto.M(**{key: config[key] for key in ('atom', 'basis', 'unit', 'charge', 'spin', 'cart')}).to_spec()
        basis = basis_from_molecule_spec(spec, basis=config['basis'])
        with np.load(args.saved_grid, allow_pickle=False) as grid:
            coords, weights = jnp.asarray(grid['coords']), jnp.asarray(grid['weights'])
        with np.load(args.saved_density, allow_pickle=False) as state:
            density = jnp.asarray(state['density'])
        raw = sync(AOGrid.from_basis(basis, coords, weights, block_size=4096, adaptive=False))
        sync(density)
        if len(spec.charges) != 96 or raw.nao != 880 or density.shape != (880, 880):
            raise ValueError('Saved input is not the expected 96-atom/880-AO calculation')
        if any(value.dtype != jnp.float64 for value in (density, coords, weights)):
            raise ValueError('Saved density/grid must retain float64')
        if not all(np.isfinite(np.asarray(value)).all() for value in (density, coords, weights)):
            raise ValueError('Saved density/grid contains nonfinite values')
        emit('basis_and_saved_grid_preparation', seconds=time.perf_counter()-start,
             nao=raw.nao, ngrids=raw.ngrids, real_zero_weights=int(np.count_nonzero(np.asarray(weights)==0)),
             density_source='saved validated GradSCF state')
        # Reserve the configured native direct-J/K host capacity conservatively,
        # although this probe executes no J/K and allocates no native worker scratch.
        start = time.perf_counter()
        topology, parameters = integrals.prepare_basis(spec, config['basis'], cart=config['cart'])
        direct = NativeDirectBasis(integrals.make_plan(topology), parameters)
        # Match SCF binding rather than the standalone provider's default quota.
        # Reserve raw parameters, the density, the current 32-matrix SCF allowance
        # and one full-AO tile before the native helper chooses its private quota.
        scf_matrix_allowance = 32*density.nbytes
        ao_workspace = raw.block_size*ao_point_bytes(raw, deriv=1)
        binding_resident = tree_storage_bytes(raw, density, direct)+scf_matrix_allowance+ao_workspace
        direct = bind_jk_workspace(direct, workers=64, max_memory=config['max_memory'],
            resident_bytes=binding_resident, cutoff=config['direct_scf_tol'])
        worker_count = min(64, len(topology.angular_momenta))
        full_private_quota = (worker_count*2*_DIRECT_JK_PRIVATE_MAX_BATCH*raw.nao**2*8
                              if worker_count > 1 else 0)
        reservation = _scf_operator_reservation(direct_basis=direct, with_k=True,
            density_dtype=density.dtype, workers=64, cutoff=config['direct_scf_tol'])
        if reservation is None:
            raise RuntimeError('Native workspace could not be estimated; refusing optional cache')
        device_reserve, host_reserve = reservation
        resident = tree_storage_bytes(density, direct)+scf_matrix_allowance+2*density.nbytes+device_reserve
        sync(parameters)
        emit('cache_budget', seconds=time.perf_counter()-start, max_memory_decimal_mb=config['max_memory'], resident_bytes=resident,
             host_workspace_bytes=host_reserve, native_scratch_actually_allocated=False,
             binding_resident_bytes=binding_resident, scf_matrix_allowance_bytes=scf_matrix_allowance,
             binding_ao_workspace_bytes=ao_workspace, bound_private_output_bytes=direct.private_output_bytes,
             full_private_output_bytes=full_private_quota,
             full_private_quota_admitted=direct.private_output_bytes == full_private_quota)
        cfg = RKSConfig(xc_spec='pbe0', max_memory=config['max_memory'])
        def action(dm, provider):
            return _restricted_xc_energy_and_matrix(density=dm,
                ao=jnp.zeros((0, provider.nao), dtype=dm.dtype),
                ao_deriv1=jnp.zeros((4, 0, provider.nao), dtype=dm.dtype),
                weights=provider.weights, cfg=cfg, xc_kind='GGA', grid_ao=provider)

        summaries = []
        reference = None
        def measure(name, provider, prep_seconds):
            nonlocal reference
            sync((density, provider))
            start = time.perf_counter()
            lowered = jax.jit(action).lower(density, provider)
            lower_seconds = time.perf_counter()-start
            start = time.perf_counter()
            executable = lowered.compile()
            compile_seconds = time.perf_counter()-start
            memory = executable.memory_analysis()
            emit('compile', variant=name, lower_seconds=lower_seconds, compile_seconds=compile_seconds,
                xla_memory=None if memory is None else {key: getattr(memory, key, None) for key in
                ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')})
            start = time.perf_counter()
            value = sync(executable(density, provider))
            first_seconds = time.perf_counter()-start
            host_value = tuple(np.asarray(part) for part in value)
            finite = all(np.isfinite(part).all() for part in host_value)
            if reference is None:
                if not finite:
                    raise RuntimeError('Raw baseline is nonfinite')
                reference = host_value
                np.savez(args.output/'baseline_xc.npz', energy=reference[0], vxc=reference[1])
            energy_error = float(abs(host_value[0]-reference[0]))
            matrix_error = float(np.max(np.abs(host_value[1]-reference[1])))
            passed = finite and energy_error <= 1e-8 and matrix_error <= 1e-9
            emit('first_execution', variant=name, seconds=first_seconds,
                 energy_hartree=float(host_value[0]), energy_error_hartree=energy_error,
                 potential_maxabs_error_hartree=matrix_error, accuracy_pass=bool(passed))
            times = []
            if passed:
                for repeat in range(3):
                    start = time.perf_counter()
                    sync(executable(density, provider))
                    times.append(time.perf_counter()-start)
                    emit('warm_execution', variant=name, repeat=repeat, seconds=times[-1])
            else:
                emit('warm_skipped', variant=name, reason='Fixed accuracy gate failed; no tolerance relaxation')
            row = dict(variant=name, provider_type=type(provider).__name__, block_size=provider.block_size,
                preparation_seconds=prep_seconds, lower_seconds=lower_seconds, compile_seconds=compile_seconds,
                first_seconds=first_seconds, warm_seconds=times,
                warm_median_seconds=statistics.median(times) if times else None,
                accuracy_pass=bool(passed), energy_error_hartree=energy_error,
                potential_maxabs_error_hartree=matrix_error,
                estimated_dense_fallback_ao_workspace_bytes=provider.block_size*ao_point_bytes(raw, deriv=1))
            summaries.append(row)
            save('summary.json', dict(measurements=summaries, complete=False))

        measure('raw_stream_4096', raw, 0.)
        variants = ((4096, 1e-14),) if args.selected_only else ((4096, 1e-14), (4096, 1e-12), (2048, 1e-14))
        for tile, cutoff in variants:
            name = f'local_{tile}_cap512_tol{cutoff:.0e}'
            start = time.perf_counter()
            local = sync(prepare_scf_grid(replace(raw, block_size=tile, adaptive=True),
                max_memory=config['max_memory'], resident_bytes=resident, host_workspace_bytes=host_reserve,
                deriv=1, expected_passes=4, screen_tol=cutoff))
            prep_seconds = time.perf_counter()-start
            if not isinstance(local, _LocalAOGrid) or local.capacity != 512:
                raise RuntimeError('Production locality planner did not produce capacity 512')
            emit('local_order_preparation', variant=name, seconds=prep_seconds,
                 cutoff=cutoff, capacity=local.capacity)
            def support_probe(provider):
                ordered_coords = provider.raw.coords[provider.order]
                ordered_weights = provider.raw.weights[provider.order]
                def body(_, index):
                    points = grid_slice(ordered_coords, index*provider.block_size, provider.block_size)
                    weights = grid_slice(ordered_weights, index*provider.block_size, provider.block_size)
                    point_mask = index*provider.block_size + jnp.arange(provider.block_size) < provider.ngrids
                    bounds = ao_block_bounds(provider.raw, points, weights, deriv=1,
                        coefficient_response=True, point_mask=point_mask)
                    count = jnp.sum((bounds > provider.cutoff) | ~jnp.isfinite(bounds))
                    _, _, overflow = select_local_aos(bounds, cutoff=provider.cutoff, capacity=provider.capacity)
                    return None, (count, overflow)
                return jax.lax.scan(body, None, jnp.arange(
                    (provider.ngrids+provider.block_size-1)//provider.block_size))[1]
            start = time.perf_counter()
            counts, overflow = sync(jax.jit(support_probe)(local))
            counts, overflow = np.asarray(counts), np.asarray(overflow)
            np.savez(args.output/(name+'_support.npz'), counts=counts, overflow=overflow)
            widths = np.where(overflow, raw.nao,
                np.where(counts <= local.capacity//2, local.capacity//2, local.capacity)
                if local.capacity >= 256 else local.capacity)
            emit('support_metadata_probe', variant=name, seconds=time.perf_counter()-start,
                blocks=len(counts), count_min=int(counts.min()), count_max=int(counts.max()),
                count_median=float(np.median(counts)), overflow_blocks=int(overflow.sum()),
                planned_column_sum=int(widths.sum()),
                bucket_blocks={str(int(width)):int((widths==width).sum()) for width in np.unique(widths)},
                scope='Independent bounds/selection metadata probe; not an actual AO evaluation counter')
            measure(name, local, prep_seconds)
        if args.cpu_cache and args.device == 'cpu':
            start = time.perf_counter()
            cached = sync(prepare_scf_grid(replace(raw, adaptive=True), max_memory=config['max_memory'],
                resident_bytes=resident, host_workspace_bytes=host_reserve, deriv=1, expected_passes=4))
            seconds = time.perf_counter()-start
            admitted = hasattr(cached, 'values')
            emit('cpu_cache_preparation', seconds=seconds, admitted=admitted,
                 cache_bytes=cached.values.nbytes if admitted else 0)
            if admitted:
                measure('admitted_cpu_cache', cached, seconds)
        else:
            emit('cpu_cache_skipped', reason='CPU cache not requested' if args.device == 'cpu' else 'GPU cache is not forced')
        save('summary.json', dict(complete=True, fixed_density_accuracy_pass=all(row['accuracy_pass'] for row in summaries),
            measurements=summaries, peak_rss_bytes=peak_rss_bytes(), gpu_allocator=allocator(),
            scope='Fixed-density semilocal XC only; no full SCF speedup claim.'))


if __name__ == '__main__':
    main()
