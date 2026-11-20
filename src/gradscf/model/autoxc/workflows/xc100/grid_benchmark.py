"""Measure the existing fixed-density XC path without changing its quadrature."""
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np


def timed_jit(function, arguments, *, repeats=3, report=None):
    """Separate tracing, compilation and synchronized first/warm execution."""
    import jax
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError('Use at least one warm repetition')
    report = report or (lambda stage: None)
    jax.block_until_ready(arguments)
    report('trace_lower')
    start = time.perf_counter()
    lowered = jax.jit(function).lower(*arguments)
    tracing = time.perf_counter()-start
    report('compile')
    start = time.perf_counter()
    compiled = lowered.compile()
    compilation = time.perf_counter()-start

    def execute():
        start = time.perf_counter()
        result = jax.block_until_ready(compiled(*arguments))
        return result, time.perf_counter()-start

    report('first_execution')
    result, first = execute()
    warm = []
    for index in range(repeats):
        report(f'warm_{index+1}')
        result, elapsed = execute()
        warm.append(elapsed)
    return result, {'trace_lower_seconds': tracing, 'compile_seconds': compilation,
                    'first_execution_seconds': first, 'warm_seconds': warm,
                    'warm_median_seconds': statistics.median(warm)}


def compare_arrays(actual, expected, *, atol, rtol=1e-9):
    """Reject shape changes and nonfinite arrays before any allclose test."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return {'passed': False, 'max_abs_error': None, 'error': 'shape mismatch'}
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {'passed': False, 'max_abs_error': None, 'error': 'nonfinite array'}
    return {'passed': bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
            'max_abs_error': float(np.max(np.abs(actual-expected))) if actual.size else 0.,
            'atol': atol, 'rtol': rtol}


def run_point(root, identifier, block_size, *, repeats=3):
    """Run one target/block pair in a fresh Slurm GPU process.

    The original PBE matrix is held fixed across block sizes. Only the candidate
    correction and grid diagnostic are reassembled; no SCF or coefficient fit
    enters this benchmark. All calls use the existing custom_xc/WuYang paths.
    """
    import jax
    import jax.numpy as jnp
    from gradscf.model.autoxc import Candidate
    from ...history import source_digest, versions, write_json
    from .campaign import memory
    from .molecule import BoundCandidate, XC100Case, file_hash

    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Run numerical benchmarks inside the single-GPU Slurm allocation')
    if len(jax.devices()) != 1 or not jax.config.jax_enable_x64:
        raise RuntimeError('Require exactly one visible GPU and float64')
    if len(os.sched_getaffinity(0)) > int(os.environ['SLURM_CPUS_PER_TASK']):
        raise RuntimeError('CPU affinity exceeds the allocation')
    # Each point uses a fresh process and an explicit cold JAX compilation path.
    jax.config.update('jax_enable_compilation_cache', False)
    root, identifier = Path(root), str(identifier).zfill(3)
    folder = root/'results'/identifier
    folder.mkdir(parents=True, exist_ok=True)
    output = folder/f'block-{block_size}.json'
    reference = json.loads((root/'grid_benchmark_reference.json').read_text())
    candidate = Candidate.from_json(reference['candidate'])
    names = [parameter.name for parameter in candidate.parameters]
    point = {name: jnp.asarray(reference['evaluation']['parameters'][name]) for name in names}
    zeros = {name: jnp.asarray(0.) for name in names}
    started = time.perf_counter()

    def progress(stage):
        write_json(folder/f'block-{block_size}.progress.json',
                   {'system_id': identifier, 'block_size': block_size, 'stage': stage,
                    'elapsed_seconds': time.perf_counter()-started, 'time': time.time()})
        print(identifier, block_size, stage, flush=True)

    progress('prepare_cached_reference')
    case = XC100Case(root/'data/xc100', identifier, block_size=block_size)
    cache = root/'baseline-cache'/identifier
    baseline = json.loads((cache/'baseline.json').read_text())
    with np.load(cache/'pbe.npz', allow_pickle=False) as stored:
        if (str(stored['core']) != source_digest() or
                str(stored['fingerprint']) != baseline['fingerprint'] or
                baseline['system_id'] != identifier):
            raise ValueError('Original PBE cache identity differs from the frozen reference')
        case.prepare(baseline=(float(stored['energy']), stored['potential']))
    preparation = time.perf_counter()-started
    matrix = case.candidate_matrix(candidate)
    offset, offset_time = timed_jit(matrix, (zeros,), repeats=repeats,
        report=lambda stage: progress('matrix/'+stage))
    directions, direction_time = timed_jit(jax.jacfwd(matrix), (zeros,), repeats=repeats,
        report=lambda stage: progress('directions/'+stage))
    bound = BoundCandidate(case, candidate, offset, directions)
    (reward, gradient), score_time = timed_jit(jax.value_and_grad(lambda params: bound(params).reward),
        (point,), repeats=repeats, report=lambda stage: progress('score_gradient/'+stage))
    score, diagnostic_time = timed_jit(bound.diagnose, (point,), repeats=repeats,
        report=lambda stage: progress('density_diagnostic/'+stage))

    values = {field: float(getattr(score, field)) for field in
              ('value', 'delta', 'reward', 'gap', 'variational_gap', 'density_error')}
    original = reference['evaluation']['samples'][identifier]
    checks = {field: compare_arrays(value, original[field], atol=1e-8)
              for field, value in values.items()}
    checks['gradient_path_reward'] = compare_arrays(float(reward), float(score.reward), atol=1e-9)
    gradient_array = np.asarray([gradient[name] for name in names])
    arrays = {'potential': np.asarray(bound.potential(point)), 'offset': np.asarray(offset),
              'directions': np.stack([np.asarray(directions[name]) for name in names]),
              'reward_gradient': gradient_array}
    finite = bool(score.valid) and all(np.isfinite(value).all() for value in arrays.values())
    passed = finite and all(check['passed'] for check in checks.values())
    matrix_file = folder/f'block-{block_size}.npz'
    np.savez_compressed(matrix_file, **arrays)
    record = {'status': 'ok' if passed else 'numerical_mismatch',
        'system_id': identifier, 'symbols': case.symbols, 'block_size': block_size,
        'nao': case.target.overlap.shape[0], 'cartesian_nao': case.grid.nao,
        'grid_points': case.grid.ngrids, 'grid_level': case.metadata['metrics']['grid_level'],
        'grid_blocks': (case.grid.ngrids+block_size-1)//block_size,
        'candidate': candidate.to_dict(), 'candidate_digest': candidate.digest,
        'parameters': reference['evaluation']['parameters'], 'parameter_order': names,
        'preparation_seconds': preparation,
        'timings': {'matrix': offset_time, 'directions': direction_time,
                    'score_gradient': score_time, 'density_diagnostic': diagnostic_time},
        'score': values, 'original_round2_checks': checks, 'finite': finite,
        'arrays': str(matrix_file), 'elapsed_seconds': time.perf_counter()-started,
        'source': source_digest(), 'benchmark_source': file_hash(__file__),
        'matrix_source': file_hash(Path(__file__).with_name('molecule.py')),
        'archive_sha256': case.archive_hash, 'metadata_sha256': case.metadata_hash,
        'baseline_cache_sha256': file_hash(cache/'pbe.npz'),
        'device': jax.devices()[0].device_kind, 'backend': jax.default_backend(),
        'float64': bool(jax.config.jax_enable_x64), 'versions': versions(),
        'job_id': os.environ['SLURM_JOB_ID'], 'node': os.environ['SLURMD_NODENAME'],
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'requested_memory_mb': os.environ.get('SLURM_MEM_PER_NODE'),
        'memory_kind': 'process RSS peak and JAX live-allocation peak; fresh process per point',
        **memory()}
    write_json(output, record)
    progress('complete')
    return record
