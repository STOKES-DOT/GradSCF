"""Validate fixed-exponent MACE contractions against reconverged RHF energies.

Prepare native DF caches on CPU, then validate on CUDA in a separate process.
Each molecule has an independent seeded model: this is a gradient/optimizer
pilot, not transferable multi-molecule training. No four-index ERI is formed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform as host_platform
import socket
import time
import traceback


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scientific_source_hash(root):
    digest = hashlib.sha256()
    package = Path(root) / 'src' / 'gradscf'
    for path in sorted(package.rglob('*')):
        if path.is_file() and path.suffix in {'.py', '.json', '.dat'}:
            digest.update(str(path.relative_to(package)).encode())
            digest.update(file_hash(path).encode())
    return digest.hexdigest()


def verify_source_imports(root, *, importer=None):
    """Reject installed or setup copies even if they report the same version."""
    if importer is None:
        from importlib import import_module
        importer = import_module
    package = Path(root).resolve() / 'src' / 'gradscf'
    sources = {'gradscf': '__init__.py', 'gradscf.scf.rks': 'scf/rks.py',
               'gradscf.solvers.nonlinear.fixed_point': 'solvers/nonlinear/fixed_point.py'}
    confirmed = {}
    for name, relative in sources.items():
        imported = importer(name)
        actual = Path(imported.__file__).resolve()
        expected = (package / relative).resolve()
        if actual != expected:
            raise RuntimeError(f'Source mismatch for {name}: imported {actual}; '
                               f'expected {expected}. Set PYTHONPATH after activating the environment.')
        confirmed[name] = dict(path=str(actual), sha256=file_hash(actual))
    return confirmed


def atomic_json(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def read_structures(path, *, ids=None, limit=None):
    """Keep source order and fail explicitly on malformed/out-of-scope input."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    names = [row['structure_id'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate structure IDs in input.')
    if ids is not None:
        unknown = sorted(set(ids) - set(names))
        if unknown:
            raise ValueError(f'Unknown structure IDs: {unknown}')
        rows = [row for row in rows if row['structure_id'] in set(ids)]
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive.')
        rows = rows[:limit]
    if not rows:
        raise ValueError('No structures selected.')
    for row in rows:
        if row.get('charge') != 0 or row.get('spin') != 0:
            raise ValueError(f"{row['structure_id']}: pilot requires neutral spin=0 structures.")
        symbols = row['symbols']; coordinates = row['coords_angstrom']
        if not symbols or any(symbol not in {'H', 'C', 'N', 'O', 'F'} for symbol in symbols):
            raise ValueError(f"{row['structure_id']}: pilot supports CHNOF only.")
        if len(coordinates) != len(symbols) or any(
            len(point) != 3 or not all(math.isfinite(float(x)) for x in point)
            for point in coordinates
        ):
            raise ValueError(f"{row['structure_id']}: invalid coordinates.")
        row['name'] = row['structure_id']
    return rows


def record_directory(root, identifier):
    # Hash prevents traversal and collisions caused by replacing '/' with '_'.
    return Path(root) / hashlib.sha256(identifier.encode()).hexdigest()[:16]


def run_suite(rows, output_dir, config, operation, *, resume):
    """Persist each outcome before continuing; failures yield a nonzero status."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    summary = dict(config=config, passed=0, failed=0, exit_code=0, results=results)
    for row in rows:
        directory = record_directory(output_dir, row['structure_id'])
        directory.mkdir(parents=True, exist_ok=True)
        signature = hashlib.sha256(json.dumps(dict(structure=row, config=config),
            sort_keys=True, allow_nan=False).encode()).hexdigest()
        path = directory / 'result.json'
        previous = json.loads(path.read_text()) if resume and path.is_file() else None
        cache_valid = True
        if previous and previous.get('cache_path'):
            cached_path = Path(previous['cache_path'])
            cache_valid = (cached_path.is_file()
                           and file_hash(cached_path) == previous.get('cache_sha256'))
        if (previous and previous.get('signature') == signature
                and previous.get('status') == 'passed' and cache_valid):
            result = previous
            print(f"Resume {row['structure_id']}: passed", flush=True)
        else:
            start = time.perf_counter()
            result = dict(structure_id=row['structure_id'], signature=signature,
                          status='running', output_dir=str(directory))
            atomic_json(path, result)
            # A stale partial report must not leak into a rerun with changed configuration.
            atomic_json(directory / 'progress.json', {})
            try:
                completed = operation(row, directory)
                json.dumps(completed, allow_nan=False)
                result.update(completed)
                result['status'] = 'passed'
            except Exception as error:
                progress = json.loads((directory / 'progress.json').read_text())
                result.update(progress, status='failed', error=str(error),
                              error_type=type(error).__name__, traceback=traceback.format_exc())
            result['elapsed_seconds'] = time.perf_counter() - start
            atomic_json(path, result)
            print(f"{row['structure_id']}: {result['status']} ({result['elapsed_seconds']:.1f} s)", flush=True)
        results.append(result)
        summary['passed'] = sum(value['status'] == 'passed' for value in results)
        summary['failed'] = sum(value['status'] == 'failed' for value in results)
        summary['exit_code'] = int(summary['failed'] > 0)
        atomic_json(output_dir / 'summary.json', summary)
    return summary


def numerical_runtime(platform):
    # Set before importing the existing experiment, whose default is CPU.
    os.environ['JAX_PLATFORMS'] = platform
    os.environ['JAX_ENABLE_X64'] = '1'
    import jax
    jax.config.update('jax_enable_x64', True)
    verify_source_imports(Path(__file__).resolve().parents[1])
    import importlib.util
    source = Path(__file__).with_name('optimize_methane_nnao.py')
    spec = importlib.util.spec_from_file_location('nnao_rhf_experiment', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    expected = 'gpu' if platform == 'cuda' else 'cpu'
    if any(device.platform != expected for device in jax.devices()):
        raise RuntimeError(f'Expected {expected}, got {jax.devices()}')
    return jax, module.MethaneRHF


def experiment_metadata(experiment, jax):
    return dict(method='RHF', basis='szp663_direct', core_primitives=6,
        jk_backend='df', auxbasis=experiment.auxbasis, dtype=str(experiment.ps.dtype),
        jax_version=jax.__version__, devices=[str(device) for device in jax.devices()],
        backend=jax.default_backend(), device_kinds=[device.device_kind for device in jax.devices()],
        hostname=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        imported_sources=verify_source_imports(Path(__file__).resolve().parents[1]),
        nao=int(experiment.layout.topology.nao),
        primitive_nao=int(experiment.primitive_topology.nao),
        nelectron=int(experiment.nelectron), df_shape=list(experiment.rep.shape),
        cache_storage=experiment.cache_storage,
        df_bytes=int(experiment.rep.size * experiment.rep.dtype.itemsize),
        integral_signature=experiment.integral_signature,
        gradient='converged RHF density fixed point, implicit backward',
        scf_config=dict(max_cycle=150, conv_tol=1e-12,
                        conv_tol_density=1e-10, conv_tol_grad=1e-9),
        implicit_config=dict(tolerance=experiment.implicit_tolerance, max_iter=100, restart=40),
        scf_rescue_level_shift=experiment.scf_rescue_level_shift,
        implicit_diagnostics='Checked GMRES residual failures produce NaN; finite gradients required. '
            'The existing API does not expose the numerical adjoint residual.')


def cache_location(cache_dir, row):
    return record_directory(cache_dir, row['structure_id']) / 'integrals.npz'


def prepare_cache(row, directory, *, cache_dir, auxbasis, jax, experiment_class,
                  implicit_tolerance=1e-9, scf_rescue_level_shift=0.0):
    experiment = experiment_class(geometry=row, basis_family='szp663_direct',
        core_primitives=6, jk_backend='df', auxbasis=auxbasis,
        implicit_tolerance=implicit_tolerance, scf_rescue_level_shift=scf_rescue_level_shift)
    path = cache_location(cache_dir, row); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('integrals.tmp.npz')
    experiment.write_integral_cache(temporary)
    os.replace(temporary, path)
    result = experiment_metadata(experiment, jax)
    from gradscf.integrals import _native
    suffix = '.dylib' if host_platform.system() == 'Darwin' else '.so'
    library = Path(_native.__file__).parent / 'lib' / ('libgradscf_integrals' + suffix)
    result.update(cache_path=str(path), cache_sha256=file_hash(path),
                  structure_id=row['structure_id'], mace_ran=False, adam_ran=False,
                  native_library_sha256=file_hash(library))
    atomic_json(path.with_suffix('.json'), result)
    jax.clear_caches()
    return result


def directional_check(evaluate, point, gradient, direction, *, step, atol, rtol):
    """Fourth-order finite difference; every evaluation reconverges SCF."""
    import numpy as np
    point = np.asarray(point); direction = np.asarray(direction)
    values = {str(scale): float(evaluate(point + scale * step * direction)[0])
              for scale in (1, -1, 2, -2)}
    finite_difference = (8 * (values['1'] - values['-1'])
                         - values['2'] + values['-2']) / (12 * step)
    analytic = float(np.sum(np.asarray(gradient) * direction))
    result = dict(analytic=analytic, finite_difference=finite_difference,
        absolute_error=abs(analytic - finite_difference), step=step, atol=atol, rtol=rtol,
        perturbed_energies_hartree=values,
        passed=bool(np.isclose(analytic, finite_difference, atol=atol, rtol=rtol)))
    return result


def validate(row, directory, *, cache_dir, auxbasis, jax, experiment_class,
             seed, adam_steps, learning_rate, fd_step, fd_atol, fd_rtol,
             implicit_tolerance=1e-9, cache_storage='device', scf_rescue_level_shift=0.0):
    import numpy as np
    import jax.numpy as jnp
    path = cache_location(cache_dir, row)
    cache_record = json.loads(path.with_suffix('.json').read_text())
    if file_hash(path) != cache_record['cache_sha256']:
        raise ValueError('DF cache SHA-256 differs from the CPU-prepared manifest.')
    experiment = experiment_class(geometry=row, basis_family='szp663_direct',
        core_primitives=6, jk_backend='df', auxbasis=auxbasis, integral_cache=path,
        implicit_tolerance=implicit_tolerance, cache_storage=cache_storage,
        scf_rescue_level_shift=scf_rescue_level_shift)
    report = experiment_metadata(experiment, jax)
    report.update(seed=seed, cache_path=str(path), cache_sha256=cache_record['cache_sha256'],
        cache_preparation=cache_record, mace_ran=False, adam_ran=False,
        training_scope='independent model per structure; no transferable training')

    def persist():
        atomic_json(directory / 'progress.json', report)

    persist()
    outputs = experiment.layout.reference_outputs()
    energy, gradient, info = experiment.evaluate(outputs)
    report.update(initial_energy_hartree=energy, initial_scf=info,
        coefficient_gradient_norm=float(np.linalg.norm(gradient)),
        implicit_gradient_finite=bool(np.isfinite(gradient).all()))
    mask = np.zeros(outputs.shape)
    for atom, slot, exponents in zip(experiment.layout.shell_atoms,
                                    experiment.layout.slots, experiment.layout.parameters.exponents):
        if slot >= 0:
            mask[atom, slot, :len(exponents)] = 1.
    direction = np.random.default_rng(seed).normal(size=outputs.shape) * mask
    direction /= np.linalg.norm(direction)
    coefficient_fd = directional_check(experiment.evaluate_value, outputs, gradient, direction,
        step=fd_step, atol=fd_atol, rtol=fd_rtol)
    report['coefficient_directional_check'] = coefficient_fd
    np.savez(directory / 'coefficient_check.npz', outputs=np.asarray(outputs),
             gradient=np.asarray(gradient), direction=direction)
    persist()
    if not coefficient_fd['passed']:
        raise RuntimeError('Implicit coefficient gradient fails reconverged SCF finite difference.')

    from flax import nnx
    import optax
    from jax.flatten_util import ravel_pytree
    from gradscf.data.molecule import atomic_number
    from gradscf.model.nnao import MACEBasisModel, build_graph
    # Fixed CHNOF vocabulary preserves head meaning across every pilot structure.
    model_config = dict(elements=(1, 6, 7, 8, 9), channels=8, num_interactions=2,
        max_ell=1, correlation=2, zero_init=True, basis_family='szp663_direct')
    model = MACEBasisModel(**model_config, rngs=nnx.Rngs(seed))
    graph = build_graph([atomic_number(symbol) for symbol in experiment.symbols],
                        experiment.coords, element_order=model.elements)
    definition, parameters, other = nnx.split(model, nnx.Param, ...)
    initial, unravel = ravel_pytree(parameters)

    @jax.jit
    def predict(vector):
        return nnx.merge(definition, unravel(vector), other)(graph)

    @jax.jit
    def backward(vector, cotangent):
        return jax.vjp(predict, vector)[1](cotangent)[0]

    def evaluate(vector):
        prediction = predict(jnp.asarray(vector))
        value, output_gradient, details = experiment.evaluate(prediction)
        parameter_gradient = np.asarray(backward(jnp.asarray(vector), output_gradient))
        if not np.isfinite(parameter_gradient).all():
            raise RuntimeError('MACE parameter gradient is not finite.')
        return value, parameter_gradient, details

    def value_only(vector):
        return experiment.evaluate_value(predict(jnp.asarray(vector)))

    initial_prediction = np.asarray(predict(initial))
    np.testing.assert_allclose(initial_prediction, outputs, atol=1e-12, rtol=1e-12)
    model_energy, parameter_gradient, model_info = evaluate(initial)
    np.testing.assert_allclose(model_energy, energy, atol=1e-10, rtol=0)
    norm = float(np.linalg.norm(parameter_gradient))
    parameter_direction = parameter_gradient / norm if norm > 1e-12 else (
        np.ones(parameter_gradient.shape) / np.sqrt(parameter_gradient.size))
    parameter_fd = directional_check(value_only, initial, parameter_gradient, parameter_direction,
        step=fd_step, atol=fd_atol, rtol=fd_rtol)
    report.update(mace_ran=True, model_config=model_config, parameter_count=int(initial.size),
        parameter_gradient_norm=norm, mace_initial_scf=model_info,
        parameter_directional_check=parameter_fd)
    persist()
    if not parameter_fd['passed']:
        raise RuntimeError('MACE parameter gradient fails reconverged SCF finite difference.')

    transform = optax.chain(optax.clip_by_global_norm(1.), optax.scale_by_adam())
    vector = initial; state = transform.init(vector)
    history = [dict(step=0, energy_hartree=model_energy, **model_info)]
    for step in range(1, adam_steps + 1):
        updates, state = transform.update(jnp.asarray(parameter_gradient), state, vector)
        vector = vector - learning_rate * updates
        model_energy, parameter_gradient, model_info = evaluate(vector)
        history.append(dict(step=step, energy_hartree=model_energy, **model_info))
        atomic_json(directory / 'adam_history.json', history)
        # Only valid, accepted parameter vectors are checkpointed.
        temporary = directory / 'checkpoint.tmp.npz'
        np.savez(temporary, parameters=np.asarray(vector),
                 adam_state=np.asarray(ravel_pytree(state)[0]), step=step,
                 initial_parameters=np.asarray(initial), outputs=np.asarray(predict(vector)))
        os.replace(temporary, directory / 'checkpoint.npz')
        report.update(adam_ran=True, accepted_adam_steps=step, adam_history=history)
        persist()
    decrease = energy - model_energy
    report.update(final_energy_hartree=model_energy, energy_decrease_hartree=decrease,
        accepted_adam_steps=adam_steps, adam_learning_rate=learning_rate,
        adam_gradient_clip_norm=1., adam_history=history,
        final_parameter_gradient_norm=float(np.linalg.norm(parameter_gradient)))
    persist()
    if adam_steps and (decrease < -1e-10 or (norm > 1e-9 and decrease <= 1e-10)):
        raise RuntimeError('Pilot Adam did not demonstrate energy reduction.')
    jax.clear_caches()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare-cache', 'validate'))
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--platform', choices=('cpu', 'cuda'))
    parser.add_argument('--auxbasis', default='def2-universal-jkfit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--adam-steps', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--fd-step', type=float, default=1e-4)
    parser.add_argument('--fd-atol', type=float, default=2e-6)
    parser.add_argument('--fd-rtol', type=float, default=2e-5)
    parser.add_argument('--implicit-tolerance', type=float, default=1e-9)
    parser.add_argument('--scf-rescue-level-shift', type=float, default=0.0)
    parser.add_argument('--cache-storage', choices=('device', 'host', 'mmap'), default='device')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.implicit_tolerance) or args.implicit_tolerance <= 0:
        parser.error('Implicit tolerance must be finite and positive.')
    if not math.isfinite(args.scf_rescue_level_shift) or args.scf_rescue_level_shift < 0:
        parser.error('SCF rescue level shift must be finite and nonnegative.')
    if args.adam_steps < 0 or min(args.learning_rate, args.fd_step, args.fd_atol, args.fd_rtol) <= 0:
        parser.error('Adam steps must be nonnegative; learning rate and FD settings must be positive.')
    platform = args.platform or ('cpu' if args.mode == 'prepare-cache' else 'cuda')
    if args.mode == 'prepare-cache' and platform != 'cpu':
        parser.error('Native integral cache preparation requires --platform cpu.')
    if args.mode == 'prepare-cache' and args.cache_storage != 'device':
        parser.error('Native integral cache preparation requires --cache-storage device.')
    rows = read_structures(args.structures, ids=args.ids, limit=args.limit)
    config = dict(mode=args.mode, platform=platform, seed=args.seed,
        source_sha256=file_hash(__file__), experiment_source_sha256=file_hash(
            Path(__file__).with_name('optimize_methane_nnao.py')),
        structures_sha256=file_hash(args.structures), auxbasis=args.auxbasis,
        cache_dir=str(args.cache_dir.resolve()), adam_steps=args.adam_steps,
        learning_rate=args.learning_rate, fd_step=args.fd_step,
        fd_atol=args.fd_atol, fd_rtol=args.fd_rtol,
        implicit_tolerance=args.implicit_tolerance, cache_storage=args.cache_storage,
        scf_rescue_level_shift=args.scf_rescue_level_shift)
    from importlib.metadata import PackageNotFoundError, version
    config['python_version'] = host_platform.python_version()
    config['scientific_source_sha256'] = scientific_source_hash(Path(__file__).resolve().parents[1])
    config['software_versions'] = {}
    for package in ('jax', 'jaxlib', 'flax', 'optax', 'e3nn-jax', 'mace-jax'):
        try:
            config['software_versions'][package] = version(package)
        except PackageNotFoundError:
            config['software_versions'][package] = 'not installed as a distribution'
    jax, experiment_class = numerical_runtime(platform)

    def operation(row, directory):
        common = dict(cache_dir=args.cache_dir, auxbasis=args.auxbasis,
                      jax=jax, experiment_class=experiment_class,
                      implicit_tolerance=args.implicit_tolerance,
                      scf_rescue_level_shift=args.scf_rescue_level_shift)
        if args.mode == 'prepare-cache':
            return prepare_cache(row, directory, **common)
        return validate(row, directory, seed=args.seed, adam_steps=args.adam_steps,
            learning_rate=args.learning_rate, fd_step=args.fd_step,
            fd_atol=args.fd_atol, fd_rtol=args.fd_rtol,
            cache_storage=args.cache_storage, **common)

    summary = run_suite(rows, args.output_dir, config, operation, resume=not args.no_resume)
    print(f"Passed {summary['passed']}; failed {summary['failed']}", flush=True)
    return summary['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
