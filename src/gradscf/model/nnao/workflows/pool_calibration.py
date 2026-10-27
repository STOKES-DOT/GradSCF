"""Small-sample active primitive-pool calibration and free-contraction audit."""
import argparse
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import platform
import threading
import time

import numpy as np

from ..runtime import atomic_json, file_hash, repository_root, scientific_source_hash


def _validate_definition(data):
    from ..initialization import FAMILY_GRID
    if data.get('method') != 'RHF' or data.get('units') != 'Angstrom':
        raise ValueError('This pilot requires explicit RHF/Angstrom reference metadata.')
    if data.get('basis_family') not in {'szp663_direct'} | FAMILY_GRID.keys() or data.get('core_primitives') not in (6, 'reference', '6-31g'):
        raise ValueError('Use szp663_direct or an explicit NNAO grid family with core6 or reference cores.')
    if 'structures' in data:
        from ..config import model_config_for_definition
        model_config_for_definition(data)


def load_references(path=None, *, ids=None):
    text = (Path(path).read_text() if path else
            files('gradscf.model.nnao').joinpath('data/pool_pilot.json').read_text())
    data = json.loads(text)
    _validate_definition(data)
    rows = data['structures']
    names = [row['structure_id'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('Reference IDs must be unique.')
    if ids is not None:
        if set(ids)-set(names):
            raise ValueError('Unknown requested reference IDs.')
        rows = [row for row in rows if row['structure_id'] in ids]
    if not rows:
        raise ValueError('A nonempty reference set is required.')
    for row in rows:
        coordinates = np.asarray(row['coords_angstrom'])
        if coordinates.shape != (len(row['symbols']), 3) or not np.isfinite(coordinates).all():
            raise ValueError(f"Invalid coordinates: {row['structure_id']}")
        if row.get('charge') != 0 or row.get('spin') != 0:
            raise NotImplementedError('The initial pilot covers neutral spin=0 molecules only.')
        if not np.isfinite(row.get('weight', 0)) or row.get('weight', 0) <= 0:
            raise ValueError('Reference weights must be explicitly positive and finite.')
    return dict(data, structures=rows)


def _measure(name, operation, measurements):
    """Sample whole-process RSS; report missing measurements rather than zeros."""
    try:
        import psutil
        process = psutil.Process()
        rss = lambda: process.memory_info().rss
        before = rss()
    except (ImportError, OSError):
        rss = None
        before = None
    samples = [] if before is None else [before]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.02):
            samples.append(rss())
    thread = threading.Thread(target=monitor, daemon=True) if rss else None
    if thread:
        thread.start()
    start = time.perf_counter()
    try:
        return operation()
    finally:
        elapsed = time.perf_counter()-start
        stop.set()
        if thread:
            thread.join()
            samples.append(rss())
        measurements.append(dict(phase=name, seconds=elapsed, rss_before_bytes=before,
            rss_after_bytes=samples[-1] if samples else None,
            sampled_process_peak_rss_bytes=max(samples) if samples else None,
            memory_method='psutil RSS sampled every 20 ms' if rss else 'unavailable',
            gpu_memory_bytes=None))


def _save_result(path, result, *, kind, identity):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in result.items()
                if key not in ('parameters', 'gradient', 'optimizer')}
    metadata.update(kind=kind, identity=identity)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, parameters=result['parameters'], gradient=result['gradient'],
            optimizer=np.frombuffer(result['optimizer'], dtype=np.uint8),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    os.replace(temporary, path)


def _report(output, result):
    lines = ['# NNAO primitive-pool / free-contraction pilot', '',
        'Neutral spin=0, all-electron RHF; fixed core basis parameters, no frozen core electrons.',
        f"Spherical {result['basis_family']}; core policy {result['core_primitives']}; fixed def2-universal-jkfit DF approximation.", '',
        '| Candidate | Molecule | Pool / compact AO | E pool / Eh | E free / Eh | Contraction gap / mEh | Free optimizer |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for row in result['comparisons']:
        lines.append(f"| {row['candidate']} | {row['structure_id']} | {row['pool_nao']} / {row['contracted_nao']} | "
            f"{row['pool_energy_hartree']:.10f} | {row['free_energy_hartree']:.10f} | "
            f"{row['delta_contraction_hartree']*1000:.5f} | {row['optimizer_status']} |")
    lines += ['', '## Interpretation', '',
        'These are local, finite-iteration reference optimizations. `max_steps` or `rejected` is not an optimized limit.',
        'No MACE model is trained in stages A/B; network energies and network gaps are therefore not reported.',
        'No initial pool is installed automatically. Compare sufficiently converged multi-start compact results before selecting one.',
        'The reference geometries and equal weights are explicit pilot choices, not a GMTKN55 benchmark or a property validation.',
        'RSS values are sampled peaks of the whole process during each phase, not exact allocation sizes or GPU memory.', '']
    (output/'REPORT.md').write_text('\n'.join(lines))


def run_pilot(data, output_dir, *, calibration_steps=3, contraction_steps=30,
              starts=2, seed=0, calibration_rate=.01, contraction_rate=.01,
              log_bound=.15, check_gradients=True, fixed_candidate=None, scf_platform='cpu'):
    import jax
    import jax.numpy as jnp
    from gradscf.integrals._native import register_integrals
    from ..basis import prepare_direct_basis
    from ..initialization import pool_channel_keys, bind_pool_candidate
    from ..training.pool import PoolRHF, minimize_reference
    _validate_definition(data)
    if not jax.config.jax_enable_x64 or scf_platform not in ('cpu','cuda'):
        raise ValueError('The reference pilot requires float64 and scf_platform cpu or cuda.')
    cpu = jax.devices('cpu')[0]
    scf_device = jax.devices(scf_platform)[0]
    if min(calibration_steps, contraction_steps) < 0 or starts < 1:
        raise ValueError('Use nonnegative step counts and at least one start.')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Pilot output directory must be empty.')
    output.mkdir(parents=True, exist_ok=True)
    rows = data['structures']
    with jax.default_device(cpu):
        layouts = [prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
                   cart=False, basis_family=data['basis_family'], core_primitives=data['core_primitives']) for row in rows]
        keys = pool_channel_keys(layouts)
        zero = jnp.zeros((len(keys), 2), dtype=jnp.float64)
        if fixed_candidate is not None:
            for layout in layouts:
                bind_pool_candidate(layout, fixed_candidate)
            if tuple(map(tuple, fixed_candidate['parameters']['keys'])) != keys:
                raise ValueError('Fixed candidate channel ordering differs from pilot.')
            zero = jnp.asarray(fixed_candidate['parameters']['values'])
    weights = np.asarray([row['weight'] for row in rows], dtype=float)
    measurements = []
    identity = dict(references=data, channel_keys=[list(key) for key in keys],
        method='all-electron restricted HF', frozen_core_electrons=False,
        basis_family=data['basis_family'], core_primitives=data['core_primitives'], auxiliary_basis='def2-universal-jkfit',
        fixed_candidate=fixed_candidate,
        parameters='log scale and log spacing per element/l/role',
        scf=dict(max_cycle=150, conv_tol_energy=1e-12, conv_tol_density=1e-10,
                 conv_tol_grad=1e-9, implicit_tolerance=1e-8, rescue_level_shift=.2,
                 final_rescue=dict(level_shift=.5,damping=.2,raw_polish=True)),
        configuration=dict(calibration_steps=calibration_steps, contraction_steps=contraction_steps,
            starts=starts, seed=seed, calibration_rate=calibration_rate,
            contraction_rate=contraction_rate, log_bound=log_bound,check_gradients=bool(check_gradients)),
        software_source_sha256=scientific_source_hash(repository_root()),
        native_sha256=file_hash(register_integrals()._name), jax=jax.__version__,
        device=[str(scf_device)], native_device=str(cpu), scf_platform=scf_platform,
        hardware=platform.platform(), dtype='float64')
    atomic_json(output/'metadata.json', identity)
    inventory = []
    for row, layout in zip(rows, layouts):
        inventory.append(dict(structure_id=row['structure_id'], nelectron=layout.nelectron,
            contracted_nao=layout.topology.nao, shells=[dict(atom=atom, element=layout.symbols[atom],
                l=l, role=role, fixed=slot < 0, exponents_bohr_minus2=np.asarray(a).tolist(),
                coefficients=np.asarray(c).tolist()) for atom, l, role, slot, a, c in zip(
                    layout.shell_atoms, layout.topology.angular_momenta, layout.roles, layout.slots,
                    layout.parameters.exponents, layout.parameters.coefficients)]))
    atomic_json(output/'basis-inventory.json', inventory)
    def measure(name, operation, records):
        atomic_json(output/'phase.json', dict(phase=name,status='running'))
        try:
            result = _measure(name, operation, records)
        except Exception as error:
            atomic_json(output/'phase.json', dict(phase=name,status='failed',error=str(error)))
            raise
        else:
            atomic_json(output/'phase.json', dict(phase=name,status='complete'))
            return result
        finally:
            atomic_json(output/'timings.json', dict(measurements=records))

    models = measure('prepare_native_plans_and_auxiliary_metrics',
                      lambda: [PoolRHF(row, keys, basis_family=data['basis_family'],
                                       core_primitives=data['core_primitives'], scf_device=scf_device)
                               for row in rows], measurements)
    def ensemble(parameters, *, gradient=True):
        values = []
        for model in models:
            atomic_json(output/'ensemble-progress.json', dict(completed=len(values),total=len(models),
                active_structure=model.row['structure_id'],gradient=gradient))
            values.append(model.evaluate(parameters, gradient=gradient))
            atomic_json(output/'ensemble-progress.json', dict(completed=len(values),total=len(models),
                last_structure=model.row['structure_id'],gradient=gradient,
                scf_backend=scf_device.platform))
        energy = float(np.dot(weights, [x[0] for x in values])/weights.sum())
        derivative = (sum(w*x[1] for w, x in zip(weights, values))/weights.sum()
                      if gradient else None)
        return energy, derivative, {'molecules': [x[2] for x in values]}
    baseline = measure('pool_first_call_including_compilation', lambda: ensemble(zero), measurements)
    hot = measure('pool_hot_value_and_gradient', lambda: ensemble(zero), measurements)
    np.testing.assert_allclose(hot[0], baseline[0], atol=1e-10, rtol=0)
    np.testing.assert_allclose(hot[1], baseline[1], atol=1e-9, rtol=1e-8)
    if check_gradients:
        direction = np.random.default_rng(seed).normal(size=zero.shape)
        direction /= np.linalg.norm(direction)
        step = 1e-4
        plus = ensemble(zero+step*direction, gradient=False)[0]
        minus = ensemble(zero-step*direction, gradient=False)[0]
        fd, ad = (plus-minus)/(2*step), float(np.sum(baseline[1]*direction))
        passed = bool(np.isclose(ad, fd, atol=2e-6, rtol=2e-5))
        atomic_json(output/'exponent-gradient-check.json', dict(passed=passed, ad=ad, fd=fd,
            step=step, atol=2e-6, rtol=2e-5, perturbations_recompute_integrals=True))
        if not passed:
            raise RuntimeError('Ensemble exponent directional finite difference failed.')
    if fixed_candidate is None:
        calibrated = measure('pool_calibration', lambda: minimize_reference(zero, ensemble,
            max_steps=calibration_steps, learning_rate=calibration_rate,
            bounds=(-log_bound, log_bound)), measurements)
        _save_result(output/'checkpoints/pool-calibration.npz', calibrated,
                     kind='active_pool_calibration', identity=identity)
        candidates = [('baseline', np.asarray(zero))]
        if calibrated['accepted_steps']:
            candidates.append(('calibrated_candidate', calibrated['parameters']))
    else:
        calibrated = dict(status='fixed_candidate_reused', accepted_steps=0)
        candidates = [('fixed_candidate', np.asarray(zero))]
    result = dict(status='running', comparisons=[], measurements=measurements,
                  basis_family=data['basis_family'],
                  core_primitives=data['core_primitives'],
                  selected_initial_pool=None, calibration_status=calibrated['status'],
                  calibration_accepted_steps=calibrated['accepted_steps'])
    for candidate, parameters in candidates:
        atomic_json(output/f'{candidate}-parameters.json', dict(keys=[list(k) for k in keys],
                    values=np.asarray(parameters).tolist()))
        for row, model in zip(rows, models):
            identifier = row['structure_id']
            pool_energy, _, pool_info = model.evaluate(parameters, gradient=False)
            free = measure(f'{candidate}/{identifier}/prepare_fixed_pool_cache',
                            lambda: model.prepare_contraction(parameters), measurements)
            with jax.default_device(cpu):
                reference = free.layout.reference_outputs()
            initial = measure(f'{candidate}/{identifier}/contraction_first_call',
                               lambda: free.evaluate(reference), measurements)
            measure(f'{candidate}/{identifier}/contraction_hot_call',
                     lambda: free.evaluate(reference), measurements)
            runs = []
            for start in range(starts):
                point = np.asarray(reference)
                if start:
                    point = point+.05*np.random.default_rng(seed+start).normal(size=point.shape)
                optimized = measure(f'{candidate}/{identifier}/free_start_{start}',
                    lambda: minimize_reference(point, free.evaluate, max_steps=contraction_steps,
                                               learning_rate=contraction_rate), measurements)
                digest = hashlib.sha256(identifier.encode()).hexdigest()[:16]
                checkpoint = Path('checkpoints')/candidate/f'{digest}-start-{start}.npz'
                _save_result(output/checkpoint, optimized, kind='free_contraction',
                    identity=dict(identity, candidate=candidate, structure_id=identifier,
                                  pool_parameters=np.asarray(parameters).tolist(), start=start))
                runs.append((optimized, checkpoint))
            best, checkpoint = min(runs, key=lambda item: item[0]['energy_hartree'])
            gap = best['energy_hartree']-pool_energy
            if gap < -1e-7:
                raise RuntimeError(f'{candidate}/{identifier}: energy ordering reversed ({gap} Ha). Check SCF branches.')
            record = dict(candidate=candidate, structure_id=identifier,
                pool_nao=model.pool_nao, contracted_nao=model.layout.topology.nao,
                pool_energy_hartree=pool_energy, template_energy_hartree=initial[0],
                free_energy_hartree=best['energy_hartree'], delta_contraction_hartree=gap,
                network_energy_hartree=None, delta_network_hartree=None,
                optimizer_status=best['status'], optimizer_converged=best['optimizer_converged'],
                gradient_norm=best['gradient_norm'], checkpoint=str(checkpoint),
                pool_condition_number=pool_info['pool_condition_number'],
                cached_pool_bytes=sum(x.size*x.dtype.itemsize for x in free.raw),
                pool_scf=pool_info, free_scf=best['info'])
            result['comparisons'].append(record)
            atomic_json(output/'summary.json', result)
            _report(output, result)
            print(f"{candidate} {identifier}: pool={pool_energy:.10f}, free={best['energy_hartree']:.10f}, "
                  f"gap={gap*1000:.4f} mEh, {best['status']}", flush=True)
    result['status'] = 'complete'
    atomic_json(output/'summary.json', result)
    _report(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--calibration-steps', type=int, default=3)
    parser.add_argument('--contraction-steps', type=int, default=30)
    parser.add_argument('--starts', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fixed-candidate', type=Path)
    parser.add_argument('--scf-platform', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args(argv)
    os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu' if args.scf_platform=='cuda' else 'cpu')
    os.environ.setdefault('JAX_ENABLE_X64', '1')
    os.environ.setdefault('GRADSCF_NATIVE_THREADS', '1')
    import jax
    jax.config.update('jax_enable_x64', True)
    data = load_references(args.references, ids=args.ids)
    run_pilot(data, args.output_dir, calibration_steps=args.calibration_steps,
              contraction_steps=args.contraction_steps, starts=args.starts, seed=args.seed,
              fixed_candidate=json.loads(args.fixed_candidate.read_text()) if args.fixed_candidate else None,
              scf_platform=args.scf_platform)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
