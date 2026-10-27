"""Small real-MACE NNAO step probe; no trained checkpoint is used.

Run ``--mode full-step`` and ``--mode components`` in separate fresh processes
for each device. Each case measures its first invocation and three warm steps.
The component case splits the shared implicit value's VJP for synchronization;
the full-step case calls unchanged NativeJointEngine.evaluate/molecule instead.
Neither case duplicates a solver or changes the validation kernel.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--mode', choices=('full-step', 'components'), default='full-step')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='preparing', fixture='small-random-MACE-two-H2-geometries',
        trained_checkpoint=False, command=sys.argv, device=args.device, mode=args.mode,
        seed=args.seed, samples=[], units=dict(geometry='Angstrom', energy='Hartree',
        exponents='Bohr^-2', beta='dimensionless log exponent scale', timings='seconds'))
    def save():
        temporary = args.output.with_name(args.output.name+'.tmp')
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
        temporary.replace(args.output)
    save()
    started = time.perf_counter()
    try:
        os.environ.setdefault('JAX_ENABLE_X64', '1')
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root/'src'))
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        import numpy as np
        from flax import nnx
        from jax.flatten_util import ravel_pytree
        from gradscf import scf
        from gradscf.integrals._native import register_integrals
        from gradscf.model.nnao import MACEBasisModel, build_graph, prepare_direct_basis
        from gradscf.model.nnao import runtime as pilot
        from gradscf.model.nnao.config import EXPONENT_KEYS
        from gradscf.model.nnao.integrals.context import prepare_native_context
        from gradscf.model.nnao.integrals.variable_exponents import native_projection
        from gradscf.model.nnao.training.native import NativeJointEngine, JointAdam
        from gradscf.model.nnao.training.objective import BasisRHF, _df_rhf_solver
        from gradscf.model.nnao.training.trainer import trainable_parameter_mask, evaluate_batch
        from gradscf.model.nnao.training.checkpoints import save_checkpoint, load_checkpoint
        from gradscf.tools.jax_runtime import BoundedJitCache
        import gradscf
        import mace_jax

        def ready(value):
            for leaf in jax.tree.leaves(value):
                if hasattr(leaf, 'block_until_ready'):
                    leaf.block_until_ready()
            return value

        def timed(function):
            tick = time.perf_counter()
            value = ready(function())
            return value, time.perf_counter()-tick

        def versions():
            result = {}
            for name in ('jax', 'jaxlib', 'flax', 'optax', 'mace-jax', 'cuequivariance'):
                try: result[name] = metadata.version(name)
                except metadata.PackageNotFoundError: result[name] = None
            return result

        def python_tree_hash(directory):
            digest = hashlib.sha256()
            for path in sorted(directory.rglob('*.py')):
                digest.update(str(path.relative_to(directory)).encode())
                digest.update(path.read_bytes())
            return digest.hexdigest()

        cpu = jax.devices('cpu')[0]
        device = jax.devices('gpu')[0] if args.device == 'gpu' else cpu
        library = Path(register_integrals()._name)
        revision_file = root/'SOURCE_REVISION'
        revision = (revision_file.read_text().strip() if revision_file.exists() else
                    subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip())
        cpu_info = Path('/proc/cpuinfo')
        cpu_model = (next((line.split(':', 1)[1].strip() for line in cpu_info.read_text().splitlines()
                          if line.startswith('model name')), platform.processor())
                     if cpu_info.exists() else platform.processor())
        report['runtime'] = dict(hostname=platform.node(), platform=platform.platform(), cpu_model=cpu_model,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            python=sys.version, versions=versions(), source_revision=revision,
            scientific_source_sha256=pilot.scientific_source_hash(root),
            probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            gradscf_source=str(Path(gradscf.__file__).resolve()),
            mace_source=str(Path(mace_jax.__file__).resolve()), native_library=str(library),
            mace_import_tree_sha256=python_tree_hash(Path(mace_jax.__file__).resolve().parent),
            native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            cpu_device=str(cpu), model_device=str(device), model_device_kind=device.device_kind,
            jax_enable_x64=bool(jax.config.jax_enable_x64),
            compilation_cache_enabled=bool(jax.config.jax_enable_compilation_cache),
            compilation_cache_dir=str(jax.config.jax_compilation_cache_dir),
            environment={name: os.environ.get(name) for name in ('JAX_PLATFORMS', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'XLA_FLAGS', 'XLA_PYTHON_CLIENT_PREALLOCATE')},
            cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)
        report['environment_seconds'] = time.perf_counter()-started

        config = dict(elements=(1,), channels=4, num_interactions=1, max_ell=1,
                      correlation=2, basis_family='szp663_direct', zero_init=False)
        tick = time.perf_counter()
        with jax.default_device(device):
            model = MACEBasisModel(**config, rngs=nnx.Rngs(args.seed))
            definition, parameters, other = nnx.split(model, nnx.Param, ...)
            theta, unravel = ravel_pytree(parameters)
            mask = trainable_parameter_mask(parameters)
            beta = jnp.zeros(len(EXPONENT_KEYS)).at[:2].set(jnp.array([.03, -.02]))
        ready((theta, other, beta))
        preparation = dict(model_seconds=time.perf_counter()-tick, geometries=[])
        predict = jax.jit(lambda vector, graph: nnx.merge(definition, unravel(vector), other)(graph))
        backward = jax.jit(lambda vector, graph, cotangent:
            jax.vjp(lambda x: predict(x, graph), vector)[1](cotangent)[0])
        value, gradient = _df_rhf_solver(2, 1e-8, .2)
        with tempfile.TemporaryDirectory(prefix='gradscf-nnao-step-') as scratch:
            # The production constructor enforces the original 7171-state
            # CHNOF checkpoint. This explicit small-fixture adapter retains its
            # exact numerical methods while providing a different MACE shape.
            engine = object.__new__(NativeJointEngine)
            engine.jax, engine.jnp, engine.cpu, engine.device = jax, jnp, cpu, device
            engine.predict, engine.backward, engine.experiment = predict, backward, BasisRHF
            engine.cpu_jit_cache = BoundedJitCache(max_entries=8)
            engine.args = SimpleNamespace(log_bound=.25, output_dir=Path(scratch))
            engine.parent_energies, engine.batch, engine.contexts, engine.rows = None, 0, {}, []
            engine.trainable_mask = mask
            for index, bond in enumerate((.74, .91)):
                tick = time.perf_counter()
                atom = [('H', (0., 0., 0.)), ('H', (0., 0., bond))]
                with jax.default_device(cpu):
                    layout = prepare_direct_basis(atom, cart=False, basis_family='szp663_direct')
                    context = prepare_native_context(layout, atom, 'def2-universal-jkfit')
                    context['physical'], context['pullback'] = native_projection(context, EXPONENT_KEYS)
                    context['value'], context['gradient'] = value, gradient
                    enuc = scf.nuclear_repulsion_energy(layout.parameters.nuclear_coords,
                                                       jnp.asarray(layout.topology.nuclear_charges))
                context['enuc'] = jax.device_put(enuc, device)
                with jax.default_device(device):
                    context['graph'] = build_graph([1, 1], [xyz for _, xyz in atom], element_order=(1,))
                ready((context['enuc'], context['graph']))
                context['energy_aux'] = lambda inputs, enuc=context['enuc']: value(inputs, enuc)
                identifier = f'H2/{index}/{bond:.2f}A'
                engine.contexts[identifier] = context
                engine.rows.append(dict(structure_id=identifier))
                top = layout.topology
                preparation['geometries'].append(dict(structure_id=identifier, bond_angstrom=bond,
                    seconds=time.perf_counter()-tick, nprim_radial=sum(top.primitive_counts),
                    nprim_ao=sum(n*(2*l+1) for n, l in zip(top.primitive_counts, top.angular_momenta)),
                    ncontracted_ao=top.nao, naux_ao=context['aux'].auxiliary.nao,
                    df_rank=context['metric'].matrix.shape[0],
                    integral_plan_interface=context['plan'].backend,
                    orbital_parameter_device=str(layout.parameters.centers.device),
                    auxiliary_parameter_device=str(context['ap'].centers.device)))
            report['preparation'] = preparation
            report['model_config'] = config
            report['solver_configuration'] = dict(method='DF-RHF', nelectron=2,
                implicit_tolerance=1e-8, scf_rescue_level_shift=.2,
                auxbasis='def2-universal-jkfit', coefficient_learning_rate=1e-5,
                exponent_learning_rate=1e-4)
            report['state'] = dict(ntheta=int(theta.size), nbeta=len(EXPONENT_KEYS),
                exponent_keys=[list(key) for key in EXPONENT_KEYS],
                trainable_theta=int(mask.sum()), frozen_theta_indices=np.flatnonzero(~mask).tolist(),
                log_bound=.25, checkpoint_kind='random fixture only')
            report['timing_contract'] = dict(cold='first measured invocation after required preparation',
                warm_steps=3, component_value='shared value primal plus VJP residual setup',
                component_response='pullback only, separately synchronized',
                full_step='NativeJointEngine.evaluate plus JointAdam.update; pre-update loss',
                progress_io='source engine progress writes retained in local node temporary directory',
                validation='final validation and checkpoint roundtrip are outside measured steps')
            report['existing_validation_gates'] = [
                'tests/nnao/test_libcint_consumers.py::test_small_mace_two_geometry_energy_update_and_checkpoint',
                'tests/nnao/test_joint_native_training.py::test_failed_joint_candidate_keeps_parameters_and_adam_moments',
                'tests/nnao/test_joint_native_training.py::test_joint_checkpoint_initialization_preserves_learned_exponents']

            def component_molecule(vector, identifier):
                context = engine.contexts[identifier]
                th, be = vector[:-len(EXPONENT_KEYS)], vector[-len(EXPONENT_KEYS):]
                phases = {}
                output, phases['model_forward'] = timed(lambda: predict(th, context['graph']))
                (out_cpu, be_cpu), phases['model_output_beta_to_cpu'] = timed(
                    lambda: jax.device_put((output, be), cpu))
                physical, phases['libcint_projection'] = timed(lambda: context['physical'](out_cpu, be_cpu))
                if any(array.device.platform != 'cpu' for array in physical):
                    raise RuntimeError('libcint physical arrays must remain on CPU')
                inputs, phases['physical_inputs_to_model_device'] = timed(lambda: jax.device_put(physical, device))
                (energy, pull, info), phases['physical_scf_value_vjp_setup'] = timed(
                    lambda: jax.vjp(context['energy_aux'], inputs, has_aux=True))
                details = BasisRHF._checked_info(energy, info)
                (cotangent,), phases['physical_implicit_pullback'] = timed(lambda: pull(jnp.ones_like(energy)))
                cot_cpu, phases['physical_cotangent_to_cpu'] = timed(lambda: jax.device_put(cotangent, cpu))
                (co, cb), phases['libcint_projection_pullback'] = timed(
                    lambda: context['pullback'](out_cpu, be_cpu, cot_cpu))
                co_device, phases['coefficient_cotangent_to_model_device'] = timed(lambda: jax.device_put(co, device))
                gt, phases['model_vjp'] = timed(lambda: backward(th, context['graph'], co_device))
                derivative, phases['gradient_to_host'] = timed(
                    lambda: np.concatenate((np.asarray(gt), np.asarray(cb))))
                details['phase_seconds'] = phases
                return float(energy), derivative, details

            def evaluate(vector):
                if args.mode == 'full-step':
                    return engine.evaluate(vector)
                return evaluate_batch(vector, [(row['structure_id'],
                    lambda x, key=row['structure_id']: component_molecule(x, key)) for row in engine.rows])

            optimizer = JointAdam(theta.size, learning_rate=1e-5, exponent_learning_rate=1e-4,
                                  log_bound=.25, trainable_mask=mask)
            with jax.default_device(device):
                vector = jnp.concatenate((theta, beta))
                state = optimizer.init(vector)
                ready((vector, state))
                report['status'] = 'running'
                save()
                for index in range(4):
                    tick = time.perf_counter()
                    (loss, derivative, records), evaluation_seconds = timed(lambda: evaluate(vector))
                    (candidate, candidate_state), optimizer_seconds = timed(
                        lambda: optimizer.update(vector, derivative, state))
                    elapsed = time.perf_counter()-tick
                    frozen = np.asarray(candidate[:theta.size])[~mask]
                    if not np.array_equal(frozen, np.asarray(vector[:theta.size])[~mask]):
                        raise RuntimeError('JointAdam changed frozen MACE state')
                    if not np.isfinite(np.asarray(candidate)).all() or np.max(np.abs(np.asarray(candidate[-14:]))) > .25:
                        raise RuntimeError('JointAdam produced invalid bounded parameters')
                    report['samples'].append(dict(kind='cold' if index == 0 else 'warm', index=index,
                        full_loss_grad_optimizer_seconds=elapsed, loss_grad_seconds=evaluation_seconds,
                        optimizer_seconds=optimizer_seconds, loss_hartree=loss,
                        gradient_norm=float(np.linalg.norm(derivative)), beta=np.asarray(candidate[-14:]).tolist(),
                        updated_parameters_evaluated=False, molecules=records,
                        cpu_jit_cache_entries=len(engine.cpu_jit_cache),
                        cpu_jit_cache_evictions=engine.cpu_jit_cache.evictions,
                        jit_cache_global_clears=engine.cpu_jit_cache.global_clears))
                    vector, state = candidate, candidate_state
                    save()
                (final_loss, final_gradient, final_records), validation_seconds = timed(lambda: evaluate(vector))
                identity = dict(fixture=report['fixture'], seed=args.seed, model_config=config,
                                exponent_keys=[list(key) for key in EXPONENT_KEYS])
                # JSON canonical form matches the real checkpoint identity rule.
                identity = json.loads(json.dumps(identity))
                checkpoint = args.output.with_suffix('.fixture.npz')
                save_checkpoint(checkpoint, dict(identity=identity, parameters=np.asarray(vector),
                    gradient=final_gradient, optimizer=state, epoch=4, loss_hartree=final_loss), optimizer)
                restored = load_checkpoint(checkpoint, identity, optimizer, vector)
                np.testing.assert_array_equal(restored['parameters'], vector)
                for left, right in zip(jax.tree.leaves(restored['optimizer']), jax.tree.leaves(state)):
                    np.testing.assert_array_equal(left, right)
                report['final_validation'] = dict(seconds=validation_seconds, loss_hartree=final_loss,
                    gradient_finite=bool(np.isfinite(final_gradient).all()), molecules=final_records,
                    checkpoint=str(checkpoint), checkpoint_roundtrip=True)
            if any(name == 'pyscf' or name.startswith('pyscf.') or name.startswith('gpu4pyscf') for name in sys.modules):
                raise RuntimeError('NNAO numerical probe imported an external chemistry runtime')
            report['device_allocator_stats'] = device.memory_stats()
            report['device_memory_method'] = 'JAX allocator counters; reserved/pool and in-use bytes are distinct'
            report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        save()
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        report['peak_rss_bytes'] = 1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['memory_method'] = 'Linux ru_maxrss including compilation and all repeats'
        save()
    print(json.dumps(dict(status=report['status'], output=str(args.output), mode=args.mode, device=args.device)))


if __name__ == '__main__':
    main()
