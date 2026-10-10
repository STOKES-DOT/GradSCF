"""Shared MACE coefficients and element/angular exponent scales, native AD + implicit HF.

Native contracted integrals and their pullbacks run on CPU; MACE and the
physical-input implicit SCF run on the selected device. No primitive cache.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('joint_baseline', Path(__file__).with_name('train_nnao_joint.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
MODEL_CONFIG = json.loads(json.dumps(base.MODEL_CONFIG))


def native_projection(context):
    """Return contracted physical inputs and their native CPU pullback."""
    import jax
    import jax.numpy as jnp
    from gradscf.integrals.molecular.density_fitting import unpack_factors

    layout = context['layout']
    indices = {key: base.EXPONENT_KEYS.index(key) for key in layout.exponent_scale_keys}

    def physical(outputs, beta):
        scaled = layout.with_log_exponent_scales({key: beta[index] for key, index in indices.items()})
        bound = scaled.bind(outputs)
        # This training graph varies basis parameters at fixed geometry. bind()
        # gathers centers under JIT; retain the concrete centers for the metric signature.
        bound = replace(bound, centers=layout.parameters.centers,
                        nuclear_coords=layout.parameters.nuclear_coords)
        s = context['plan'].evaluate('overlap', bound)
        h = context['plan'].evaluate('kinetic', bound) + context['plan'].evaluate('nuclear', bound)
        packed = context['aux'].factors(bound, context['ap'], metric_factor=context['metric'])
        return s, h, unpack_factors(packed, layout.topology.nao)

    def pullback(outputs, beta, cotangent):
        return jax.vjp(physical, outputs, beta)[1](cotangent)

    return jax.jit(physical), jax.jit(pullback)


class JointAdam(base.trainer.Adam):
    """Separate learning rates, shared clipping, projected log-exponent bounds."""
    def __init__(self, ntheta, *, learning_rate, exponent_learning_rate, log_bound, trainable_mask=None):
        super().__init__(1.)
        if min(learning_rate, exponent_learning_rate, log_bound) <= 0:
            raise ValueError('Learning rates and exponent bound must be positive.')
        self.ntheta = ntheta
        self.coefficient_rate = learning_rate
        self.exponent_rate = exponent_learning_rate
        self.log_bound = log_bound
        self.trainable_mask = (np.ones(ntheta, dtype=bool) if trainable_mask is None
                               else np.asarray(trainable_mask, dtype=bool))
        if self.trainable_mask.shape != (ntheta,):
            raise ValueError('Trainable mask must match the flat MACE vector.')

    def _step(self, vector, gradient, state):
        import jax.numpy as jnp
        import optax
        gradient = gradient.at[:self.ntheta].set(gradient[:self.ntheta]*self.trainable_mask)
        updates, state = self.transform.update(gradient, state, vector)
        rates = jnp.where(jnp.arange(vector.size) < self.ntheta,
                          self.coefficient_rate, self.exponent_rate)
        candidate = optax.apply_updates(vector, updates*rates)
        candidate = candidate.at[:self.ntheta].set(jnp.where(
            self.trainable_mask, candidate[:self.ntheta], vector[:self.ntheta]))
        candidate = candidate.at[self.ntheta:].set(
            jnp.clip(candidate[self.ntheta:], -self.log_bound, self.log_bound))
        return candidate, state

    def learning_rate_at(self, count):
        return self.coefficient_rate


class NativeJointEngine(base.JointEngine):
    """Reuse model/graph/metric setup; bypass the legacy primitive-cache path."""
    def __init__(self, rows, args, parent_identity, parent_energies):
        super().__init__(rows, args, parent_identity)
        self.parent_energies = parent_energies
        self.batch = 0
        for context in self.contexts.values():
            context['physical'], context['pullback'] = native_projection(context)
            context['value'], context['gradient'] = self.implicit_solver(
                context['layout'].nelectron, args.implicit_tolerance, args.scf_rescue_level_shift)

    def molecule(self, vector, identifier, *, gradient=True):
        jax = self.jax
        theta, beta = vector[:-len(base.EXPONENT_KEYS)], vector[-len(base.EXPONENT_KEYS):]
        context = self.contexts[identifier]
        tick = time.perf_counter()
        theta = jax.device_put(theta, self.device)
        outputs = self.predict(theta, context['graph'])
        with jax.default_device(self.cpu):
            outputs_cpu, beta_cpu = jax.device_put((outputs, beta), self.cpu)
            physical = context['physical'](outputs_cpu, beta_cpu)
            jax.block_until_ready(physical)
        integral_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        inputs = jax.device_put(physical, self.device)
        if gradient:
            (value, info), physical_cotangent = context['gradient'](inputs, context['enuc'])
        else:
            value, info = context['value'](inputs, context['enuc'])
        details = self.experiment._checked_info(value, info)
        scf_seconds = time.perf_counter()-tick
        if not gradient:
            return float(value), details
        tick = time.perf_counter()
        with jax.default_device(self.cpu):
            output_cotangent, beta_cotangent = context['pullback'](
                outputs_cpu, beta_cpu, jax.device_put(physical_cotangent, self.cpu))
            jax.block_until_ready((output_cotangent, beta_cotangent))
        native_backward_seconds = time.perf_counter()-tick
        theta_cotangent = self.backward(theta, context['graph'], jax.device_put(output_cotangent, self.device))
        result = np.concatenate((np.asarray(theta_cotangent), np.asarray(beta_cotangent)))
        details.update(integral_seconds=integral_seconds, scf_seconds=scf_seconds,
            native_backward_seconds=native_backward_seconds,
            exponent_gradient_norm=float(np.linalg.norm(result[-len(base.EXPONENT_KEYS):])))
        return float(value), result, details

    def evaluate(self, vector):
        beta = np.asarray(vector[-len(base.EXPONENT_KEYS):])
        if not np.isfinite(beta).all() or np.any(np.abs(beta) > self.args.log_bound+1e-12):
            raise ValueError('Invalid bounded exponent scales.')
        self.batch += 1
        evaluators = [(r['structure_id'], lambda x, key=r['structure_id']: self.molecule(x, key)) for r in self.rows]
        def progress(records):
            base.pilot.atomic_json(self.args.output_dir/'batch-progress.json', dict(batch=self.batch,
                evaluated=len(records), total=len(self.rows), last=records[-1], beta=beta.tolist()))
        loss, g, records = base.trainer.evaluate_batch(vector, evaluators, progress=progress)
        if self.parent_energies is not None:
            if np.any(beta != 0.):
                raise ValueError('Parent parity gate requires zero initial exponent scales.')
            actual = {r['structure_id']: r['energy_hartree'] for r in records}
            base.check_energy_parity(self.parent_energies, actual, tolerance=1e-7)
            base.pilot.atomic_json(self.args.output_dir/'parent-parity.json', dict(
                max_absolute_error_hartree=max(abs(actual[k]-self.parent_energies[k]) for k in actual),
                energies=actual, tolerance_hartree=1e-7, all_gradients_finite=True))
            self.parent_energies = None
        return loss, g, records

    def check_gradient(self, vector, identifier):
        direction = np.sin(np.arange(vector.size)+1.)
        direction[:-14] *= self.trainable_mask
        direction[:-14] /= np.linalg.norm(direction[:-14])
        direction[-14:] *= .1
        value, g, details = self.molecule(vector, identifier)
        analytic = float(g@direction)
        h = 1e-4
        energies = {k: self.molecule(vector+k*h*direction, identifier, gradient=False)[0]
                    for k in (-2, -1, 1, 2)}
        fd = (8*(energies[1]-energies[-1])-energies[2]+energies[-2])/(12*h)
        if not np.isfinite(analytic) or abs(analytic-fd) > 2e-6+2e-5*abs(fd):
            raise RuntimeError(f'{identifier}: joint GPU/native gradient mismatch: {analytic} vs {fd}')
        return dict(structure_id=identifier, energy_hartree=value, analytic=analytic,
            finite_difference=fd, absolute_error=abs(analytic-fd), step=h, diagnostics=details)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--exponent-learning-rate', type=float, default=1e-3)
    parser.add_argument('--log-bound', type=float, default=.25)
    parser.add_argument('--implicit-tolerance', type=float, default=1e-8)
    parser.add_argument('--scf-rescue-level-shift', type=float, default=.2)
    parser.add_argument('--auxbasis', default='def2-universal-jkfit')
    parser.add_argument('--platform', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    values = [args.learning_rate, args.exponent_learning_rate, args.log_bound, args.implicit_tolerance]
    if args.epochs < 0 or not np.isfinite(values).all() or min(values) <= 0:
        parser.error('Require nonnegative epochs and finite positive numerical settings.')
    if (args.output_dir/'last.npz').exists() and not args.resume:
        parser.error('Existing accepted checkpoint: use --resume or a fresh output directory.')
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu' if args.platform == 'cuda' else 'cpu'
    os.environ['JAX_ENABLE_X64'] = '1'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import jax
    jax.config.update('jax_enable_x64', True)
    rows = base.pilot.read_structures(args.structures)
    theta, parent_identity = base.read_parent(args.checkpoint, rows)
    if len(rows) != 24 or set(r['structure_id'] for r in rows) != set(parent_identity['structure_ids']):
        raise ValueError('This continuation requires exactly the original 24 structures.')
    if base.pilot.file_hash(args.structures) != parent_identity['structures_sha256']:
        raise ValueError('The original 24-structure file bytes differ.')
    with np.load(args.checkpoint, allow_pickle=False) as saved:
        parent = json.loads(str(saved['metadata'].item()))
    parent_energies = {r['structure_id']: r['energy_hartree'] for r in parent['molecules']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from gradscf.integrals import _native
    library = Path(_native.__file__).parent/'lib'/('libgradscf_integrals.dylib'
        if platform.system() == 'Darwin' else 'libgradscf_integrals.so')
    identity = dict(parent_checkpoint_sha256=base.pilot.file_hash(args.checkpoint),
        structures_sha256=base.pilot.file_hash(args.structures), structure_ids=[r['structure_id'] for r in rows],
        scientific_source_sha256=base.pilot.scientific_source_hash(ROOT),
        driver_sha256=base.pilot.file_hash(__file__), native_library_sha256=base.pilot.file_hash(library),
        imported_sources=base.pilot.verify_source_imports(ROOT), model_config=MODEL_CONFIG,
        exponent_keys=[list(k) for k in base.EXPONENT_KEYS], ntheta=len(theta), dtype='float64',
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                       if k not in {'resume'}},
        coefficient_gradient='native VJP + implicit SCF + MACE VJP',
        exponent_gradient='native VJP + implicit SCF', optimizer_reset=True,
        fixed_core_primitives=6, fixed_auxiliary_basis=True, primitive_cache=False)
    runtime = dict(hostname=platform.node(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        devices=[str(d) for d in jax.devices()], device_kinds=[d.device_kind for d in jax.devices()],
        cpu_devices=[str(d) for d in jax.devices('cpu')], jax_version=jax.__version__)
    base.pilot.atomic_json(args.output_dir/'runtime.json', runtime)
    base.pilot.atomic_json(args.output_dir/'metadata.json', identity)
    initial = np.concatenate((theta, np.zeros(len(base.EXPONENT_KEYS))))
    engine = NativeJointEngine(rows, args, parent_identity, None if args.resume else parent_energies)
    identity.update(trainable_mace_parameters=int(engine.trainable_mask.sum()),
                    frozen_parameter_indices=np.flatnonzero(~engine.trainable_mask).tolist())
    base.pilot.atomic_json(args.output_dir/'metadata.json', identity)
    if not args.resume:
        checks = [engine.check_gradient(initial, key) for key in ('W4-11/ch4', 'W4-11/n2')
                  if key in engine.contexts]
        base.pilot.atomic_json(args.output_dir/'gradient-checks.json', checks)
        print('Joint native/GPU gradient checks:', json.dumps(checks), flush=True)
    optimizer = JointAdam(len(theta), learning_rate=args.learning_rate,
        exponent_learning_rate=args.exponent_learning_rate, log_bound=args.log_bound,
        trainable_mask=engine.trainable_mask)
    summary = base.trainer.run_training(initial, engine.evaluate, optimizer, args.output_dir,
        identity, epochs=args.epochs, resume=args.resume)
    print(json.dumps({k: summary[k] for k in ('status', 'accepted_epoch', 'loss_hartree') if k in summary}), flush=True)
    return int(summary['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
