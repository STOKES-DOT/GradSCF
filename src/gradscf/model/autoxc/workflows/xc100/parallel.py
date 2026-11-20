"""One GPU coordinator and persistent CPU shards inside one Slurm step."""
from contextlib import ExitStack
import json
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zipfile import ZipFile

from .schedule import affinity_plan, distribute


def rank_environment(mask, *, gpu):
    env = dict(os.environ, JAX_PLATFORMS='cuda' if gpu else 'cpu')
    if not gpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[variable] = str(len(mask))
    return env


def make_shard(root, options, rank):
    import jax
    import numpy as np

    from gradscf.model.autoxc import Candidate
    from gradscf.model.autoxc.loop import sample_reports, stream_value_and_grad
    from ...history import write_json
    from .campaign import Campaign, memory

    class Shard(Campaign):
        def status(self, phase, **details):
            write_json(self.root / 'ranks' / f'{rank}.json',
                       {'phase': phase, 'prepared': len(self.rows), **details})

        def qualify(self, identifiers):
            self.prepare_cases(identifiers, receipt=None, persist=False)
            return self.rows, memory()

        def bind_payload(self, payload):
            candidate = Candidate.from_json(payload)
            self.callbacks = self.bind(candidate)
            names = tuple(candidate.initial)
            self.gradient = stream_value_and_grad(
                self.callbacks, tuple(self.cases), lambda vector: dict(zip(names, vector)))
            return self.cached_affine(candidate)

        def partial(self, vector):
            from jax import numpy as jnp
            start = time.perf_counter()
            value, gradient = self.gradient(jnp.asarray(vector))
            return float(value), np.asarray(gradient), time.perf_counter() - start

        def reports(self, parameters):
            return sample_reports(self.callbacks, tuple(self.cases), parameters)

        def calibrate(self, identifiers, fixtures):
            rows, _ = self.qualify(identifiers)
            measurements = {}
            for label, payload in fixtures.items():
                candidate = Candidate.from_json(payload)
                names = tuple(candidate.initial)
                vector = jax.numpy.asarray([candidate.initial[key] for key in names])
                callbacks = self.bind(candidate)
                points = []
                for identifier in identifiers:
                    score_and_grad = stream_value_and_grad(
                        callbacks, (identifier,), lambda v: dict(zip(names, v)))
                    jax.block_until_ready(score_and_grad(vector))
                    start = time.perf_counter()
                    for _ in range(3):
                        value, gradient = jax.block_until_ready(score_and_grad(vector))
                    points.append({'system_id': identifier, 'seconds': (time.perf_counter() - start) / 3,
                                   'loss': float(value), 'gradient': np.asarray(gradient).tolist()})
                measurements[label] = points
            self.cases.clear()
            self.rows = []
            return rows, measurements

    return Shard(root, **options)


def child_main(pipe, root, options, rank):
    import jax

    with pipe:
        if jax.default_backend() != 'cpu':
            raise RuntimeError(f'Shard {rank} requires JAX CPU')
        print(f'CPU shard {rank}: {sorted(os.sched_getaffinity(0))}', flush=True)
        shard = make_shard(root, options, rank)
        while True:
            command, arguments = pipe.recv()
            if command == 'close':
                return
            pipe.send(invoke(shard, command, arguments))


def invoke(shard, command, arguments):
    from .campaign import CampaignReleased

    try:
        return None, getattr(shard, command)(*arguments)
    except CampaignReleased:
        return 'release', None
    except (ValueError, FloatingPointError) as exc:
        return 'numerical', str(exc)
    except Exception:
        return 'worker', traceback.format_exc()


class Workers:
    def __init__(self, root, options, masks):
        self.local = make_shard(root, options, 0)
        self.wait_timeout = None if options.get('unbounded_waits') else 1800
        self.children = []
        try:
            for rank, mask in enumerate(masks[1:], 1):
                pipe, child = Pipe()
                try:
                    with child:
                        descriptor = child.fileno()
                        process = subprocess.Popen(
                            ['taskset', '--cpu-list', ','.join(map(str, mask)), sys.executable,
                             '-m', __name__, str(descriptor), str(root), str(rank), json.dumps(options)],
                            env=rank_environment(mask, gpu=False), pass_fds=(descriptor,))
                except BaseException:
                    pipe.close()
                    raise
                self.children.append((pipe, process))
        except BaseException:
            self.close()
            raise

    def call(self, command, arguments, *, ranks=None, sequential=False):
        from .campaign import CampaignReleased

        ranks = tuple(range(1 + len(self.children))) if ranks is None else tuple(ranks)
        requests = tuple(zip(ranks, arguments, strict=True))
        if not sequential:
            for rank, args in requests:
                if rank:
                    self.children[rank - 1][0].send((command, args))
        results = []
        for rank, args in requests:
            if rank == 0:
                results.append(invoke(self.local, command, args))
            else:
                pipe, process = self.children[rank - 1]
                if sequential:
                    pipe.send((command, args))
                if self.wait_timeout is not None and not pipe.poll(self.wait_timeout):
                    raise RuntimeError(f'CPU shard {rank} timed out during {command}')
                try:
                    results.append(pipe.recv())
                except EOFError as exc:
                    raise RuntimeError(f'CPU shard {rank} exited during {command}: {process.poll()}') from exc
        for error, message in results:
            if error == 'release':
                raise CampaignReleased()
            if error == 'worker':
                raise RuntimeError(message)
            if error == 'numerical':
                raise ValueError(message)
        return [result for _, result in results]

    def close(self):
        for pipe, process in self.children:
            if process.poll() is None:
                try:
                    pipe.send(('close', ()))
                except (OSError, EOFError):
                    pass
        for pipe, process in self.children:
            pipe.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


class BoundBatch:
    def __init__(self, workers, sample_ids):
        self.workers = workers
        self.sample_ids = sample_ids
        self.last_shard_seconds = ()

    def __iter__(self):
        return iter(self.sample_ids)

    def value_and_grad(self, vector, *, sequential=False):
        import math
        import numpy as np

        packed = np.asarray(vector)
        chunks = self.workers.call('partial', [(packed,)] * (1 + len(self.workers.children)),
                                   sequential=sequential)
        self.last_shard_seconds = tuple(chunk[2] for chunk in chunks)
        size = len(self.sample_ids)
        return (math.fsum(chunk[0] for chunk in chunks) / size,
                np.sum([chunk[1] for chunk in chunks], axis=0) / size)

    def sample_reports(self, parameters):
        reports = {}
        args = [(parameters,)] * (1 + len(self.workers.children))
        for partial in self.workers.call('reports', args):
            if reports.keys() & partial.keys():
                raise RuntimeError('Duplicate molecule in the shard reports')
            reports.update(partial)
        return reports


def run_parallel(root):
    """Use the Slurm step's actual CPU mask as the allocation envelope."""
    root = Path(root)
    expected = int(os.environ['SLURM_CPUS_PER_TASK'])
    allowed = os.sched_getaffinity(0)
    if len(allowed) != expected:
        raise RuntimeError(f'Expected the job step to bind {expected} CPUs, got {len(allowed)}')
    masks = affinity_plan(allowed)
    os.sched_setaffinity(0, masks[0])
    os.environ.update(rank_environment(masks[0], gpu=True))

    import jax
    import numpy as np

    from gradscf.model.autoxc import Candidate, Parameter
    from ...history import write_json
    from .campaign import Campaign, CampaignReleased, IDS, LDA_EXCHANGE_POTENTIAL, memory
    from .molecule import file_hash

    class ParallelCampaign(Campaign):
        def prepare(self):
            self.check_release()
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1 or not jax.config.jax_enable_x64:
                raise RuntimeError('Parallel qualification requires one GPU and float64')
            self.status('calibrating')
            base = LDA_EXCHANGE_POTENTIAL + '+' if self.standalone_potential else ''
            nonlinear = 'a*n**0.3333333333333333*tanh(b*u + .2*z)'
            definitions = ()
            if not self.standalone_potential:
                nonlinear = 'a*n**0.3333333333333333*tanh(b*u + .2*z + .1*c)'
                definitions = (('c', '(rho_occ-n)/(rho_occ+n)'),)
            fixtures = {
                'affine': Candidate(base+'n**0.3333333333333333*(a*u + b*z)',
                    parameters=(Parameter('a', .003, -.05, .05),
                                Parameter('b', -.002, -.05, .05))).to_dict(),
                'nonlinear': Candidate(base+nonlinear, definitions=definitions,
                    parameters=(Parameter('a', .015, -.1, .1),
                                Parameter('b', .8, .1, 3.))).to_dict(),
            }
            if self.candidate_mode == 'open_xc':
                fixtures['energy'] = Candidate('pbe_x*exp(a*u) + pbe_c', kind='energy',
                    parameters=(Parameter('a', .03, -.5, .5),)).to_dict()
            probe_ids = ('010', '017', '065')
            probe_ranks = (0, 1) if len(masks) > 1 else (0,)
            calibration = self.workers.call('calibrate',
                [(probe_ids, fixtures)] * len(probe_ranks), ranks=probe_ranks)
            gpu_rows, gpu_checks = calibration[0]
            cpu_checks = calibration[-1][1]
            for label in fixtures:
                for gpu_point, cpu_point in zip(gpu_checks[label], cpu_checks[label], strict=True):
                    for point in (gpu_point, cpu_point):
                        if not np.isfinite([point['loss'], *point['gradient']]).all():
                            raise ValueError('Nonfinite score or gradient in shard calibration')
                    np.testing.assert_allclose(gpu_point['loss'], cpu_point['loss'], atol=2e-8, rtol=1e-7)
                    np.testing.assert_allclose(gpu_point['gradient'], cpu_point['gradient'], atol=1e-7, rtol=2e-5)
            # Normalize costs against measured shapes, then keep the same shards across fits.
            def weights(row):
                matrix = (row['nao'] / 100) ** 3
                grid_work = row['grid_points'] / 1e5 * (row['nao'] / 100) ** 2
                return {label: 1 + matrix + (0 if label == 'affine' else grid_work)
                        for label in fixtures}

            def timing_scale(checks):
                return {label: float(np.median([point['seconds'] / weights(row)[label]
                         for point, row in zip(checks[label], gpu_rows, strict=True)]))
                        for label in fixtures}

            gpu_scale, cpu_scale = timing_scale(gpu_checks), timing_scale(cpu_checks)
            costs = {}
            for identifier in IDS:
                folder = self.data / identifier
                row = json.loads((folder / 'metadata.json').read_text())
                with ZipFile(folder / 'density.npz') as archive, archive.open('grid_weights.npy') as handle:
                    version = np.lib.format.read_magic(handle)
                    header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    row['grid_points'] = header(handle)[0][0]
                costs[identifier] = [sum(weights(row)[label] * scale[label] for label in fixtures)
                                     for scale in [gpu_scale] + [cpu_scale] * (len(masks) - 1)]
            self.shards = distribute(costs, len(masks))
            self.execution = {'cpu_masks': masks, 'sample_shards': self.shards,
                              'device_scales': {'gpu': gpu_scale, 'cpu': cpu_scale}}
            write_json(root / 'execution.json', self.execution)
            self.status('preparing_shards')
            qualifications = self.workers.call('qualify', [(ids,) for ids in self.shards])
            self.rows = sorted((row for rows, _ in qualifications for row in rows), key=lambda r: r['system_id'])
            if tuple(row['system_id'] for row in self.rows) != IDS:
                raise RuntimeError('Shard qualification did not cover XC100 exactly once')
            self.execution['qualification_memory'] = [stats for _, stats in qualifications]
            for filename in ('parallel.py', 'schedule.py'):
                self.helpers[filename] = file_hash(Path(__file__).with_name(filename))
            self.status('parallel_acceptance')
            preflight = {'baseline': 'fresh PBE', 'fixture_source': 'local independent probes',
                         'calibration': {'gpu': gpu_checks, 'cpu': cpu_checks}, 'candidates': {}}
            for label, payload in fixtures.items():
                candidate = Candidate.from_json(payload)
                parameters = candidate.initial
                vector = np.array([parameters[name] for name in candidate.initial])
                start = time.perf_counter()
                batch = self.bind(candidate)
                bind_seconds = time.perf_counter() - start
                batch.value_and_grad(vector)
                start = time.perf_counter()
                serial = batch.value_and_grad(vector, sequential=True)
                serial_seconds = time.perf_counter() - start
                start = time.perf_counter()
                parallel = batch.value_and_grad(vector)
                parallel_seconds = time.perf_counter() - start
                np.testing.assert_allclose(parallel[0], serial[0], atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(parallel[1], serial[1], atol=1e-12, rtol=1e-12)
                step = 1e-4
                shift = np.zeros_like(vector)
                shift[0] = step
                plus, _ = batch.value_and_grad(vector + shift)
                minus, _ = batch.value_and_grad(vector - shift)
                finite_difference = (plus - minus) / (2 * step)
                np.testing.assert_allclose(parallel[1][0], finite_difference, atol=1e-7, rtol=2e-5)
                reports = batch.sample_reports(parameters)
                if tuple(sorted(reports)) != IDS or not all(row['valid'] for row in reports.values()):
                    raise ValueError('Full-batch parallel diagnostics failed')
                np.testing.assert_allclose(-parallel[0],
                    np.mean([reports[identifier]['reward'] for identifier in IDS]), atol=1e-10, rtol=1e-9)
                preflight['candidates'][label] = dict(reward=-parallel[0],
                    candidate=payload, gradient=parallel[1].tolist(), finite_difference=finite_difference,
                    finite_difference_step=step,
                    serial_shard_seconds=serial_seconds, parallel_seconds=parallel_seconds,
                    bind_seconds=bind_seconds, shard_seconds=batch.last_shard_seconds)
            self.parallel_preflight = preflight
            self.finish_preparation(receipt=None)

        def validate_prepared(self):
            write_json(root / 'preflight.json', self.parallel_preflight)
            return self.parallel_preflight['candidates']['affine']['reward']

        def bind(self, candidate, sample_ids=None):
            self.check_release()
            self.status('binding_potential', candidate_digest=candidate.digest)
            self.cached = self.workers.call('bind_payload', [(candidate.to_dict(),)] * len(masks))
            self.status('fitting', candidate_digest=candidate.digest, bound_systems=len(IDS))
            return BoundBatch(self.workers, IDS)

        def cached_affine(self, candidate):
            return all(self.cached)

    options = json.loads((root / 'campaign.json').read_text())
    campaign = ParallelCampaign(root, **options)
    with ExitStack() as cleanup:
        try:
            campaign.workers = Workers(root, options, masks)
            cleanup.callback(campaign.workers.close)
            campaign.serve()
        except CampaignReleased:
            campaign.status('released')
        except BaseException as error:
            campaign.status('failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    fd, directory, rank, options = sys.argv[1:]
    child_main(Connection(int(fd)), directory, json.loads(options), int(rank))
