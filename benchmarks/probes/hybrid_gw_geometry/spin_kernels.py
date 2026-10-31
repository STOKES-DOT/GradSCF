"""Shared-input UKS/GKS kernels; no molecular SCF or integral preparation."""
import argparse
import hashlib
import inspect
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--kernel', choices=('uks-jk', 'gks-pure', 'gks-hybrid', 'gks-energy'), required=True)
    parser.add_argument('--nao', type=int, default=30)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--ad', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path; existing raw measurements are immutable.')
    allowed = sorted(os.sched_getaffinity(0))
    if min(args.nao, args.threads, args.repeats) < 1 or args.threads > len(allowed):
        parser.error('Positive dimensions/repeats and CPUs within the allocation are required.')
    if args.ad and args.nao > 30:
        parser.error('Array AD probes are intentionally bounded to at most 30 spatial AOs.')
    os.sched_setaffinity(0, allowed[:args.threads])
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('JAX_ENABLE_COMPILATION_CACHE', 'false')
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import numpy as np
    from gradscf.scf import gks, uks
    from gradscf.integrals._native import register_integrals

    target = jax.devices(args.device)[0]
    rng, n = np.random.default_rng(1103), args.nao
    factors = rng.normal(size=(16, n, n))*.02
    factors = (factors+factors.transpose(0, 2, 1))*.5
    p, q = np.tril_indices(n)
    pairs = factors[:, p, q]
    pair_matrix = pairs.T@pairs
    data = dict(eri=pair_matrix[np.tril_indices(len(p))])
    nalpha, nbeta = n//4+1, n//4
    if args.kernel == 'uks-jk':
        a, b = rng.normal(size=(n, nalpha))/np.sqrt(n), rng.normal(size=(n, nbeta))/np.sqrt(n)
        data['density'] = np.stack([a@a.T, b@b.T])
    else:
        occupied = rng.normal(size=(2*n, nalpha+nbeta))+1j*rng.normal(size=(2*n, nalpha+nbeta))
        occupied, _ = np.linalg.qr(occupied)
        ao = rng.normal(size=(257, n))*.02
        data.update(density=occupied@occupied.conj().T, hcore=np.diag(np.linspace(-1., 1., n)),
            ao=ao, ao_deriv1=np.stack([ao, .11*ao, -.07*ao, .03*ao]),
            grid_weights=np.full(257, 1/257))
    input_hash = hashlib.sha256()
    for key, value in sorted(data.items()):
        input_hash.update(key.encode()+str((value.shape, value.dtype)).encode())
        input_hash.update(np.ascontiguousarray(value).tobytes())
    library = Path(register_integrals()._name).resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    unrestricted = args.kernel == 'uks-jk'
    common = dict(kernel=args.kernel, nao=n,
        nalpha=nalpha if unrestricted else None, nbeta=nbeta if unrestricted else None,
        nocc_spinor=None if unrestricted else nalpha+nbeta,
        nvir_spinor=None if unrestricted else 2*n-nalpha-nbeta,
        ngrid=0 if args.kernel == 'uks-jk' else 257,
        nvir_alpha=n-nalpha if unrestricted else None, nvir_beta=n-nbeta if unrestricted else None,
        input_sha256=input_hash.hexdigest(), device=str(target), dtype='float64/complex128',
        input_shapes={k: list(v.shape) for k, v in data.items()},
        input_dtypes={k: str(v.dtype) for k, v in data.items()},
        versions={name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'numpy')},
        affinity=sorted(os.sched_getaffinity(0)), integral_representation='s8',
        approximation='none in measured contraction; synthetic symmetry-valid ERI',
        timing_scope='shared-input kernel; no Gaussian integrals or SCF',
        native_path=str(library), native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        source={name: dict(path=inspect.getfile(module),
            sha256=hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest())
            for name, module in (('uks', uks), ('gks', gks))})
    def record(**row):
        with args.output.open('a') as stream:
            stream.write(json.dumps(dict(common, **row), allow_nan=False)+'\n')
    tick = time.perf_counter()
    data = jax.block_until_ready(jax.device_put(data, target))
    record(stage='input_transfer', wall_seconds=time.perf_counter()-tick,
           input_bytes=sum(x.nbytes for x in data.values()))
    energy_only = args.kernel == 'gks-energy'
    supports_energy_only = 'compute_potential' in inspect.signature(gks.generalized_energy_and_fock).parameters
    config = gks.GKSConfig(xc_spec='pbe' if args.kernel == 'gks-pure' else 'pbe0')
    def kernel(values):
        if args.kernel == 'uks-jk':
            return uks._spin_jk_matrices(values['eri'], *values['density'])
        options = dict(compute_potential=False) if energy_only and supports_energy_only else {}
        result = gks.generalized_energy_and_fock(**values, nuclear_repulsion=0., config=config, **options)
        return result[0] if energy_only else (result[0], result[2])
    with jax.default_device(target):
        tick = time.perf_counter(); lowered = jax.jit(kernel).lower(data)
        record(stage='lower', wall_seconds=time.perf_counter()-tick)
        tick = time.perf_counter(); executable = lowered.compile()
        record(stage='compile', wall_seconds=time.perf_counter()-tick,
            xla_temp_bytes=executable.memory_analysis().temp_size_in_bytes)
        for repeat in range(args.repeats+1):
            tick = time.perf_counter(); output = jax.block_until_ready(executable(data))
            elapsed = time.perf_counter()-tick
            record(stage='action', repeat=repeat, wall_seconds=elapsed)
        arrays = [np.asarray(x) for x in jax.tree.leaves(output)]
        np.savez(args.output.with_suffix('.npz'), **{f'output{i}': x for i, x in enumerate(arrays)})
        updated = jax.block_until_ready(executable(dict(data, density=data['density']*1.001)))
        change = max(float(np.max(np.abs(np.asarray(a)-b))) for a, b in zip(jax.tree.leaves(updated), arrays))
        record(stage='dynamic_input_check', difference=change,
            scientific_pass=bool(change > 0 and all(np.isfinite(x).all() for x in arrays)))
        if args.ad:
            for channel in ('eri', 'density'):
                def loss(theta, values):
                    moved = dict(values, **{channel: values[channel]*(1+.01*theta)})
                    return sum(jnp.sum(x).real for x in jax.tree.leaves(kernel(moved)))
                tick = time.perf_counter()
                lowered_ad = jax.jit(jax.value_and_grad(loss, argnums=0)).lower(jnp.asarray(0.), data)
                record(stage='ad_lower', channel=channel, wall_seconds=time.perf_counter()-tick)
                tick = time.perf_counter(); derivative = lowered_ad.compile()
                record(stage='ad_compile', channel=channel, wall_seconds=time.perf_counter()-tick)
                for repeat in range(4):
                    tick = time.perf_counter(); value, grad = jax.block_until_ready(derivative(jnp.asarray(0.), data))
                    record(stage='value_and_grad', channel=channel, repeat=repeat,
                        wall_seconds=time.perf_counter()-tick, value=float(value), gradient=float(grad))
                tangent = jax.jit(lambda t, p: jax.jvp(lambda x: loss(x, p), (t,), (jnp.ones_like(t),))[1])
                error = abs(float(tangent(jnp.asarray(0.), data))-float(grad))
                record(stage='jvp_vjp_check', channel=channel, absolute_error=error,
                    scientific_pass=bool(np.isfinite(error) and error <= 1e-8+1e-9*abs(float(grad))))
                finite_value = jax.jit(loss)
                for step in (3e-4, 1e-4):
                    finite = float((finite_value(step, data)-finite_value(-step, data))/(2*step))
                    error = abs(float(grad)-finite)
                    record(stage='array_fd', channel=channel, step=step, absolute_error=error,
                        scientific_pass=bool(np.isfinite(finite) and error <= 1e-7+1e-6*abs(finite)))
    record(stage='process_peak', cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        gpu_allocator_statistics=target.memory_stats() if args.device == 'gpu' else None)


if __name__ == '__main__':
    main()
