"""Isolated native ERI product probe; all integral/AD rules come from src.

Basis variables are independent libcint-normalized coefficients and exponents:
exponent products hold normalized coefficients fixed, without a raw-normalizer
chain. These are fixed-input integral products, not relaxed molecular response.
Run every kind/layout/build/thread configuration in its own process.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback


KINDS = ('coefficient_jvp', 'coefficient_vjp', 'coefficient_hessian_jvp',
         'coefficient_hessian_vjp', 'exponent_jvp', 'exponent_vjp',
         'geometry_jvp', 'geometry_vjp', 'hessian_jvp', 'hessian_vjp')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=KINDS, required=True)
    parser.add_argument('--basis', default='def2-svp')
    parser.add_argument('--atom', default='O 0 0 0; H 0 -.757 .587; H 0 .757 .587')
    parser.add_argument('--unit', choices=('Angstrom', 'Bohr'), default='Angstrom')
    parser.add_argument('--layout', choices=('full', 'window', 's4', 's8'), default='full')
    parser.add_argument('--window', nargs=8, type=int)
    parser.add_argument('--spherical', action='store_true')
    parser.add_argument('--threads', type=int, help='native worker count; affinity/BLAS are controlled by the caller')
    parser.add_argument('--counts', action='store_true', help='requires separate --diagnostics build')
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--output', type=Path, required=True, help='JSON report path; numeric arrays use a sibling .npz')
    args = parser.parse_args()
    if args.output.suffix != '.json':
        parser.error('--output must end in .json so the report cannot overwrite its numeric .npz sibling')
    basis_kind = args.kind.startswith(('coefficient', 'exponent'))
    if args.layout not in (('full', 'window') if basis_kind else ('full', 's4', 's8')):
        parser.error('Basis products use full/window; coordinate products use full/s4/s8')
    if args.window is not None and args.layout != 'window':
        parser.error('--window applies only to the basis window layout')
    if args.threads is not None:
        if args.threads < 1 or (hasattr(os, 'sched_getaffinity') and args.threads > len(os.sched_getaffinity(0))):
            parser.error('--threads must fit the caller-owned CPU affinity')
        os.environ['GRADSCF_NATIVE_THREADS'] = str(args.threads)
    os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
    os.environ['JAX_ENABLE_X64'] = '1'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = dict(status='preparing', command=sys.argv, config=vars(args).copy(), samples=[],
        counts=[], timing_eligible=False, scope='fixed-input ERI products; CPU native kernels; no SCF relaxation',
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        coefficient_chart='independent libcint-normalized ENV coefficients',
        exponent_chart='Bohr^-2; partial at fixed normalized coefficients',
        coordinate_chart='Bohr; nuclear positions plus independent shell centers',
        timing_contract='first JIT lower+compile+execute, then three synchronized compiled calls')
    report['config']['output'] = str(args.output)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    save()
    try:
        import numpy as np
        import jax
        import jax.numpy as jnp
        from gradscf import integrals
        from gradscf.integrals._native import diagnostics
        from gradscf.integrals.backends.native.autodiff.eri_basis import evaluate_eri_basis
        from gradscf.integrals.backends.native.autodiff.geometry import evaluate_geometry
        from probes import CounterAdapter, counted_call
        jax.config.update('jax_enable_x64', True)
        jax.config.update('jax_enable_compilation_cache', False)
        cpu = jax.devices('cpu')[0]
        jax.config.update('jax_default_device', cpu)
        ready = jax.block_until_ready
        counter = CounterAdapter(diagnostics.reset_counts, diagnostics.get_counts) if args.counts else None
        if counter is not None:
            report['native'] = {k: x for k, x in counter.snapshot().items() if k != 'families'}
        tick = time.perf_counter()
        topology, parameters = integrals.prepare_basis(args.atom, args.basis,
            unit=args.unit, cart=not args.spherical)
        plan = integrals.make_plan(topology)
        fixed = replace(parameters, centers=jnp.zeros_like(parameters.centers),
                        nuclear_coords=jnp.zeros_like(parameters.nuclear_coords))
        atoms, basis, env, coefficients, exponents = plan.coefficient_data(fixed, separate_exponents=True)
        coordinates = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
        ns = len(topology.angular_momenta)
        window = None
        if args.layout == 'window':
            window = tuple(args.window or (0, 1, min(1, ns-1), min(3, ns),
                                           0, min(2, ns), max(0, ns-2), ns))
        compact = {'s4': 0, 's8': 1}.get(args.layout)
        if not basis_kind:
            atoms, basis, env = plan.pack(fixed)
        if window is not None:
            from gradscf.integrals.backends.native.ffi import _eri_shell_slice
            window, shape = _eri_shell_slice(basis, topology.cart, window)
        elif compact is not None:
            pairs = topology.nao*(topology.nao+1)//2
            shape = (pairs, pairs) if compact == 0 else (pairs*(pairs+1)//2,)
        else:
            shape = (topology.nao,)*4
        size = coefficients.size if args.kind.startswith('coefficient') else exponents.size if basis_kind else coordinates.size
        rng = np.random.default_rng(args.seed)
        direction_shape = (size,) if basis_kind else coordinates.shape
        u, v = (jnp.asarray(rng.normal(scale=.03, size=direction_shape)) for _ in range(2))
        cotangent = jnp.asarray(rng.normal(size=shape))  # Arbitrary, nonsymmetric cotangent.
        inputs = ready((env, coefficients, exponents, coordinates, u, v, cotangent))
        report.update(preparation_seconds=time.perf_counter()-tick, nao=topology.nao, nshell=ns,
            ncoeff=int(coefficients.size), nexponents=int(exponents.size), integral_shape=list(shape),
            shell_slice=window, cart=topology.cart, device=str(cpu), device_kind=cpu.device_kind,
            dtype='float64', direction_shape=list(direction_shape), persistent_cache=False,
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            environment={k: os.environ.get(k) for k in ('GRADSCF_NATIVE_THREADS', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'JAX_PLATFORMS', 'XLA_FLAGS')})

        def operation(environment, c, a, r, left, right, weight):
            if basis_kind:
                if args.kind.startswith('coefficient'):
                    point = c
                    # Differentiate only c. Alpha remains a dynamic, unrelated
                    # argument; requesting both argnums would introduce mixed AD.
                    values = lambda x: evaluate_eri_basis(atoms, basis, environment, x, r,
                        jnp.zeros(3), topology.nao, cart=topology.cart, exponents=a, shell_slice=window)
                else:
                    point = a
                    values = lambda x: evaluate_eri_basis(atoms, basis, environment, c, r,
                        jnp.zeros(3), topology.nao, cart=topology.cart, exponents=x, shell_slice=window)
            else:
                point = r
                values = lambda x: evaluate_geometry('eri', atoms, basis, environment, x,
                    jnp.zeros(3), topology.nao, cart=topology.cart, compact_layout=compact)
            second = 'hessian' in args.kind
            if args.kind.endswith('_vjp'):
                if second:
                    gradient = jax.grad(lambda x: jnp.vdot(weight, values(x)).real)
                    return jax.jvp(gradient, (point,), (left,))[1]
                return jax.vjp(values, point)[1](weight)[0]
            if second:
                return jax.jvp(lambda x: jax.jvp(values, (x,), (left,))[1], (point,), (right,))[1]
            return jax.jvp(values, (point,), (left,))[1]

        report['status'] = 'running'
        save()
        cold_start = time.perf_counter()
        tick = time.perf_counter()
        lowered = jax.jit(operation).lower(*inputs)
        report['lower_seconds'] = time.perf_counter()-tick
        tick = time.perf_counter()
        compiled = lowered.compile()
        report['compile_seconds'] = time.perf_counter()-tick
        tick = time.perf_counter()
        cold = ready(compiled(*inputs))
        report['first_execution_seconds'] = time.perf_counter()-tick
        report['cold_first_total_seconds'] = time.perf_counter()-cold_start
        result = cold
        for index in range(3):
            tick = time.perf_counter()
            if counter is None:
                result = ready(compiled(*inputs))
            else:
                result, snapshot = counted_call(lambda: compiled(*inputs), ready=ready, counter=counter)
                report['counts'].append(dict(warm=index, snapshot=snapshot))
            report['samples'].append(dict(warm=index, seconds=time.perf_counter()-tick))
        values = np.asarray(result)
        if not np.isfinite(values).all():
            raise FloatingPointError('Product result is nonfinite')
        archive = args.output.with_suffix('.npz')
        np.savez(archive, result=values, cold_result=np.asarray(cold), env=np.asarray(env),
            coefficients=np.asarray(coefficients), exponents=np.asarray(exponents),
            coordinates=np.asarray(coordinates), direction_u=np.asarray(u), direction_v=np.asarray(v),
            cotangent=np.asarray(cotangent), atoms=np.asarray(atoms), basis=np.asarray(basis))
        identity = hashlib.sha256()
        for value in (*inputs, atoms, basis):
            array = np.ascontiguousarray(value)
            identity.update(repr((array.shape, array.dtype.str)).encode()+array.tobytes())
        native = diagnostics.get_counts()  # After clean timings: do not preload the library.
        import gradscf
        root = Path(__file__).resolve().parents[2]
        revision = root/'SOURCE_REVISION'
        report.update(status='complete', result_shape=list(values.shape), output_norm=float(np.linalg.norm(values.ravel())),
            scalar_checksum=float(values.ravel()@np.linspace(.3, 1.3, values.size)),
            arrays=str(archive), arrays_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            numerical_input_sha256=identity.hexdigest(),
            native={k: x for k, x in native.items() if k != 'families'},
            timing_eligible=not args.counts and not native['enabled'],
            source_path=str(Path(gradscf.__file__).resolve()), probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_revision=revision.read_text().strip() if revision.exists() else
                subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
            jax_version=jax.__version__, numpy_version=np.__version__)
    except Exception as error:
        report.update(status='failed', exception=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report.update(process_seconds=time.perf_counter()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
            memory_method='process high-water RSS including preparation/compilation/repeats and output checks')
        save()
    print(json.dumps(dict(status=report['status'], kind=args.kind, layout=args.layout, output=str(args.output))))


if __name__ == '__main__':
    main()
