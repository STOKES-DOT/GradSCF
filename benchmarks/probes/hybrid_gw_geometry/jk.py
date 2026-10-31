"""Fixed synthetic AO density probe, not an independent molecular SCF run."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 32])
    parser.add_argument('--omega', type=float, nargs='+', default=[0., .2])
    parser.add_argument('--geometry-vjp', action='store_true')
    parser.add_argument('--density-checkpoint', type=Path)
    parser.add_argument('--exchange-fraction', type=float, default=1.)
    parser.add_argument('--max-memory', type=float,
                        help='Existing SCF memory allowance in MB; binds native private work.')
    args = parser.parse_args()
    available = sorted(os.sched_getaffinity(0))
    if args.threads > len(available) or max(args.workers) > args.threads:
        raise ValueError('Requested parallelism exceeds the Slurm allocation')
    os.sched_setaffinity(0, available[:args.threads])
    affinity = sorted(os.sched_getaffinity(0))
    cores = {(Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id').read_text().strip(),
              Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/core_id').read_text().strip())
             for cpu in affinity}
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import numpy as np
    from gradscf import integrals
    from gradscf.integrals._native import register_integrals
    from gradscf.integrals.backends.native.jk import NativeDirectBasis

    case = json.loads(args.case.read_text())
    top, parameters = integrals.prepare_basis(case['atom'], case['basis'], cart=True)
    provider = NativeDirectBasis(integrals.make_plan(top), parameters)
    rng = np.random.default_rng(91031)
    occupied = rng.normal(size=(top.nao, 47)) / np.sqrt(top.nao)
    density = jnp.asarray(2 * occupied @ occupied.T)
    if args.density_checkpoint:
        with np.load(args.density_checkpoint, allow_pickle=False) as saved:
            density = jnp.asarray(saved['density'])
        if density.shape != (top.nao, top.nao) or jnp.iscomplexobj(density):
            raise ValueError('Use a real spatial AO density matching this basis.')
    owners = np.argmin(np.linalg.norm(np.asarray(parameters.centers)[:, None]
        - np.asarray(parameters.nuclear_coords)[None], axis=2), axis=1)
    library = Path(register_integrals()._name).resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for omega in args.omega:
        for workers in args.workers:
            from gradscf.integrals.backends.native.jk import native_jk_workspace_bytes
            selected = provider
            if args.max_memory is not None:
                from gradscf.integrals.backends.native.workspace import bind_jk_workspace
                selected = bind_jk_workspace(provider, workers=workers, max_memory=args.max_memory,
                                             resident_bytes=density.nbytes)
            if args.geometry_vjp:
                def energy(r, d):
                    moving = replace(selected,
                        parameters=replace(parameters, centers=r[owners], nuclear_coords=r))
                    j, k = moving.get_jk(d, workers=workers, omega=omega)
                    return .5*jnp.sum(d*j)-.25*args.exchange_fraction*jnp.sum(d*k)
                run = jax.jit(jax.grad(energy, argnums=0))
                inputs = (parameters.nuclear_coords, density)
            else:
                run = jax.jit(lambda d: selected.get_jk(d, workers=workers, omega=omega))
                inputs = (density,)
            tick = time.perf_counter()
            executable = run.lower(*inputs).compile()
            compile_seconds = time.perf_counter() - tick
            for repeat in range(4):
                usage = resource.getrusage(resource.RUSAGE_SELF)
                tick = time.perf_counter()
                value = jax.block_until_ready(executable(*inputs))
                elapsed = time.perf_counter()-tick
                finished = resource.getrusage(resource.RUSAGE_SELF)
                arrays = (value,) if args.geometry_vjp else value
                row = dict(stage='fixed-density-geometry-vjp' if args.geometry_vjp else 'fixed-density-jk', repeat=repeat,
                    wall_seconds=elapsed, compile_seconds=compile_seconds,
                    cpu_user_seconds=finished.ru_utime-usage.ru_utime,
                    cpu_system_seconds=finished.ru_stime-usage.ru_stime,
                    nao=top.nao, workers=workers, omega=omega, cutoff=0.,
                    max_memory_mb=args.max_memory,
                    private_output_bytes=getattr(selected, 'private_output_bytes', None),
                    planned_native_workspace_bytes=native_jk_workspace_bytes(selected, workers, 0.),
                    affinity=affinity, physical_core_count=len(cores), dtype='float64',
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    source=integrals.__file__, native_path=str(library),
                    native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    density_sha256=hashlib.sha256(np.asarray(density).tobytes()).hexdigest(),
                    case_sha256=hashlib.sha256(args.case.read_bytes()).hexdigest(),
                    density_checkpoint_sha256=(hashlib.sha256(args.density_checkpoint.read_bytes()).hexdigest()
                        if args.density_checkpoint else None),
                    scope='Fixed-density integral product; excludes orbital/Pulay/grid/nuclear-energy response.',
                    exchange_fraction=args.exchange_fraction,
                    norms=[float(jnp.linalg.norm(x)) for x in arrays],
                    process_peak_rss_bytes=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                with args.output.open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
            payload = dict(gradient=np.asarray(value)) if args.geometry_vjp else dict(
                coulomb=np.asarray(value[0]), exchange=np.asarray(value[1]))
            np.savez(args.output.with_name(args.output.stem+f'-w{workers}-o{omega}.npz'), **payload)


if __name__ == '__main__':
    main()
