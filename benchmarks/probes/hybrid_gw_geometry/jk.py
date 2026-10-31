"""Fixed synthetic AO density probe, not an independent molecular SCF run."""
import argparse
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
    args = parser.parse_args()
    available = sorted(os.sched_getaffinity(0))
    if args.threads > len(available) or max(args.workers) > args.threads:
        raise ValueError('Requested parallelism exceeds the Slurm allocation')
    os.sched_setaffinity(0, available[:args.threads])
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
    library = Path(register_integrals()._name).resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for omega in args.omega:
        for workers in args.workers:
            run = jax.jit(lambda d: provider.get_jk(d, workers=workers, omega=omega))
            tick = time.perf_counter()
            executable = run.lower(density).compile()
            compile_seconds = time.perf_counter() - tick
            for repeat in range(4):
                tick = time.perf_counter()
                value = jax.block_until_ready(executable(density))
                row = dict(stage='fixed-density-jk', repeat=repeat,
                    wall_seconds=time.perf_counter()-tick, compile_seconds=compile_seconds,
                    nao=top.nao, workers=workers, omega=omega, cutoff=0.,
                    affinity=sorted(os.sched_getaffinity(0)), dtype='float64',
                    source=integrals.__file__, native_path=str(library),
                    native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    density_sha256=hashlib.sha256(np.asarray(density).tobytes()).hexdigest(),
                    norms=[float(jnp.linalg.norm(x)) for x in value],
                    process_peak_rss_bytes=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                with args.output.open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
            np.savez(args.output.with_name(args.output.stem+f'-w{workers}-o{omega}.npz'),
                     coulomb=np.asarray(value[0]), exchange=np.asarray(value[1]))


if __name__ == '__main__':
    main()
