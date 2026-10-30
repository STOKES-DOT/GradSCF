"""Synchronized fixed-input matrix GW probes; no SCF/end-to-end speed claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("qsgw", "scgw", "charge"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--nmo", type=int, default=16)
    parser.add_argument("--naux", type=int, default=64)
    parser.add_argument("--nw", type=int, default=16)
    args = parser.parse_args()
    affinity = sorted(os.sched_getaffinity(0))
    if args.threads > len(affinity):
        raise ValueError("Thread request exceeds this allocation")
    os.sched_setaffinity(0, affinity[:args.threads])
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import gradscf
    from gradscf.gw.freq import scaled_legendre_grid
    from gradscf.gw.qsgw import _static_self_energy
    from gradscf.gw.matsubara import _screened
    target = jax.devices(args.device)[0]
    with jax.default_device(target):
        rng = np.random.default_rng(961)
        if args.kind == "qsgw":
            factors = rng.normal(size=(args.naux, args.nmo, args.nmo))*.03
            factors = jnp.asarray((factors+factors.swapaxes(-1, -2))*.5)
            no = args.nmo//3
            energy = jnp.asarray(np.r_[np.linspace(-2., -.3, no), np.linspace(.2, 2., args.nmo-no)])
            frequencies, weights = scaled_legendre_grid(args.nw)
            def function(b):
                return _static_self_energy(b_mn=b, b_ov=b[:, :no, no:], mo_energy=energy,
                    nocc=no, ef=.5*(energy[no-1]+energy[no]), freqs=frequencies, wts=weights, eta=.001)
            inputs = factors
        elif args.kind == "charge":
            from gradscf.gw.scgw import _solve_chemical_potential
            from gradscf.gw.matsubara import matsubara_grid
            grid = matsubara_grid(nw=args.nw, beta=10.)
            inputs = (jnp.diag(jnp.linspace(-.4, .6, args.nmo)),
                      jnp.zeros((2*args.nw, args.nmo, args.nmo), dtype=jnp.complex128))
            function = lambda arrays: _solve_chemical_potential(*arrays, -.25, grid, args.nmo, 1e-11)
        else:
            raw = rng.normal(size=(2*args.nw+1, args.naux, args.naux))*.005
            inputs = jnp.asarray(-raw@raw.swapaxes(-1, -2))
            function = _screened
        inputs = jax.block_until_ready(jax.device_put(inputs, target))
        if args.kind == "charge":
            # Eager API includes its internal compiled search and host check.
            executable, lower_seconds, compile_seconds, memory = function, None, None, None
        else:
            compiled = jax.jit(function)
            tick = time.perf_counter()
            lowered = compiled.lower(inputs)
            lower_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            executable = lowered.compile()
            compile_seconds = time.perf_counter()-tick
            memory = executable.memory_analysis()
        input_sha256 = hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree.leaves(inputs))).hexdigest()
        threads = {key: os.getenv(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "GRADSCF_NATIVE_THREADS", "JAX_PLATFORMS")}
        rows = []
        for repeat in range(4):
            tick = time.perf_counter()
            value = jax.block_until_ready(executable(inputs))
            rows.append(dict(kind=args.kind, timing_scope="fixed-input kernel", repeat=repeat,
                stage="first_execution" if repeat == 0 else "warm_execution",
                wall_seconds=time.perf_counter()-tick,
                norm=sum(float(jnp.linalg.norm(jnp.asarray(x))) for x in jax.tree.leaves(value)),
                finite=all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(value)), nmo=args.nmo, naux=args.naux,
                nw=args.nw, device=str(target), logical_cpus=sorted(os.sched_getaffinity(0)),
                threads=threads, input_sha256=input_sha256, dtype=str(jax.tree.leaves(inputs)[0].dtype)))
        rows.append(dict(stage="compile", lower_seconds=lower_seconds, compile_seconds=compile_seconds,
            xla_temporary_bytes=getattr(memory, "temp_size_in_bytes", None),
            xla_output_bytes=getattr(memory, "output_size_in_bytes", None),
            process_peak_rss_bytes=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            memory_scope="whole process including compilation; XLA estimate listed separately",
            python=os.path.realpath(os.sys.executable), source=gradscf.__file__,
            source_sha256=hashlib.sha256(Path(function.__code__.co_filename).read_bytes()).hexdigest(),
            gw_source_sha256={name: hashlib.sha256((Path(gradscf.__file__).parent/"gw"/(name+".py")).read_bytes()).hexdigest()
                              for name in ("qsgw", "matsubara", "scgw", "scgw_response", "self_energy", "screened")},
            fixture="synthetic fixed-input; not a molecule or complete GW workflow",
            jax=jax.__version__, jaxlib=jax.lib.__version__, dtype=str(jax.tree.leaves(inputs)[0].dtype),
            device_statistics=target.memory_stats(), input_sha256=hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree.leaves(inputs))).hexdigest()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in rows))


if __name__ == "__main__":
    main()
