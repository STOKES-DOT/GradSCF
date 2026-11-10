"""Forward ordinary direct J/K at D and 1e-4 D; not SCF or normalized densities.
Run in CPU64 Slurm under the existing 300-second process/RSS guard:
  python probe_direct_delta.py frozen-grad-direct.json density.npy new-output
The same provider, batch shape, cutoff and combined J/K selection are reused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'density', 'output'): parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if not os.getenv('SLURM_JOB_ID'): parser.error('Numerical probes require Slurm')
    if (cfg['engine'], cfg['representation'], cfg['threads'], cfg['direct_scf_tol']) != ('gradscf', 'direct', 64, 1e-12):
        parser.error('Requires frozen GradSCF direct/CPU64/cutoff1e-12 input')
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 64: parser.error('CPU64 allocation is required')
    os.sched_setaffinity(0, cpus[:64])
    os.environ.update(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1', JAX_ENABLE_COMPILATION_CACHE='false', GRADSCF_NATIVE_THREADS='64')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[key] = '1'
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import jax
    import jax.numpy as jnp
    import gradscf
    from gradscf.integrals import prepare_basis, make_plan
    from gradscf.integrals.backends.native.jk import NativeDirectBasis
    from gradscf.integrals._native import register_integrals
    jax.config.update('jax_enable_x64', True)
    save = lambda name, value: (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    tick = time.perf_counter()
    topology, parameters = prepare_basis(cfg['atom'], cfg['basis'], unit=cfg['unit'], charge=cfg['charge'], spin=cfg['spin'], cart=cfg['cart'])
    provider = NativeDirectBasis(make_plan(topology), parameters, private_output_bytes=1024**3)
    density = np.load(args.density, allow_pickle=False)
    if density.shape != (topology.nao, topology.nao) or density.dtype != np.float64 or not np.isfinite(density).all():
        raise ValueError('Shared density must be finite float64 with the actual spatial AO dimension')
    dm = jax.block_until_ready(jnp.asarray(density))
    native, package = Path(register_integrals()._name).resolve(), Path(gradscf.__file__).parent
    paths = ('integrals/backends/native/jk.py', 'integrals/plan.py', 'integrals/_native/csrc/direct_jk.cc',
             'integrals/_native/csrc/coulomb_shells.h', 'integrals/_native/csrc/tables.h')
    runtime = dict(job=os.environ['SLURM_JOB_ID'], host=platform.node(), python=sys.executable, jax=jax.__version__, numpy=np.__version__,
        module=gradscf.__file__, backend=jax.default_backend(), dtype=str(dm.dtype), affinity=sorted(os.sched_getaffinity(0)),
        requested_workers=64, selected_workers=None, actual_busy_workers=None, private_output_bytes=provider.private_output_bytes,
        omp_blas_threads=1, native_threads_environment=64, with_j=True, with_k=True, omega=0., cutoff=cfg['direct_scf_tol'],
        nao=topology.nao, batch_shape=[], density_shape=list(dm.shape), config_sha256=sha(args.config), density_sha256=sha(args.density),
        probe_sha256=sha(__file__), native=str(native), native_sha256=sha(native),
        source_sha256={name: sha(package/name) if (package/name).exists() else None for name in paths},
        preparation_seconds=time.perf_counter()-tick, ru_maxrss_bytes=rss(),
        scope='shared PySCF PBE reference D and scaled trial D; direct on-demand 4c J/K; no SCF/grid/full ERI tensor/RI',
        interpretation='Amplitude timing is separate from source-proven density-independent Schwarz screening; not a full-SCF speed ratio',
        units='seconds, bytes, atomic units; compiler memory excludes opaque native heap and is not RSS; RSS is cumulative process high-water')
    save('runtime.json', runtime)
    action = jax.jit(lambda basis, d: basis.get_jk(d, workers=64, screening_threshold=cfg['direct_scf_tol'], with_j=True, with_k=True))
    tick = time.perf_counter(); lowered = action.lower(provider, dm); runtime['lower_seconds'] = time.perf_counter()-tick
    tick = time.perf_counter(); executable = lowered.compile(); runtime['compile_seconds'] = time.perf_counter()-tick
    try:
        memory = executable.memory_analysis()
        runtime['compiler_memory_not_rss'] = {key: int(getattr(memory, key)) for key in
            ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes') if getattr(memory, key, None) is not None}
    except (AttributeError, NotImplementedError, RuntimeError): runtime['compiler_memory_not_rss'] = None
    save('runtime.json', runtime)
    records, reference = [], None
    for scale in (1., 1e-4):
        d = jax.block_until_ready(scale * dm)
        samples = []
        for _ in range(4):
            tick = time.perf_counter(); value = jax.block_until_ready(executable(provider, d)); samples.append(time.perf_counter()-tick)
        outputs = tuple(np.asarray(x) for x in value)
        if reference is None: reference = outputs
        errors = []
        for output, original in zip(outputs, reference):
            assert np.isfinite(output).all(), 'Nonfinite direct output'
            np.testing.assert_allclose(output, scale * original, atol=1e-10, rtol=1e-10)
            errors.append(float(np.max(np.abs(output - scale * original))))
        records.append(dict(scale=scale, first_seconds=samples[0], warm_seconds=samples[1:], warm_median_seconds=float(np.median(samples[1:])),
            jk_norms=[float(np.linalg.norm(x)) for x in outputs], linearity_max_abs=errors, atol=1e-10, rtol=1e-10, ru_maxrss_bytes=rss()))
        save('measurements.json', records); print(json.dumps(records[-1]), flush=True)
    save('summary.json', dict(records=records, linearity_passed=True, scientific_pass=False,
        scope='same-input amplitude/action probe only', selected_workers=None, actual_busy_workers=None, ru_maxrss_bytes=rss()))


if __name__ == '__main__': main()
