"""CPU8 Slurm probe: fixed-density Gaussian J/K value and directional AD costs.

Three separated water fragments, 6-31g*, Bohr, float64; no SCF, NN, full ERI
or Jacobian. Print JSONL runtime/measurements/summary; redirect stdout to save.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time


VALUE_ATOL, DERIVATIVE_ATOL = 1e-8, 1e-7
SEED, CUTOFFS, KINDS = 20261109, (0., 1e-12), ('value', 'geometry', 'coefficients', 'log_exponents')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    if (not os.getenv('SLURM_JOB_ID') or not 1 <= args.workers <= 8 or allocated < args.workers
            or len(os.sched_getaffinity(0)) < args.workers):
        parser.error('Run inside Slurm with allocated CPUs; workers must be within the authorized CPU8 cap')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.workers])
    os.environ.update(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1', JAX_ENABLE_COMPILATION_CACHE='false',
                      GRADSCF_NATIVE_THREADS=str(args.workers))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import jax
    import jax.numpy as jnp
    import numpy as np
    import gradscf
    from gradscf import integrals
    from gradscf.integrals._native import register_integrals
    from gradscf.integrals.backends.native.jk import NativeDirectBasis
    jax.config.update('jax_enable_x64', True)
    emit = lambda data: print(json.dumps(data, allow_nan=False), flush=True)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)

    def array_hash(arrays):
        digest = hashlib.sha256()
        for value in jax.tree.leaves(arrays):
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(str((array.shape, array.dtype.str)).encode()); digest.update(array.tobytes())
        return digest.hexdigest()

    start = time.perf_counter()
    water = (('O', (0., 0., 0.)), ('H', (1.4304, 0., 1.1072)), ('H', (-1.4304, 0., 1.1072)))
    atom = [(symbol, (xyz[0]+8.*fragment, xyz[1], xyz[2]))
            for fragment in range(3) for symbol, xyz in water]
    topology, parameters = integrals.prepare_basis(atom, '6-31g*', unit='Bohr', cart=True)
    plan, rng = integrals.make_plan(topology), np.random.default_rng(SEED)
    n, rank = topology.nao, min(25, topology.nao)
    factor = rng.normal(size=(n, rank)); matrix = factor@factor.T
    density = jnp.asarray(matrix/np.linalg.norm(matrix))
    nuclei, centers = np.asarray(parameters.nuclear_coords), np.asarray(parameters.centers)
    distance = np.sum((centers[:, None, :]-nuclei[None, :, :])**2, axis=-1)
    owner = np.argmin(distance, axis=1)
    if np.max(distance[np.arange(len(centers)), owner]) > 1e-20:
        raise ValueError('Geometry probe requires atom-centered Gaussian shells')
    nuclear_direction = rng.normal(size=nuclei.shape); nuclear_direction /= np.linalg.norm(nuclear_direction)
    def direction(values):
        arrays = tuple(rng.normal(size=value.shape) for value in values)
        norm = np.sqrt(sum(np.sum(value**2) for value in arrays))
        return tuple(jnp.asarray(value/norm) for value in arrays)
    dr, dc, da = jnp.asarray(nuclear_direction), direction(parameters.coefficients), direction(parameters.exponents)
    point = jnp.asarray(0., dtype=jnp.float64)
    jax.block_until_ready((parameters, density, dr, dc, da))
    preparation = time.perf_counter()-start
    library, package = Path(register_integrals()._name).resolve(), Path(gradscf.__file__).parent
    sources = ('integrals/plan.py', 'integrals/backends/native/packing.py', 'integrals/backends/native/jk.py',
               'integrals/backends/native/autodiff/direct_basis.py', 'integrals/_native/csrc/eri_basis.cc',
               'integrals/_native/csrc/eri_basis.h', 'integrals/_native/csrc/eri_symmetry.h',
               'integrals/_native/csrc/direct_jk_geometry.cc', 'integrals/_native/csrc/coulomb_shells.h')
    emit(dict(type='runtime', job=os.environ['SLURM_JOB_ID'], host=platform.node(), python=sys.executable,
        module=gradscf.__file__, module_sha256=sha(gradscf.__file__), jax=jax.__version__, numpy=np.__version__,
        native=str(library), native_sha256=sha(library), script_sha256=sha(__file__),
        source_sha256={name: sha(package/name) for name in sources}, backend=jax.default_backend(), dtype='float64',
        atom_bohr=atom, basis='6-31g*', cart=True, atoms=len(nuclei), shells=len(centers), nao=n, rank=rank,
        density_shape=list(density.shape), density_normalization='Frobenius norm 1, PSD random rank min(25,N)',
        seed=SEED, parameters_sha256=array_hash(parameters), density_sha256=array_hash(density),
        directions_sha256=array_hash((dr, dc, da)), geometry_direction='unit nuclear displacement; shells follow owning atom',
        coefficient_direction='unit Euclidean raw coefficients', exponent_direction='unit Euclidean log exponents',
        requested_cpu_cap=8, slurm_cpus_per_task=allocated, workers=args.workers,
        affinity=sorted(os.sched_getaffinity(0)), preparation_seconds=preparation, ru_maxrss_bytes=rss(),
        gates=dict(value_absolute=VALUE_ATOL, directional_derivative_absolute=DERIVATIVE_ATOL),
        scope='Raw basis normalization + direct J/K + scalar contraction; fixed density, no SCF/NN/full ERI/Jacobian',
        scalar='0.5*sum(D*J)-0.25*sum(D*K), in Hartree; not a total SCF energy',
        units='seconds, bytes; t dimensionless with geometry direction in Bohr; RSS cumulative process high-water'))
    records, reference = [], {}
    for kind in KINDS:
        for cutoff in CUTOFFS:
            def loss(t, base, dm):
                current = base
                if kind == 'geometry':
                    current = replace(base, nuclear_coords=base.nuclear_coords+t*dr, centers=base.centers+t*dr[owner])
                elif kind == 'coefficients':
                    current = replace(base, coefficients=tuple(c+t*v for c, v in zip(base.coefficients, dc)))
                elif kind == 'log_exponents':
                    current = replace(base, exponents=tuple(a*jnp.exp(t*v) for a, v in zip(base.exponents, da)))
                j, k = NativeDirectBasis(plan, current).get_jk(dm, workers=args.workers, screening_threshold=cutoff)
                return .5*jnp.sum(dm*j)-.25*jnp.sum(dm*k)
            action = jax.jit(loss if kind == 'value' else jax.value_and_grad(loss, argnums=0))
            tick = time.perf_counter(); lowered = action.lower(point, parameters, density); lower = time.perf_counter()-tick
            tick = time.perf_counter(); executable = lowered.compile(); compile_time = time.perf_counter()-tick
            samples, outputs = [], []
            for _ in range(4):
                tick = time.perf_counter(); result = jax.block_until_ready(executable(point, parameters, density))
                samples.append(time.perf_counter()-tick)
                outputs.append([float(x) for x in jax.tree.leaves(result)])
            values = np.asarray(outputs)
            if not np.isfinite(values).all(): raise ValueError('Nonfinite value/directional derivative')
            if cutoff == 0.: reference[kind] = values[0]
            errors = np.max(np.abs(values-reference[kind]), axis=0)
            passed = errors[0] <= VALUE_ATOL and (kind == 'value' or errors[1] <= DERIVATIVE_ATOL)
            row = dict(type='measurement', kind=kind, cutoff=cutoff, omega=0., lower_seconds=lower,
                compile_seconds=compile_time, first_compiled_seconds=samples[0], warm_seconds=samples[1:],
                warm_median_seconds=statistics.median(samples[1:]), value=float(values[0, 0]),
                directional_derivative=None if kind == 'value' else float(values[0, 1]),
                max_abs_error_vs_zero_cutoff=errors.tolist(), gate_passed=bool(passed), ru_maxrss_bytes=rss())
            records.append(row); emit(row)
    passed = all(row['gate_passed'] for row in records)
    emit(dict(type='summary', complete=True, cases=len(records), consistency_passed=passed,
              scope='Fixed-density Gaussian product costs only; no SCF/NN training claim', ru_maxrss_bytes=rss()))
    if not passed: raise SystemExit('Predeclared zero-cutoff consistency gate failed')


if __name__ == '__main__':
    main()
