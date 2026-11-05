"""Shared-input PBE tile algebra, not SCF or complete geometry AD.
Run inside a CPU64 Slurm step with the process/RSS guard:
  python probe_xc_algebra.py frozen-grad-config.json density.npy new-output
Density is the supplied PySCF reference; no SCF/ERI/RI or atomic guess is built.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
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
    if not cfg['cart'] or (cfg['engine'], cfg['method'], cfg['xc'], cfg['grid_level'], cfg['threads']) != ('gradscf', 'RKS', 'pbe', 3, 64):
        parser.error('Requires the frozen GradSCF RKS/PBE/grid3/CPU64 input')
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
    from gradscf import gto
    from gradscf.integrals.basis import basis_from_molecule_spec
    from gradscf.integrals.grids import build_molecular_grid_from_spec
    from gradscf.integrals.grids.ao import evaluate_cartesian_ao_with_derivatives
    from gradscf.scf import rks
    from gradscf.tools.features import _spin_density_and_gradient
    jax.config.update('jax_enable_x64', True)
    rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    package = Path(gradscf.__file__).parent
    sources = ('scf/rks.py', 'tools/features.py', 'dft/xc/evaluate.py', 'dft/xc/_jax_xc.py',
               'dft/xc/parser.py', 'dft/xc/registry.py', 'integrals/grids/ao.py', 'integrals/grids/__init__.py')
    runtime = dict(job=os.environ['SLURM_JOB_ID'], host=platform.node(), python=sys.executable, jax=jax.__version__, numpy=np.__version__,
        module=gradscf.__file__, backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64),
        dtype='float64', affinity=sorted(os.sched_getaffinity(0)), native_threads=64, omp_blas_threads=1,
        config_sha256=sha(args.config), density_sha256=sha(args.density), probe_sha256=sha(__file__),
        runtime_source_sha256={name: sha(package/name) for name in sources},
        scope='shared PySCF density, one real grid tile; algebra kernels only; no SCF/ERI/RI/full geometry AD',
        units='seconds, bytes, atomic units; compiler memory is not RSS; RSS is cumulative process high-water')
    save('runtime.json', runtime)
    precision = jax.lax.Precision.HIGHEST
    def shared(a, da, d):
        c = jnp.matmul(a, d.T, precision=precision)
        return jnp.sum(a*c, axis=1), 2*jnp.einsum('xrp,rp->rx', da[1:4], c, precision=precision)
    def current_v(a, da, w, vr, vg):
        zero = jnp.zeros_like(w)
        return rks._vxc_matrix_from_grid_potential(ao=a, ao_deriv1=da, ao_laplacian=None, weights=w,
            vxc_rho=vr, vxc_grad=vg, vxc_tau=zero, vxc_lapl=zero, xc_kind='GGA')
    def fused_v(a, da, w, vr, vg):
        aow = .5*(w*vr)[:, None]*a + jnp.einsum('rx,xrp->rp', w[:, None]*vg, da[1:4], precision=precision)
        m = jnp.matmul(aow.T, a, precision=precision)
        return m + m.T
    def compare(got, expected, tol):
        maximum = 0.
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
            a, b = np.asarray(a), np.asarray(b)
            assert np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite algebra output'
            maximum = max(maximum, float(np.max(np.abs(a-b))))
            np.testing.assert_allclose(a, b, atol=tol, rtol=tol)
        return dict(max_abs=maximum, atol=tol, rtol=tol, passed=True)
    rng = np.random.default_rng(20261105)
    tiny = tuple(jnp.asarray(rng.normal(size=shape)*.1) for shape in ((7, 4), (4, 7, 4), (4, 4), (7,), (7,), (7, 3)))
    checks = {}
    for name, old, new, inputs in (('density', _spin_density_and_gradient, shared, tiny[:3]),
                                  ('potential', current_v, fused_v, (tiny[0], tiny[1], *tiny[3:]))):
        directions = tuple(jnp.asarray(rng.normal(size=x.shape)*.03) for x in inputs)
        value, tangent = jax.jvp(old, inputs, directions)
        other, other_tangent = jax.jvp(new, inputs, directions)
        weights = tuple(jnp.asarray(rng.normal(size=x.shape)) for x in jax.tree.leaves(value))
        def loss(fn, xs): return sum(jnp.vdot(v, w).real for v, w in zip(jax.tree.leaves(fn(*xs)), weights))
        grads = [jax.grad(lambda *xs: loss(fn, xs), argnums=tuple(range(len(inputs))))(*inputs) for fn in (old, new)]
        checks[name] = dict(value=compare(value, other, 1e-10), jvp=compare(tangent, other_tangent, 1e-9),
                            vjp=compare(grads[0], grads[1], 1e-9), inputs='independent nonsymmetric arrays, including D or grid weights')
    save('tiny_array_checks.json', checks)
    started = time.perf_counter()
    spec = gto.M(atom=cfg['atom'], basis=cfg['basis'], unit=cfg['unit'], charge=cfg['charge'], spin=cfg['spin'], cart=cfg['cart']).to_spec()
    basis = basis_from_molecule_spec(spec, basis=cfg['basis'])
    coords, weights = build_molecular_grid_from_spec(spec, level=3)
    start, stop = len(coords)//2, min(len(coords)//2 + 4096, len(coords))
    ao, da = evaluate_cartesian_ao_with_derivatives(basis, coords[start:stop], deriv=1)
    density = np.load(args.density, allow_pickle=False)
    if density.shape != (basis.nao, basis.nao) or density.dtype != np.float64 or not np.isfinite(density).all():
        raise ValueError('Reference density must be finite float64 with the actual AO dimension')
    d, w = jnp.asarray(density), weights[start:stop]
    rho, grad = _spin_density_and_gradient(ao, da, d)
    _, vr, vg = rks._xc_energy_and_potential_on_grid(rho=rho, grad=grad, tau=jnp.zeros_like(rho), weights=w,
        xc_spec='pbe', density_floor=1e-12, potential_clip=None, xc_kind='GGA')
    jax.block_until_ready((ao, da, d, w, vr, vg))
    runtime.update(nao=basis.nao, total_ngrid=len(coords), tile_start=start, tile_stop=stop, tile_points=stop-start,
        array_shapes=dict(ao=list(ao.shape), deriv1=list(da.shape), density=list(d.shape), weights=list(w.shape)),
        preparation_seconds=time.perf_counter()-started, ru_maxrss_bytes=rss(), scientific_pass=False)
    save('runtime.json', runtime)
    def dots(hlo):
        found = []
        for line in hlo.splitlines():
            match = re.search(r'=\s*(?:f64|c128)\[([0-9,]+)\].*\bdot\(', line)
            if match and sum(int(n) >= 64 for n in match.group(1).split(',')) >= 2: found.append(line.strip())
        return dict(count=len(found), lines=found, definition='syntactic HLO dots with at least two output axes >=64; not runtime invocations')
    records = []
    def measure(name, function, inputs):
        tick = time.perf_counter(); lowered = jax.jit(function).lower(*inputs); lower = time.perf_counter()-tick
        tick = time.perf_counter(); executable = lowered.compile(); compile_time = time.perf_counter()-tick
        row = dict(name=name, lower_seconds=lower, compile_seconds=compile_time,
                   lowered_large_dots=dots(lowered.compiler_ir('hlo').as_hlo_text()))
        try: row['optimized_large_dots'] = dots(executable.as_text())
        except (AttributeError, NotImplementedError): row['optimized_large_dots'] = None
        try:
            memory = executable.memory_analysis()
            row['compiler_memory_not_rss'] = {key: int(getattr(memory, key)) for key in
                ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes') if getattr(memory, key, None) is not None}
        except (AttributeError, NotImplementedError): row['compiler_memory_not_rss'] = None
        tick = time.perf_counter(); value = jax.block_until_ready(executable(*inputs)); row['cold_after_compile_seconds'] = time.perf_counter()-tick
        samples = []
        for _ in range(10):
            tick = time.perf_counter(); value = jax.block_until_ready(executable(*inputs)); samples.append(time.perf_counter()-tick)
        row.update(warm_seconds=samples, warm_median_seconds=float(np.median(samples)), ru_maxrss_bytes=rss())
        records.append(row); save('measurements.json', records); print(json.dumps(row), flush=True)
        return value
    density_old = measure('builtin_density', _spin_density_and_gradient, (ao, da, d))
    density_new = measure('shared_C_density', shared, (ao, da, d))
    potential_old = measure('current_vxc', current_v, (ao, da, w, vr, vg))
    potential_new = measure('weighted_AO_vxc', fused_v, (ao, da, w, vr, vg))
    save('summary.json', dict(tiny_checks=checks, density=compare(density_old, density_new, 1e-10),
        potential=compare(potential_old, potential_new, 1e-10), algebra_checks_passed=True, scientific_pass=False,
        scope='one shared-input tile; no full-SCF timing attribution', ru_maxrss_bytes=rss()))


if __name__ == '__main__': main()
