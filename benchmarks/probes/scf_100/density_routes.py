"""Density-route kernel diagnostics; never an end-to-end SCF speed certificate.

Run CPU and GPU in separate fresh processes in the authorized CPU64/single-GPU
Slurm allocation. Synthetic (25, 260, 880)-AO cases are labelled explicitly;
they are NOT the saved 96-atom PBE0 calculation. A small GradSCF water/PBE0
calculation supplies one physical AO block and its actual converged density.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.probes.scf_solver.profile_stages import plain, peak_rss_bytes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'gpu'), required=True)
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        parser.error('Run inside the authorized Slurm allocation')
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 64:
        parser.error('A CPU64 allocation is required')
    os.sched_setaffinity(0, affinity[:64])
    os.environ.update(JAX_PLATFORMS='cpu' if args.device == 'cpu' else 'cuda,cpu',
        JAX_ENABLE_X64='1', JAX_ENABLE_COMPILATION_CACHE='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', GRADSCF_NATIVE_THREADS='64')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output/name).write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')

    def emit(row):
        with (args.output/'measurements.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(row), allow_nan=False)+'\n')
        print(json.dumps(plain(row), allow_nan=False), flush=True)

    import numpy as np
    import jax
    import jax.numpy as jnp
    import gradscf
    from gradscf.tools import features
    from gradscf.scf import rks
    from gradscf.scf.inputs.assembly import build_rks_integral_inputs
    from gradscf.integrals.grids.blocks import AOGrid
    from gradscf.integrals._native import register_integrals
    from gradscf.solvers.scf import orbital_gradient_norm

    jax.config.update('jax_enable_x64', True)
    sync = jax.block_until_ready
    cpu, target = jax.devices('cpu')[0], jax.devices(args.device)[0]
    if args.device == 'gpu' and len(jax.devices('gpu')) != 1:
        raise RuntimeError('Expose only the allocated GPU')
    rng = np.random.default_rng(20261109)
    native = Path(register_integrals()._name).resolve()
    cpu_model = next((line.split(':', 1)[1].strip() for line in
        Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), platform.processor())
    save('manifest.json', dict(job=os.environ['SLURM_JOB_ID'], host=platform.node(), cpu_model=cpu_model,
        command=sys.argv, python=sys.executable, jax=jax.__version__, numpy=np.__version__,
        device=str(target), kind=target.device_kind, affinity=sorted(os.sched_getaffinity(0)),
        memory_allocation_mb=os.getenv('SLURM_MEM_PER_NODE'), native_workers=64, blas_threads=1,
        dtype='float64', seed=20261109, block_size=4096, warm_repetitions=10,
        script_sha256=digest(__file__), features_sha256=digest(features.__file__),
        module=gradscf.__file__, native_sha256=digest(native), value_atol=1e-10,
        rks_sha256=digest(rks.__file__), grid_blocks_sha256=digest(sys.modules[AOGrid.__module__].__file__),
        field_units='Real: rho electron/bohr^3, grad electron/bohr^4, tau electron/bohr^5; synthetic: unitless',
        scope='Kernel diagnostic: complete density helper including exact guard; excludes XC, Vxc and SCF.'))

    def rebuild(c, o):
        return jnp.einsum('pi,i,qi->pq', c, o, c, precision=jax.lax.Precision.HIGHEST)

    def error(left, right):
        differences = [np.asarray(a)-np.asarray(b) for a, b in zip(
            jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))]
        if not all(np.isfinite(value).all() for value in differences):
            raise RuntimeError('Nonfinite numerical comparison')
        result = max(float(np.max(np.abs(value), initial=0.)) for value in differences)
        if result > 1e-10:
            raise RuntimeError(f'Density-route numerical gate failed: {result}')
        return result

    def functions(kind, nocc):
        def baseline(a, da, d, c, o):
            rho, grad = features._spin_density_and_gradient(a, da, d)
            tau = features._spin_tau(da, d) if kind == 'MGGA' else jnp.zeros_like(rho)
            return rho, grad, tau
        def candidate(a, da, d, c, o):
            return features._spin_density_fields(a, da, d, xc_kind=kind,
                mo_coeff=c, mo_occ=o, occupied_columns=nocc)
        return baseline, candidate

    summaries = []
    def measure(label, provenance, values, nocc):
        a, da, d, c, o = sync(values)
        nao = a.shape[1]
        # This is the helper's exact full-shape consistency expression, compiled
        # with D/C/occ as dynamic operands. It is not an allclose substitute.
        guard = jax.jit(lambda dm, coeff, occ: (
            jnp.all(dm == rebuild(coeff, occ)) & jnp.all(occ[nocc:] == 0)
            & jnp.all(jnp.isfinite(coeff)) & jnp.all(jnp.isfinite(occ))))
        guard_matches = bool(sync(guard(d, c, o)))
        if provenance == 'kernel diagnostic synthetic' and not guard_matches:
            raise RuntimeError('Synthetic input did not enter the exact-match orbital branch')
        for kind in ('GGA', 'MGGA'):
            baseline, candidate = functions(kind, nocc)
            compiled, first_values, timings = [], [], []
            for name, function in (('dense', baseline), ('candidate_with_guard', candidate)):
                start = time.perf_counter()
                lowered = jax.jit(function).lower(*values)
                lower_seconds = time.perf_counter()-start
                start = time.perf_counter()
                executable = lowered.compile()
                compile_seconds = time.perf_counter()-start
                start = time.perf_counter()
                result = sync(executable(*values))
                execute_seconds = time.perf_counter()-start
                memory = executable.memory_analysis()
                timings.append(dict(route=name, lower_seconds=lower_seconds,
                    compile_seconds=compile_seconds, first_execute_seconds=execute_seconds,
                    first_total_seconds=lower_seconds+compile_seconds+execute_seconds,
                    warm_seconds=[], xla_memory=None if memory is None else {
                        key: getattr(memory, key, None) for key in
                        ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes')}))
                compiled.append(executable)
                first_values.append(result)
            maximum_error = error(*first_values)
            for repetition in range(10):
                # Alternate order to reduce fixed-order thermal/clock bias.
                for index in (0, 1) if repetition % 2 == 0 else (1, 0):
                    start = time.perf_counter()
                    sync(compiled[index](*values))
                    timings[index]['warm_seconds'].append(time.perf_counter()-start)
            for timing in timings:
                timing['warm_median_seconds'] = statistics.median(timing['warm_seconds'])
            eligible = (4 if kind == 'GGA' else 1)*nocc < nao
            row = dict(label=label, provenance=provenance, kind=kind, nao=nao,
                ngrids=a.shape[0], occupied_columns=nocc, exact_guard_matches=guard_matches,
                cost_eligible=eligible, orbital_branch_expected=eligible and guard_matches,
                maxabs_error=maximum_error, timings=timings,
                warm_speedup=timings[0]['warm_median_seconds']/timings[1]['warm_median_seconds'])
            emit(row)
            summaries.append(row)

    with jax.default_device(target):
        for nao, nocc in ((25, 5), (260, 47), (880, 148)):
            a = jnp.asarray(rng.normal(size=(4096, nao))/np.sqrt(nao))
            da = jnp.asarray(rng.normal(size=(4, 4096, nao))/np.sqrt(nao)).at[0].set(a)
            c = jnp.asarray(rng.normal(size=(nao, nao))/np.sqrt(nao))
            o = jnp.zeros(nao).at[:nocc].set(2.)
            d = sync(rebuild(c, o))
            measure(f'synthetic_{nao}', 'kernel diagnostic synthetic', (a, da, d, c, o), nocc)
            if nao == 25:
                small = (a[:7], da[:, :7], d, c, o)
                tangent = jnp.asarray(rng.normal(scale=.03, size=d.shape))
                for kind in ('GGA', 'MGGA'):
                    baseline, candidate = functions(kind, nocc)
                    dense_d = lambda density: baseline(small[0], small[1], density, c, o)
                    hinted_d = jax.jit(lambda density: candidate(small[0], small[1], density, c, o))
                    jvp_error = error(jax.jvp(dense_d, (d,), (tangent,)),
                                      jax.jvp(hinted_d, (d,), (tangent,)))
                    values, pullback = jax.vjp(hinted_d, d)
                    _, dense_pullback = jax.vjp(dense_d, d)
                    cotangent = tuple(jnp.asarray(rng.normal(size=value.shape)) for value in values)
                    emit(dict(stage='independent_density_response', kind=kind, nao=nao,
                        maxabs_jvp_error=jvp_error, maxabs_vjp_error=error(
                            pullback(cotangent), dense_pullback(cotangent))))

    # Real molecular input comes solely from GradSCF. Preparation is separated
    # from the contraction timings, and its elapsed time is not a SCF benchmark.
    fixture = dict(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587', basis='def2-svp',
                   unit='Angstrom', cart=True, charge=0, spin=0, grids_level=1)
    cfg = rks.RKSConfig(xc_spec='pbe0', max_cycle=80, conv_tol=1e-10,
        conv_tol_density=1e-8, conv_tol_grad=1e-7, jk_backend='full', max_memory=8000)
    start = time.perf_counter()
    with jax.default_device(cpu):
        inputs = build_rks_integral_inputs(**fixture, config=cfg, xc_spec='pbe0',
                                           init_guess='hcore', include_dipole_integrals=False)
        result = sync(rks.run_rks_from_integrals(**inputs.as_rks_kwargs(), config=cfg))
        residual = float(orbital_gradient_norm(result.fock_matrix, result.mo_coeff, result.mo_occ))
        if not bool(result.converged) or not np.isfinite(float(result.total_energy)) or residual > 1e-7:
            raise RuntimeError('Real GradSCF fixture did not meet convergence criteria')
        provider = AOGrid.from_basis(inputs.basis, inputs.coords, inputs.grid_weights,
                                     block_size=4096, adaptive=False)
        block = sync(provider.block(0, 4096, deriv=1))
        fixture.update(config=asdict(cfg), energy_hartree=float(result.total_energy),
            residual_hartree=residual, cycles=int(result.cycles), ngrids=int(inputs.grid_weights.size),
            nao=int(inputs.overlap.shape[0]), preparation_seconds=time.perf_counter()-start)
        save('fixture.json', fixture)
    values = sync(jax.device_put((block[0], block[1], result.density_matrix,
                                 result.mo_coeff, result.mo_occ), target))
    with jax.default_device(target):
        measure('water_def2_svp_pbe0', 'real GradSCF converged RKS, first AO grid block',
                values, inputs.nelectron//2)
    save('summary.json', dict(kernel_checks_passed=True, peak_rss_bytes=peak_rss_bytes(),
        measurements=summaries, scope='No full-XC, SCF speedup or 96-atom density claim.'))


if __name__ == '__main__':
    main()
