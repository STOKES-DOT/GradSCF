"""Read-only S22/08 implicit-response diagnosis; never forms a four-index ERI."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from validate_nnao_pilot import atomic_json, file_hash, verify_source_imports

# Frozen GMTKN55 v2 S22/08 geometry, matching the pilot selection exactly.
GEOMETRY = dict(name='S22/08', structure_id='S22/08', charge=0, spin=0,
    symbols=['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
    coords_angstrom=[
        [0., 0., -1.8591615463148785], [0., 0., 1.8591615463148785],
        [.8886213302703314, -.5130457642392212, -1.494593141564464],
        [0., 1.0260915284784475, -1.494593141564464], [0., 0., -2.9482847278708633],
        [-.8886213302703314, -.5130457642392212, -1.494593141564464],
        [0., 0., 2.9482847278708633], [-.8886213302703314, .5130457642392212, 1.494593141564464],
        [.8886213302703314, .5130457642392212, 1.494593141564464],
        [0., -1.0260915284784475, 1.494593141564464]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('response-diagnosis.json'))
    parser.add_argument('--dense', action='store_true', help='Bounded 2704x2704 response matrix, not ERI.')
    args = parser.parse_args()
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ['JAX_ENABLE_X64'] = '1'
    import importlib.util
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.scipy.sparse.linalg import gmres
    from gradscf.df import build_jk_from_df
    from gradscf.scf.core import _build_density_from_occ, _diagonalize_fock, _orthogonalizer
    from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable
    from gradscf.solvers.linear.gmres import gmres_solve
    from gradscf.solvers.diagnostics import linear_residual
    from gradscf.solvers.linear import solve_linear
    from gradscf.solvers.types import LinearSolverConfig
    jax.config.update('jax_enable_x64', True)
    source = ROOT / 'tools' / 'optimize_methane_nnao.py'
    spec = importlib.util.spec_from_file_location('nnao_rhf_diagnosis', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    start = time.perf_counter()
    report = dict(structure_id='S22/08', dtype='float64', jax_version=jax.__version__,
        devices=[str(device) for device in jax.devices()],
        device_kinds=[device.device_kind for device in jax.devices()],
        backend=jax.default_backend(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        imported_sources=verify_source_imports(ROOT),
        cache=None if args.cache is None else str(args.cache),
        cache_sha256=None if args.cache is None else file_hash(args.cache), gmres=[])

    def number(value):
        value = float(value)
        return value if np.isfinite(value) else None

    def persist():
        report['elapsed_seconds'] = time.perf_counter() - start
        atomic_json(args.output, report)

    persist()
    ex = module.MethaneRHF(geometry=GEOMETRY, basis_family='szp663_direct',
        core_primitives=6, jk_backend='df', integral_cache=args.cache)
    out = ex.layout.reference_outputs()
    t, s, h = ex._contracted_one_electron(out, ex.ps, ex.ph)
    factors = ex._project_df_factors(t, ex.rep)
    n = s.shape[0]
    cfg = RKSConfig(xc_spec='hf', jk_backend='df', max_cycle=150,
        conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9)
    result = jax.jit(lambda: run_rks_from_integrals_traceable(
        overlap=s, hcore=h, eri=None, df_factors=factors,
        nelectron=ex.nelectron, nuclear_repulsion=ex.enuc, ao=jnp.zeros((0, n)),
        ao_deriv1=jnp.zeros((4, 0, n)), grid_weights=jnp.zeros(0), config=cfg))()
    d = result.density_matrix
    j, k = build_jk_from_df(factors, d); fock = h + j - .5 * k
    energies, _ = _diagonalize_fock(fock, _orthogonalizer(s, cfg.orthogonalization_eps))
    nocc = ex.nelectron // 2
    report['primal'] = dict(energy_hartree=number(result.total_energy),
        converged=bool(result.converged), scf_cycles=int(result.cycles),
        orbital_residual=number(jnp.linalg.norm(fock @ d @ s - s @ d @ fock)),
        min_overlap_eigenvalue=number(jnp.linalg.eigvalsh(s)[0]),
        homo_lumo_gap_hartree=number(energies[nocc] - energies[nocc - 1]),
        min_occupied_internal_gap_hartree=number(jnp.min(jnp.diff(energies[:nocc]))),
        min_virtual_internal_gap_hartree=number(jnp.min(jnp.diff(energies[nocc:]))),
        nao=int(n), primitive_nao=int(ex.primitive_topology.nao))
    occupation = jnp.zeros(n).at[:nocc].set(2.)
    persist(); print('Primal:', report['primal'], flush=True)

    def phi(density, outputs):
        tl, sl, hl = ex._contracted_one_electron(outputs, ex.ps, ex.ph)
        rl = ex._project_df_factors(tl, ex.rep)
        jl, kl = build_jk_from_df(rl, density)
        _, cl = _diagonalize_fock(hl + jl - .5 * kl, _orthogonalizer(sl, cfg.orthogonalization_eps))
        return _build_density_from_occ(cl, occupation)

    _, pull_density = jax.vjp(lambda density: phi(density, out), d)
    operator = jax.jit(lambda flat: (pull_density(flat.reshape((n, n)))[0]
        - flat.reshape((n, n))).reshape(-1))
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.normal(size=(n, n))); v = jnp.asarray(rng.normal(size=(n, n)))
    jvp = jax.jit(lambda tangent: jax.jvp(lambda density: phi(density, out), (d,), (tangent,))[1])(u)
    vjp = pull_density(v)[0]
    report['density_response'] = dict(jvp_finite=bool(jnp.isfinite(jvp).all()),
        vjp_finite=bool(jnp.isfinite(vjp).all()),
        jvp_norm=number(jnp.linalg.norm(jvp)), vjp_norm=number(jnp.linalg.norm(vjp)),
        transpose_dot_error=number(jnp.abs(jnp.sum(jvp * v) - jnp.sum(u * vjp))),
        fixed_point_residual=number(jnp.linalg.norm(phi(d, out) - d)))
    _, pull_outputs = jax.vjp(lambda outputs: phi(d, outputs), out)
    coefficient_probe = pull_outputs(v)[0]
    report['coefficient_response_probe'] = dict(finite=bool(jnp.isfinite(coefficient_probe).all()),
        norm=number(jnp.linalg.norm(coefficient_probe)))
    rhs = (-fock).reshape(-1)
    report['adjoint'] = dict(shape=[n * n, n * n], rhs_norm=number(jnp.linalg.norm(rhs)),
                            operator='transpose of (dPhi/dD - I)')
    persist(); print('Density response:', report['density_response'], flush=True)

    for restart, maxiter, tolerance in ((40, 100, 1e-9), (80, 200, 1e-9),
                                       (160, 300, 1e-9), (40, 100, 1e-8)):
        tick = time.perf_counter()
        raw = jax.jit(lambda b: gmres_solve(operator, b, rtol=tolerance, atol=0.,
            maxiter=maxiter, restart=restart))(rhs)
        residual, valid = linear_residual(operator, raw, rhs, rtol=tolerance)
        item = dict(method='checked-wrapper raw candidate', restart=restart, maxiter=maxiter,
            rtol=tolerance, atol=0., candidate_finite=bool(jnp.isfinite(raw).all()),
            true_residual=number(residual), rhs_norm=report['adjoint']['rhs_norm'],
            relative_residual=number(residual / jnp.linalg.norm(rhs)), valid=bool(valid),
            elapsed_seconds=time.perf_counter() - tick)
        report['gmres'].append(item)
        if bool(valid):
            response = pull_outputs(raw.reshape((n, n)))[0]
            item['coefficient_response_finite'] = bool(jnp.isfinite(response).all())
            item['coefficient_response_norm'] = number(jnp.linalg.norm(response))
        persist(); print('GMRES:', item, flush=True)

    # Independent upstream algorithms at the original settings, without NaN masking.
    scale = jnp.max(jnp.abs(rhs)); scaled_rhs = rhs / scale
    for method in ('incremental', 'batched'):
        tick = time.perf_counter()
        unit, _ = jax.jit(lambda b: gmres(operator, b, tol=1e-9, atol=0., restart=40,
            maxiter=100, solve_method=method))(scaled_rhs)
        raw = unit * scale
        residual, valid = linear_residual(operator, raw, rhs, rtol=1e-9)
        item = dict(method=method, restart=40, maxiter=100, rtol=1e-9, atol=0.,
            candidate_finite=bool(jnp.isfinite(raw).all()), true_residual=number(residual),
            relative_residual=number(residual / jnp.linalg.norm(rhs)), valid=bool(valid),
            elapsed_seconds=time.perf_counter() - tick)
        report['gmres'].append(item); persist(); print('Raw GMRES:', item, flush=True)

    if args.dense:
        tick = time.perf_counter()
        # Sequential columns bound intermediate memory: matrix is a response Jacobian.
        matrix = jax.jit(lambda: jax.lax.map(operator, jnp.eye(n * n)).T)()
        dense = jnp.linalg.solve(matrix, rhs)
        residual, valid = linear_residual(operator, dense, rhs, rtol=1e-9)
        report['dense_response'] = dict(candidate_finite=bool(jnp.isfinite(dense).all()),
            true_residual=number(residual), relative_residual=number(residual / jnp.linalg.norm(rhs)),
            valid=bool(valid), matrix_bytes=int(matrix.size * matrix.dtype.itemsize),
            elapsed_seconds=time.perf_counter() - tick)
        persist(); print('Dense:', report['dense_response'], flush=True)

    (energy, info), gradient = ex._value_grad(out, ex.ps, ex.ph, ex.rep)
    report['full_implicit'] = dict(energy_hartree=number(energy),
        gradient_finite=bool(jnp.isfinite(gradient).all()),
        nonfinite_gradient_count=int(jnp.sum(~jnp.isfinite(gradient))),
        gradient_norm=number(jnp.linalg.norm(gradient)),
        scf={key: bool(value) if key == 'converged' else int(value) if key == 'scf_cycles'
             else number(value) for key, value in info.items()})
    persist(); print('Full implicit:', report['full_implicit'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
