"""Bounded public DFT -> GW -> static BSE timings (run only inside Slurm).

The default measures fresh public workflows, first call plus warm repeats.
Optional preparation profiling changes synchronization and is diagnostic only.
All physical calculations remain in src/gradscf; no supplied/fabricated QP poles.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--case-input', type=Path)
    parser.add_argument('--molecule', choices=('h2', 'water', 'oh'), default='h2')
    parser.add_argument('--basis', default='sto-3g')
    parser.add_argument('--ks', choices=('rks', 'uks', 'gks'), default='rks')
    parser.add_argument('--xc', choices=('svwn', 'pbe', 'pbe0', 'wb97xd'), default='pbe')
    parser.add_argument('--method', '--gw-method', dest='method',
                        choices=('g0w0', 'evgw', 'evgw0', 'qsgw', 'scgw'), default='g0w0')
    parser.add_argument('--scf-backend', choices=('full', 'direct', 'df'), default='full')
    parser.add_argument('--gw-backend', choices=('df', 'scf', 'conventional'), default='df')
    parser.add_argument('--auxbasis', default='weigend')
    parser.add_argument('--grid', type=int, default=3)
    parser.add_argument('--grid-block-size', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--max-memory', type=float, default=60000.)
    parser.add_argument('--nw', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=1, help='Warm fresh-workflow repeats after the first call.')
    parser.add_argument('--nroots', type=int, default=1)
    parser.add_argument('--tda', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--solver', choices=('davidson', 'dense'), default='davidson')
    parser.add_argument('--max-space', type=int, default=40)
    parser.add_argument('--max-dense', type=int, default=256)
    parser.add_argument('--max-aux', type=int, default=1024)
    parser.add_argument('--max-factor-elements', type=int, default=20_000_000)
    parser.add_argument('--gw-max-cycle', type=int, default=100)
    parser.add_argument('--gw-conv-tol', type=float, default=1e-7)
    parser.add_argument('--gw-damp', type=float, default=.3)
    parser.add_argument('--bse-conv-tol', type=float, default=1e-9)
    parser.add_argument('--profile-preparation', action='store_true')
    parser.add_argument('--preparation-repeats', type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        parser.error('Numerical execution requires a Slurm job step; use srun.')
    if args.output.exists() or args.output.with_suffix('.manifest.json').exists():
        parser.error('Use a new output path; previous raw records are immutable.')
    if args.repeats < 0 or args.preparation_repeats < 0 or args.threads < 1:
        parser.error('Repeats must be nonnegative and threads positive.')
    if args.preparation_repeats and not args.profile_preparation:
        parser.error('--preparation-repeats requires --profile-preparation.')
    allowed = sorted(os.sched_getaffinity(0))
    if args.threads > len(allowed):
        parser.error('Requested CPUs exceed the Slurm affinity.')
    os.sched_setaffinity(0, allowed[:args.threads])
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('JAX_ENABLE_COMPILATION_CACHE', 'false')
    import jax
    jax.config.update('jax_enable_x64', True)
    import numpy as np
    import gradscf
    from gradscf import bse, dft, gto, gw
    import gradscf.bse.api as bse_api
    from gradscf.integrals._native import register_integrals
    cpu = jax.devices('cpu')[0]
    jax.config.update('jax_default_device', cpu)
    atoms = dict(h2='H 0 0 0; H 0 0 .74',
                 water='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                 oh='O 0 0 0; H 0 0 .9697')
    case = json.loads(args.case_input.read_text()) if args.case_input else {}
    physical = dict(atom=case.get('atom', atoms[args.molecule]),
                    basis=case.get('basis', args.basis), unit=case.get('unit', 'Angstrom'),
                    charge=int(case.get('charge', 0)),
                    spin=int(case.get('spin', 1 if args.molecule == 'oh' else 0)),
                    cart=case.get('cart', True))
    nw = int(case.get('nw', args.nw))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    common = dict(ks=args.ks, xc=args.xc, method=args.method, dtype='float64',
                  basis=physical['basis'], grid_level=args.grid, nw=nw,
                  case_id=case.get('case_id', args.molecule),
                  scf_backend=args.scf_backend, gw_backend=args.gw_backend,
                  tda=args.tda, solver=args.solver, nroots=args.nroots,
                  units='Ha, Angstrom or case unit, s, bytes', device=str(cpu),
                  affinity=sorted(os.sched_getaffinity(0)),
                  timing_scope='diagnostic_synchronized_preparation' if args.profile_preparation
                  else 'public_workflow_fresh_objects',
                  eligible_for_speed_ratio=not args.profile_preparation,
                  gw_integral_representation=('independent_auxiliary_ri' if args.gw_backend == 'df'
                    else 'inherited_auxiliary_ri' if args.scf_backend == 'df'
                    else 'eri_pair_spectral_factors_tol_1e-12' if args.gw_backend == 'scf'
                    else 'conventional_four_center'))

    def safe(value):
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return safe(value.tolist())
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def record(**row):
        with args.output.open('a') as stream:
            stream.write(json.dumps(safe(dict(common, **row)), allow_nan=False) + '\n')

    library = Path(register_integrals()._name).resolve()
    sources = [Path(gradscf.__file__).parent / name for name in
               ('gw/rgw.py', 'gw/ugw.py', 'gw/g0w0.py', 'gw/evgw.py', 'gw/qsgw.py',
                'gw/scgw.py', 'bse/api.py', 'bse/reference.py', 'bse/response.py',
                'bse/kernel.py', 'scf/facade.py')]
    source_commit = os.environ.get('GRADSCF_SOURCE_COMMIT')
    commit_source = 'GRADSCF_SOURCE_COMMIT' if source_commit else 'git'
    if source_commit is None:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], text=True, capture_output=True)
        source_commit = revision.stdout.strip() if revision.returncode == 0 else None
        if source_commit is None:
            commit_source = 'unavailable; source hashes recorded'
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    physical=physical, case=case, source=gradscf.__file__,
                    python=os.sys.executable,
                    git_head=source_commit, commit_source=commit_source,
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    native_path=str(library), native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    versions={name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'numpy', 'scipy')},
                    devices=[str(d) for d in jax.devices()],
                    slurm={key: os.getenv(key) for key in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURMD_NODENAME')},
                    environment={key: os.getenv(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'MKL_NUM_THREADS', 'GRADSCF_NATIVE_THREADS', 'JAX_ENABLE_COMPILATION_CACHE',
                        'JAX_COMPILATION_CACHE_DIR')})
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Unsupported chains are explicit capability records, not substitute calculations.
    reason = None
    if args.ks == 'gks':
        reason = 'No spinor GW facade or complex/noncollinear molecular BSE adapter.'
    elif args.ks == 'uks' and args.method in ('qsgw', 'scgw'):
        reason = 'UGW.kernel accepts g0w0, evgw and evgw0 only.'
    elif args.method == 'scgw':
        reason = 'GW.get_bse_inputs rejects Matsubara scGW: validated analytic continuation and QP coverage are absent; static Fock eigenvalues are not QP poles.'
    elif args.ks == 'uks' and args.gw_backend == 'df' and args.method != 'g0w0':
        reason = 'UGW.density_fit independent auxiliary fitting requires g0w0; use an explicit SCF factor source.'
    elif args.gw_backend == 'conventional':
        reason = 'Static BSE export of conventional four-center GW is not implemented.'
    elif args.ks == 'rks' and args.gw_backend == 'scf' and args.scf_backend == 'direct':
        reason = 'Direct RKS supplies no cached Coulomb factors for factor-based GW/BSE; select explicit GW RI.'
    if reason:
        record(stage='capability', status='source_verified_unsupported', reason=reason, scientific_pass=False,
               evidence='source audit only; no numerical execution of this combination')
        return 2

    for repeat in range(args.repeats + 1):
        stage = 'scf'
        workflow_start = time.perf_counter()
        profile = []
        try:
            record(stage='workflow_start', repeat=repeat, cache_state='first' if repeat == 0 else 'warm_fresh_objects')
            mol = gto.M(**physical)
            cls = dft.RKS if args.ks == 'rks' else dft.UKS
            mf = cls(mol, xc=args.xc, grids_level=args.grid, grid_block_size=args.grid_block_size,
                     conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                     max_cycle=200, max_memory=args.max_memory, execution_device='cpu')
            if args.scf_backend == 'direct':
                mf.direct_scf_workers, mf.direct_scf_tol = args.threads, 0.
                mf.direct_scf()
            elif args.scf_backend == 'df':
                mf.density_fit(args.auxbasis)
            start = time.perf_counter()
            mf.run()
            jax.block_until_ready(mf.scf_result if mf.scf_result is not None else mf.reference)
            scf_seconds = time.perf_counter() - start
            record(stage=stage, repeat=repeat, wall_seconds=scf_seconds,
                   energy_hartree=float(mf.e_tot), nao=int(mf.mo_coeff.shape[-2]),
                   cycles=None if mf.cycles is None else int(mf.cycles), scientific_pass=bool(mf.converged))
            if not mf.converged:
                raise ArithmeticError('SCF did not converge; no downstream GW/BSE result.')
            stage = 'gw'
            options = dict(method=args.method, nw=nw, max_cycle=args.gw_max_cycle,
                           conv_tol=args.gw_conv_tol, damp=args.gw_damp, max_memory=args.max_memory)
            if args.method == 'scgw':
                options.update(beta=20., mixing=.5)
            qp = (gw.GW if args.ks == 'rks' else gw.UGW)(mf, **options)
            if args.gw_backend == 'df':
                qp.density_fit(args.auxbasis)
            with ExitStack() as stack:
                if args.profile_preparation:
                    def timed(original, name):
                        def wrapped(*positional, **keywords):
                            tick = time.perf_counter()
                            out = original(*positional, **keywords)
                            if isinstance(out, bse.BSEReference):
                                jax.block_until_ready((out.qp_energy, out.screening_energy, out.mo_factors, out.dipole_mo))
                            else:
                                jax.block_until_ready(out)
                            profile.append(dict(stage=name, wall_seconds=time.perf_counter() - tick))
                            return out
                        return wrapped
                    stack.enter_context(patch.object(qp, '_df_factors', timed(qp._df_factors, 'gw_factor_preparation')))
                    stack.enter_context(patch.object(bse_api, 'reference_from_source',
                                                   timed(bse_api.reference_from_source, 'bse_reference_preparation')))
                start = time.perf_counter()
                qp.run(orbs=None if case.get('orbs') is None else tuple(case['orbs']))
                jax.block_until_ready(qp.result)
                gw_seconds = time.perf_counter() - start
                if args.method == 'scgw':
                    record(stage=stage, repeat=repeat, wall_seconds=gw_seconds,
                           scientific_pass=bool(qp.converged), qp_poles_available=False,
                           particle_number_error=float(qp.result.particle_number_error),
                           fixed_point_residual=float(qp.result.fixed_point_residual),
                           total_energy_hartree=float(qp.result.total_energy))
                else:
                    result = qp.result
                    mask = np.asarray(result.qp_computed_mask)
                    selected = dict(qp_targets=np.argwhere(mask), qp_energy_hartree=np.asarray(result.mo_energy)[mask],
                                    screening_energy_hartree=np.asarray(result.screening_energy),
                                    qp_computed_mask=mask, qp_converged_mask=np.asarray(result.converged_mask))
                    for name in ('qp_residual', 'qp_weight', 'qp_branch_valid'):
                        value = getattr(result, name, None)
                        selected[name] = None if value is None else np.asarray(value)[mask]
                    gw_pass = bool(qp.converged) and bool(np.asarray(result.converged_mask)[mask].all())
                    if result.qp_branch_valid is not None:
                        gw_pass &= bool(np.asarray(result.qp_branch_valid)[mask].all())
                    record(stage=stage, repeat=repeat, wall_seconds=gw_seconds, scientific_pass=gw_pass, **selected)
                if not qp.converged or (args.method != 'scgw' and not gw_pass):
                    raise ArithmeticError('GW convergence or selected QP branch checks failed; no BSE result.')
                stage = 'bse'
                windows = {key: case[key] for key in ('occupied', 'virtual', 'screening_occupied', 'screening_virtual') if key in case}
                response = bse.BSE(qp, nroots=args.nroots, tda=args.tda, solver=args.solver,
                                   conv_tol=args.bse_conv_tol, max_space=args.max_space,
                                   max_dense=args.max_dense, max_aux=args.max_aux,
                                   max_factor_elements=args.max_factor_elements, **windows)
                start = time.perf_counter()
                response.run()
                jax.block_until_ready(response.result)
                bse_seconds = time.perf_counter() - start
            result = response.result
            residuals = np.asarray(result.residual_norms)
            energies = np.asarray(response.e)
            residual_pass = bool(np.all(np.isfinite(residuals))
                                 and np.all(residuals <= args.bse_conv_tol))
            bse_pass = bool(result.screening_valid) and residual_pass and bool(
                np.all(np.asarray(result.converged) & np.asarray(result.stable))
                and np.all(np.isfinite(energies)))
            record(stage=stage, repeat=repeat, wall_seconds=bse_seconds, scientific_pass=bse_pass,
                   excitation_energy_hartree=energies, residual_hartree=residuals,
                   residual_gate_pass=residual_pass, residual_tolerance_hartree=args.bse_conv_tol,
                   converged=np.asarray(result.converged), stable=np.asarray(result.stable),
                   response_valid=np.asarray(result.response_valid), screening_valid=bool(result.screening_valid),
                   stability_certified=bool(result.stability_certified),
                   stability_margins=None if result.stability_margins is None else np.asarray(result.stability_margins),
                   mo_factor_shape=list(response.reference.mo_factors.shape))
            start = time.perf_counter()
            strengths = jax.block_until_ready(response.oscillator_strength())
            optical_pass = bool(np.isfinite(strengths).all())
            record(stage='optical_properties', repeat=repeat, wall_seconds=time.perf_counter() - start,
                   oscillator_strength=np.asarray(strengths), scientific_pass=optical_pass)
            workflow_seconds = time.perf_counter() - workflow_start
            record(stage='workflow', repeat=repeat, wall_seconds=workflow_seconds,
                   scf_seconds=scf_seconds, gw_seconds=gw_seconds, bse_seconds=bse_seconds,
                   scientific_pass=bse_pass and optical_pass, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            # Resolve metadata after the formal end timer, using the production
            # space constructor; None denotes a complete channel, not no W.
            reference = response.reference
            screening_space = bse.make_bse_space(
                reference.qp_energy.shape[-1], reference.nocc,
                occupied=reference.screening_occupied if response.screening_occupied is None
                    else response.screening_occupied,
                virtual=reference.screening_virtual if response.screening_virtual is None
                    else response.screening_virtual)
            record(stage='effective_windows', repeat=repeat,
                   optical_window=dict(occupied=response.space.occupied, virtual=response.space.virtual),
                   screening_window=dict(occupied=screening_space.occupied, virtual=screening_space.virtual))
            for row in profile:
                record(repeat=repeat, **row)
            for probe in range(args.preparation_repeats):
                start = time.perf_counter()
                inputs = qp.get_bse_inputs(max_aux=args.max_aux, max_factor_elements=args.max_factor_elements)
                jax.block_until_ready(inputs)
                record(stage='duplicate_bse_preparation_diagnostic', repeat=repeat, probe=probe,
                       wall_seconds=time.perf_counter() - start, mo_factor_shape=list(inputs['mo_factors'].shape))
                del inputs
            # Fresh workflow objects must not overlap the next SCF's resident
            # arrays. Keep JAX compilation caches intact for the warm call.
            del response, qp, mf, reference, result, strengths
            if not (bse_pass and optical_pass):
                return 1
        except Exception as exc:
            for row in profile:
                record(repeat=repeat, **row)
            record(stage=stage, repeat=repeat, status='runtime_verified_unsupported' if isinstance(exc, NotImplementedError) else 'error',
                   error_type=type(exc).__name__, error=str(exc), scientific_pass=False,
                   elapsed_workflow_seconds=time.perf_counter() - workflow_start)
            return 2 if isinstance(exc, NotImplementedError) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
