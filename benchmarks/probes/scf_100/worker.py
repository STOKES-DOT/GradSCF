"""One independent, guarded CPU HF/KS workflow from a frozen SCF-100 config.

Run inside Slurm with PYTHONPATH=src:.  Facade timing includes molecule and
reference preparation; imports, basis preflight and final certificates are
outside it. Repeats create new objects and never inject converged states.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.probes.scf_100.make_plan import build_cases
from benchmarks.probes.scf_solver.profile_stages import digest, plain, peak_rss_bytes, synchronize


def validate_config(cfg):
    cases, _ = build_cases(ROOT, cfg['id'], cfg['basis'], [cfg['engine']])
    expected = next(case for case in cases if case['case_id'] == cfg['case_id'])
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'Frozen input differs at {key}: {cfg.get(key)!r} != {value!r}')


def configure_gradscf(mf, cfg):
    """Call public representations explicitly; missing capabilities stay missing."""
    mf.direct_scf_tol, mf.direct_scf_workers = cfg['direct_scf_tol'], cfg['threads']
    if cfg['representation'] == 'direct':
        return mf.direct_scf()
    options = dict(auxbasis=cfg['auxbasis'])
    if cfg['only_dfj']:
        options['only_dfj'] = True
    try:
        inspect.signature(mf.density_fit).bind(**options)
    except TypeError as error:
        raise NotImplementedError(f'Explicit RI capability unavailable: {options}') from error
    mf.df_tol = cfg['df_tol']
    mf = mf.density_fit(**options)
    config = mf._config()
    if getattr(config, 'df_tol', None) != cfg['df_tol']:
        raise NotImplementedError('The facade does not forward the registered df_tol')
    if cfg['only_dfj'] and not getattr(config, 'only_dfj', False):
        raise NotImplementedError('The facade does not preserve only_dfj=True')
    return mf


def gradscf_dimensions(mf, state):
    """Return scalars only; large input/reference arrays stay local to this call."""
    naux, ngrid = None, None
    for payload in (getattr(mf, '_scf_inputs', None), getattr(mf, 'reference', None), state):
        factors = getattr(payload, 'df_factors', None)
        weights = getattr(payload, 'grid_weights', None)
        if factors is not None: naux = int(factors.shape[0])
        if weights is not None: ngrid = int(weights.size)
        grid = getattr(payload, 'grid_ao', None)
        if grid is None: grid = getattr(payload, 'grid', None)
        if grid is not None and hasattr(grid, 'weights'): ngrid = int(grid.weights.size)
    return naux, ngrid


def reference_strategy(cfg, *, retained_eri):
    if cfg['representation'] == 'df':
        if cfg['only_dfj'] and retained_eri:
            return 'ri_j_packed_k'
        return cfg['ri_strategy']
    return 'packed_incore' if retained_eri else 'shell_direct'


def strategy_matches(cfg, strategy):
    return strategy == (cfg['ri_strategy'] or 'shell_direct')


def pyscf_auxiliary_dimensions(df):
    """Read nominal basis size and an existing factor's row count, without build.

    PySCF 2.13 DF.get_naoaux() builds full factors when _cderi is absent,
    although the pure-J solver may never need them. Disk rank stays unknown;
    this observer neither opens factor files nor requests missing integrals.
    """
    auxmol = getattr(df, 'auxmol', None)
    nominal = None if auxmol is None else int(auxmol.nao_nr())
    shape = getattr(getattr(df, '_cderi', None), 'shape', None)
    rank = None if shape is None else int(shape[0])
    return nominal, rank


def run_reference(mf, history, *, max_cycle, continuation=False, finite_state):
    """Bound main iterations across explicit final-check restarts; extra checks are separate.

    Reentering kernel keeps this mf and its prepared data, but default PySCF
    creates a new local DIIS history. The caller owns the one logical-call timer.
    """
    original_limit = mf.max_cycle
    calls = main_total = extra_total = 0
    begin = len(history)
    try:
        mf.max_cycle = max_cycle
        mf.run()  # Original atom initialization and single-kernel default.
        while True:
            calls += 1
            main = int(mf.cycles)
            checks = history[begin:]
            extra = max(0, len(checks) - main)
            valid_checks = main > 0 and len(checks) in (main, main + 1)
            for index, gate in enumerate(checks):
                gate.update(kernel_call=calls, phase='main' if index < main else 'extra',
                            logical_main_cycle=main_total + min(index + 1, main))
            main_total += main
            extra_total += extra
            terminal = checks[-1] if checks else {}
            finite = bool(finite_state(mf)) and all(math.isfinite(float(terminal.get(key, float('nan'))))
                for key in ('energy_change', 'density_rms', 'gradient'))
            final_passed = bool(terminal.get('passed', False))
            accepted = bool(mf.converged and finite and valid_checks and final_passed)
            if not (continuation and not mf.converged and valid_checks and extra == 1
                    and checks[main - 1].get('passed', False) and not final_passed
                    and finite and main_total < max_cycle):
                break
            mf.max_cycle = max_cycle - main_total
            begin = len(history)
            mf.kernel(dm0=mf.make_rdm1())
    finally:
        mf.max_cycle = original_limit
    return dict(reference_policy='bounded_final_check_restart_v1' if continuation else 'original_single_kernel',
        kernel_calls=calls, total_main_cycles=main_total, total_extra_checks=extra_total,
        total_density_updates=main_total + extra_total, terminal_kernel_cycles=int(mf.cycles),
        reference_finite_terminal=finite, reference_final_gate_passed=final_passed,
        reference_accepted=accepted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference-continuation', action='store_true',
                        help='PySCF only: bounded restart after a rejected strict final check')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    if args.reference_continuation and cfg['engine'] != 'pyscf':
        parser.error('--reference-continuation is only valid for PySCF')
    if not os.getenv('SLURM_JOB_ID'):
        parser.error('Scientific runs require an allocated Slurm step')
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < cfg['threads'] or cfg['threads'] > 64:
        parser.error('The configured CPU count exceeds this allocation')
    os.sched_setaffinity(0, cpus[:cfg['threads']])
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.update(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1',
                      JAX_ENABLE_COMPILATION_CACHE='false', GRADSCF_NATIVE_THREADS=str(cfg['threads']))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'input.json').write_text(json.dumps(cfg, indent=2)+'\n')
    runtime = dict(python=sys.executable, host=platform.node(), job=os.environ['SLURM_JOB_ID'],
        step=os.getenv('SLURM_STEP_ID'), affinity=sorted(os.sched_getaffinity(0)),
        probe_sha256=digest(__file__), input_sha256=digest(args.config), device='cpu', dtype='float64',
        observer_policy='metadata_only_no_df_build',
        reference_continuation=args.reference_continuation,
        reference_policy=('bounded_final_check_restart_v1' if args.reference_continuation else 'original_single_kernel')
            if cfg['engine'] == 'pyscf' else None,
        reference_cycle_budget='200 registered main cycles; extra checks counted separately',
        source_snapshot=os.getenv('GRADSCF_SOURCE_SNAPSHOT'), source_commit=os.getenv('GRADSCF_SOURCE_COMMIT'),
        units='seconds, bytes, Hartree; max_memory in decimal MB',
        initial_guess_note='Both use atom, but GradSCF native atomic UHF and PySCF atomic HF densities differ.')

    def save(name, value):
        (args.output/name).write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')

    def failure(status, note):
        save('summary.json', dict(case_id=cfg['case_id'], engine=cfg['engine'], method=cfg['method'],
            representation=cfg['representation'], ri_strategy=cfg['ri_strategy'],
            status=status, scientific_pass=False, accepted_for_comparison=False, notes=note,
            reference_policy=runtime['reference_policy'], reference_continuation=args.reference_continuation))
        save('runtime.json', runtime)
        return 0

    timer = None
    try:
        import numpy as np
        if cfg['engine'] == 'gradscf':
            import jax
            import gradscf
            from gradscf import dft, gto, scf
            from gradscf.integrals._native import register_integrals
            from gradscf.integrals.basis import basis_from_molecule_spec
            from gradscf.solvers.scf import orbital_gradient_norm
            native = Path(register_integrals()._name).resolve()
            package = Path(gradscf.__file__).parent
            runtime.update(module=gradscf.__file__, jax=jax.__version__, native=str(native),
                           native_sha256=digest(native), source_sha256={str(p.relative_to(package)):digest(p)
                           for p in sorted(package.rglob('*.py'))})
        else:
            # Set the reference's documented process-local config before DF imports
            # bind their lindep defaults. No runtime integral implementation is patched.
            reference_config = args.output/'pyscf_config.py'
            reference_config.write_text(f"df_df_DF_lindep = {cfg['df_tol']!r}\n")
            os.environ['PYSCF_CONFIG_FILE'] = str(reference_config.resolve())
            import pyscf
            from pyscf import dft, gto, lib, scf
            from pyscf.df import incore
            lib.num_threads(cfg['threads'])
            threshold = inspect.signature(incore.cholesky_eri).parameters['lindep'].default
            if threshold != cfg['df_tol']:
                raise RuntimeError('PySCF did not load the registered RI metric threshold')
            native = Path(lib.load_library('libcgto')._name).resolve()
            runtime.update(module=pyscf.__file__, pyscf=pyscf.__version__, libxc=dft.libxc.__version__,
                           native=str(native), native_sha256=digest(native), openmp_threads=int(lib.num_threads()),
                           df_lindep_effective=threshold, pyscf_config_sha256=digest(reference_config))
        runtime.update(numpy=np.__version__, threads={key:os.getenv(key) for key in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'GRADSCF_NATIVE_THREADS')})
        mol_options = dict(atom=cfg['atom'], basis=cfg['basis'], charge=cfg['charge'],
                           spin=cfg['spin'], unit=cfg['unit'], cart=cfg['cart'])
        t = time.perf_counter()
        mol = gto.M(**mol_options)
        nao = (basis_from_molecule_spec(mol.to_spec(), basis=cfg['basis']).nao
               if cfg['engine'] == 'gradscf' else mol.nao_nr())
        runtime.update(nao=int(nao), basis_preflight_seconds=time.perf_counter()-t,
                       forecast_full_eri_bytes=8*nao**4,
                       forecast_s4_eri_bytes=8*(nao*(nao+1)//2)**2)
        del mol
        save('runtime.json', runtime)
        for repeat in range(cfg['repeats']):
            def timeout():
                save('timeout.json', dict(repeat=repeat, scope='one SCF facade', seconds=6000))
                os._exit(124)
            timer = threading.Timer(cfg['single_call_limit_seconds'], timeout)
            timer.daemon = True
            timer.start()
            start = time.perf_counter()
            mol = gto.M(**mol_options)
            factory = getattr(scf if cfg['method'].endswith('HF') else dft, cfg['method'])
            gates, reference = [], {}
            if cfg['engine'] == 'gradscf':
                options = dict(max_cycle=cfg['max_cycle'], conv_tol=cfg['conv_tol'],
                    conv_tol_density=cfg['density_tol'], conv_tol_grad=cfg['gradient_tol'],
                    max_memory=cfg['max_memory'], grids_level=cfg['grid_level'],
                    execution_device='cpu', init_guess=cfg['init_guess'])
                if cfg['method'].endswith('KS'):
                    options['xc'] = cfg['xc']
                mf = configure_gradscf(factory(mol, **options), cfg)
                mf.run()
                synchronize((mf.e_tot, mf.mo_coeff, mf.mo_energy, mf.mo_occ, mf.scf_result, mf.reference), jax)
            else:
                mol.verbose = 0
                mf = factory(mol)
                mf.max_memory, mf.max_cycle = cfg['max_memory'], cfg['max_cycle']
                mf.conv_tol, mf.conv_tol_grad = cfg['conv_tol'], cfg['gradient_tol']
                mf.init_guess, mf.direct_scf_tol = cfg['init_guess'], cfg['direct_scf_tol']
                mf.direct_scf = True
                if cfg['method'].endswith('KS'):
                    mf.xc, mf.grids.level, mf.small_rho_cutoff = cfg['pyscf_xc'], cfg['grid_level'], 0.
                    if cfg['method'] == 'GKS':
                        mf.collinear = 'col'
                if cfg['representation'] == 'df':
                    mf = mf.density_fit(auxbasis=cfg['auxbasis'], only_dfj=cfg['only_dfj'])
                def check(env):
                    de = float(abs(env['e_tot']-env['last_hf_e']))
                    dd = float(np.sqrt(np.mean(np.abs(env['dm']-env['dm_last'])**2)))
                    g = float(env['norm_gorb'])
                    passed = bool(np.isfinite(de+dd+g) and de < cfg['conv_tol']
                                  and dd < cfg['density_tol'] and g < cfg['gradient_tol'])
                    gates.append(dict(energy_change=de, density_rms=dd, gradient=g, passed=passed))
                    return passed
                mf.check_convergence = check
                reference = run_reference(mf, gates, max_cycle=cfg['max_cycle'],
                    continuation=args.reference_continuation,
                    finite_state=lambda current: all(np.isfinite(np.asarray(value)).all() for value in
                        (current.e_tot, current.mo_coeff, current.mo_occ, current.mo_energy)))
            elapsed = time.perf_counter()-start
            timer.cancel()
            coeff, occ = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ)
            naux, ngrid, eri_bytes = None, None, None
            effective_df_rank, naux_kind = None, None
            if cfg['engine'] == 'gradscf':
                state = mf.scf_result if mf.scf_result is not None else mf.reference
                overlap = np.asarray(state.overlap_matrix)
                gradient = float(orbital_gradient_norm(state.fock_matrix, mf.mo_coeff, mf.mo_occ))
                naux, ngrid = gradscf_dimensions(mf, state)
                naux_kind = 'factor_rows' if naux is not None else None
                strategy = cfg['ri_strategy'] or 'shell_direct'
            else:
                overlap = np.asarray(mf.get_ovlp())
                gradient = gates[-1]['gradient'] if gates else float('nan')
                if cfg['representation'] == 'df':
                    naux, effective_df_rank = pyscf_auxiliary_dimensions(mf.with_df)
                    naux_kind = 'nominal_auxiliary_basis' if naux is not None else None
                if hasattr(mf, 'grids') and mf.grids.weights is not None: ngrid = int(mf.grids.weights.size)
                stored = getattr(mf, '_eri', None)
                eri_bytes = None if stored is None else int(stored.nbytes)
                strategy = reference_strategy(cfg, retained_eri=stored is not None)
            if overlap.shape[0]*2 == coeff.shape[-2]:
                zero = np.zeros_like(overlap)
                overlap = np.block([[overlap, zero], [zero, overlap]])
            finite = bool(np.isfinite(mf.e_tot) and np.isfinite(gradient) and np.isfinite(coeff).all())
            scientific_pass = bool(mf.converged and finite and gradient < cfg['gradient_tol']
                                   and reference.get('reference_accepted', True))
            strategy_matched = strategy_matches(cfg, strategy)
            status = 'nonconverged' if not scientific_pass else 'ok' if strategy_matched else 'not_comparable'
            row = dict(case_id=cfg['case_id'], engine=cfg['engine'], method=cfg['method'],
                basis=cfg['basis'], xc=cfg['xc'], xc_family=cfg['xc_family'], natom=cfg['natom'], nao=int(nao),
                nocc=np.count_nonzero(occ, axis=-1).tolist(), nvir=(occ.shape[-1]-np.count_nonzero(occ, axis=-1)).tolist(),
                naux=naux, ngrid=ngrid, naux_kind=naux_kind, effective_df_rank=effective_df_rank,
                representation=cfg['representation'], execution_strategy_actual=strategy, ri_strategy=cfg['ri_strategy'],
                only_dfj=cfg['only_dfj'], auxbasis=cfg['auxbasis'], df_tol=cfg['df_tol'], direct_scf_tol=cfg['direct_scf_tol'],
                approximation=cfg['ri_strategy'] or 'conventional Coulomb and direct exchange, no auxiliary RI',
                grid_level=cfg['grid_level'], init_guess=cfg['init_guess'], dtype='float64', device='cpu', threads=cfg['threads'],
                repeat=repeat, cache_state='cold' if repeat == 0 else 'new-object warm; no converged-state reuse',
                scf_facade_seconds=elapsed, energy_hartree=float(mf.e_tot), gradient_norm=gradient,
                converged=bool(mf.converged), cycles=reference.get('total_main_cycles', int(mf.cycles)),
                scientific_pass=scientific_pass, reference_continuation=args.reference_continuation,
                reference_gate_history=gates if cfg['engine'] == 'pyscf' else None, **reference,
                accepted_for_comparison=scientific_pass and strategy_matched, status=status, retained_eri_bytes=eri_bytes,
                orthogonality_error=float(np.linalg.norm(coeff.conj().swapaxes(-1,-2)@overlap@coeff-np.eye(coeff.shape[-1]))),
                ru_maxrss_bytes=peak_rss_bytes(),
                final_gate=gates[-1] if gates else None, geometry_sha256=cfg['geometry_sha256'],
                protocol_sha256=cfg['protocol_sha256'], notes='Facade includes preparation; certificates/imports excluded. RSS is cumulative process peak, not a per-repeat peak. Auxiliary metadata never builds missing DF factors; nominal naux and existing factor rank are distinct. See guarded process record for full cost.')
            with (args.output/'measurements.jsonl').open('a') as stream:
                stream.write(json.dumps(plain(row), allow_nan=False)+'\n')
            save('summary.json', row)
            if cfg['engine'] == 'gradscf': del state
            else: del stored
            del mf, mol, coeff, occ, overlap
            gc.collect()
            if not scientific_pass: break
        return 0
    except Exception as error:
        if timer is not None: timer.cancel()
        traceback.print_exc()
        status = ('unsupported' if isinstance(error, (NotImplementedError, ImportError)) else
                  'resource_limited' if isinstance(error, MemoryError) or 'memory' in str(error).lower() else 'error')
        return failure(status, f'{type(error).__name__}: {error}')


if __name__ == '__main__':
    raise SystemExit(main())
