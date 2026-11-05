"""One invasive PySCF RKS diagnostic; launch under an external <=6000 s guard.
CPU calls are synchronous. Nested inclusive spans include observer overhead;
they must not be added or substituted for a clean reference timing.
"""
import argparse
from collections import defaultdict
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if not os.getenv('SLURM_JOB_ID'):
        parser.error('Scientific diagnostics require Slurm')
    if cfg['engine'] != 'pyscf' or cfg['method'] != 'RKS':
        parser.error('This diagnostic requires a frozen PySCF RKS configuration')
    if not 0 < cfg['max_cycle'] <= 200 or not 0 < cfg['single_call_limit_seconds'] <= 6000:
        parser.error('The reference contract is at most 200 main cycles / 6000 seconds')
    cpus = sorted(os.sched_getaffinity(0))
    if cfg['threads'] != 64 or len(cpus) < 64:
        parser.error('This reference diagnostic requires 64 allocated logical CPUs')
    os.sched_setaffinity(0, cpus[:64])
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    args.output.mkdir(parents=True, exist_ok=False)
    from benchmarks.probes.scf_100.worker import run_reference, validate_config, digest, plain, peak_rss_bytes, pyscf_auxiliary_dimensions
    validate_config(cfg)
    reference_config = args.output/'pyscf_config.py'
    reference_config.write_text(f"df_df_DF_lindep = {cfg['df_tol']!r}\n")
    os.environ['PYSCF_CONFIG_FILE'] = str(reference_config.resolve())
    import numpy as np
    import pyscf
    from pyscf import dft, gto, lib
    from pyscf.df import incore
    from pyscf.dft import numint
    lib.num_threads(cfg['threads'])
    threshold = inspect.signature(incore.cholesky_eri).parameters['lindep'].default
    if threshold != cfg['df_tol']:
        raise RuntimeError('PySCF did not load the registered RI metric threshold')
    package = Path(pyscf.__file__).parent
    runtime = dict(mode='diagnostic', python=sys.executable, python_version=platform.python_version(),
        module=pyscf.__file__, pyscf=pyscf.__version__, numpy=np.__version__, host=platform.node(),
        job=os.environ['SLURM_JOB_ID'], step=os.getenv('SLURM_STEP_ID'), affinity=sorted(os.sched_getaffinity(0)),
        probe_sha256=digest(__file__), config_sha256=digest(args.config), worker_sha256=digest(run_reference.__code__.co_filename),
        openmp_threads=int(lib.num_threads()), threadpools=getattr(lib, 'threadpool_info', lambda: None)(),
        environment={k:os.getenv(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        df_lindep_effective=threshold, configured_repeats=cfg['repeats'], measured_logical_calls=1,
        observer_policy='metadata_only_no_df_build', timing_kind='synchronous_CPU_inclusive_nonadditive',
        device='cpu', dtype='float64', units='seconds, bytes, Hartree',
        timing_scope='run_reference incl. atom guess/continuation; excludes molecule construction and final density export',
        logical_call_limit_seconds=cfg['single_call_limit_seconds'], hard_limit_owner='external process guard',
        source_sha256={name:digest(package/name) for name in ('dft/rks.py','dft/numint.py','dft/gen_grid.py','df/df.py','df/df_jk.py','scf/hf.py')})
    def save(name, value):
        (args.output/name).write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n')
    save('input.json', cfg)
    save('runtime.json', runtime)
    mol = gto.M(atom=cfg['atom'], basis=cfg['basis'], charge=cfg['charge'], spin=cfg['spin'], unit=cfg['unit'], cart=cfg['cart'], verbose=0)
    mf = dft.RKS(mol)
    mf.max_memory, mf.max_cycle = cfg['max_memory'], cfg['max_cycle']
    mf.conv_tol, mf.conv_tol_grad = cfg['conv_tol'], cfg['gradient_tol']
    mf.init_guess, mf.direct_scf_tol, mf.direct_scf = cfg['init_guess'], cfg['direct_scf_tol'], True
    mf.xc, mf.grids.level, mf.small_rho_cutoff = cfg['pyscf_xc'], cfg['grid_level'], 0.
    if cfg['representation'] == 'df':
        mf = mf.density_fit(auxbasis=cfg['auxbasis'], only_dfj=cfg['only_dfj'])
    gates, patches, stack, missing = [], [], [], []
    aggregate = defaultdict(lambda: dict(calls=0, inclusive_seconds=0., errors=0))
    spans = (args.output/'spans.jsonl').open('x')
    def patch(owner, name, label, alias=None):
        aggregate[label]
        original = getattr(owner, name, None)
        if original is None:
            missing.append(label)
            return
        @wraps(original)
        def wrapped(*values, **options):
            call_id, parent = sum(row['calls'] for row in aggregate.values())+len(stack), stack[-1] if stack else None
            stack.append(call_id)
            started, status = time.perf_counter(), 'ok'
            try:
                return original(*values, **options)
            except BaseException:
                status = 'error'
                raise
            finally:
                seconds = time.perf_counter()-started
                spans.write(json.dumps(dict(call_id=call_id, parent_id=parent, depth=len(stack)-1,
                    stage=label, status=status, seconds=seconds, timing_kind='synchronous_CPU_inclusive_nonadditive'))+'\n')
                row = aggregate[label]
                row['calls'] += 1
                row['inclusive_seconds'] += seconds
                row['errors'] += status != 'ok'
                stack.pop()
        for target in (owner, alias) if alias is not None and getattr(alias, name, None) is original else (owner,):
            patches.append((target, name, getattr(target, name)))
            setattr(target, name, wrapped)
    patch(mf.grids, 'build', 'mf.grids.build')
    patch(mf, 'get_veff', 'mf.get_veff')
    patch(getattr(mf, 'with_df', None), 'get_jk', 'mf.with_df.get_jk')
    for name in ('nr_rks', 'eval_xc_eff'):
        patch(mf._numint, name, 'mf._numint.'+name)
    for name in ('eval_ao', '_dot_ao_dm_sparse', '_dot_ao_ao_sparse', '_scale_ao_sparse'):
        patch(numint, name, 'numint.'+name, alias=mf._numint)
    def check(env):
        de = float(abs(env['e_tot']-env['last_hf_e']))
        dd = float(np.sqrt(np.mean(np.abs(env['dm']-env['dm_last'])**2)))
        gradient = float(env['norm_gorb'])
        passed = bool(np.isfinite(de+dd+gradient) and de < cfg['conv_tol'] and dd < cfg['density_tol'] and gradient < cfg['gradient_tol'])
        gates.append(dict(energy_change=de, density_rms=dd, gradient=gradient, passed=passed))
        return passed

    mf.check_convergence = check
    started = time.perf_counter()
    try:
        reference = run_reference(mf, gates, max_cycle=cfg['max_cycle'], continuation=True,
            finite_state=lambda current: all(np.isfinite(np.asarray(x)).all() for x in (current.e_tot,current.mo_coeff,current.mo_occ,current.mo_energy)))
        seconds = time.perf_counter()-started
        nominal, rank = pyscf_auxiliary_dimensions(getattr(mf, 'with_df', None))
        save('summary.json', dict(mode='diagnostic', case_id=cfg['case_id'], seconds=seconds,
            status='diagnostic_accepted' if reference['reference_accepted'] else 'not_accepted',
            energy_hartree=float(mf.e_tot), gradient_norm=gates[-1]['gradient'] if gates else None,
            nao=mol.nao_nr(), ngrid=len(mf.grids.weights), nominal_naux=nominal, effective_df_rank=rank,
            scientific_pass=reference['reference_accepted'], accepted_for_timing_comparison=False, **reference))
        if reference['reference_accepted']:
            np.save(args.output/'density.npy', mf.make_rdm1())
        return 0 if reference['reference_accepted'] else 2
    except Exception as error:
        save('summary.json', dict(status='error', error=f'{type(error).__name__}: {error}', scientific_pass=False, accepted_for_timing_comparison=False))
        raise
    finally:
        for owner, name, original in reversed(patches): setattr(owner, name, original)
        spans.close()
        save('gate_history.json', gates)
        save('aggregate.json', dict(timing_kind='synchronous_CPU_inclusive_nonadditive', stages=dict(aggregate), unavailable=missing))
        runtime.update(diagnostic_seconds=time.perf_counter()-started, ru_maxrss_bytes=peak_rss_bytes(),
            actual_mo_dtype=None if mf.mo_coeff is None else str(mf.mo_coeff.dtype))
        save('runtime.json', runtime)


if __name__ == '__main__':
    raise SystemExit(main())
