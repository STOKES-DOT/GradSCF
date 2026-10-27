"""Fresh-object end-to-end measurements; no external orbitals in GradSCF."""
import argparse
import cProfile
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np


def plain(x):
    if hasattr(x, 'shape'):
        return plain(np.asarray(x).tolist())
    if isinstance(x,float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    return x


def ready(x):
    if hasattr(x, 'block_until_ready'):
        x.block_until_ready()
    elif isinstance(x, (tuple, list)):
        for value in x:
            ready(value)
    elif isinstance(x, dict):
        for value in x.values():
            ready(value)
    elif is_dataclass(x):
        for field in fields(x):
            ready(getattr(x, field.name))
    elif getattr(x, 'result', None) is not None:
        ready(x.result)
    return x


def calculate(cfg, folder, repeat):
    engine, method = cfg['engine'], cfg['method']
    is_dft = method.startswith('pbe')
    xc = 'pbe0' if method.startswith('pbe0') else 'pbe'
    options = dict(atom=cfg['atom'], basis=cfg['basis'], charge=0, spin=0,
                   unit='Angstrom', cart=True)
    stages = {}

    def timed(name, fn):
        t0 = time.perf_counter()
        value = fn()
        ready(value)
        stages[name] = time.perf_counter() - t0
        return value

    if engine == 'gradscf':
        import jax
        jax.config.update('jax_enable_x64', True)
        from gradscf import gto, scf, dft
        task_start = time.perf_counter()
        mol = timed('molecule', lambda: gto.M(**options))
        kw = dict(execution_device=cfg.get('device', 'cpu'),
                  conv_tol=1e-10, conv_tol_density=1e-9, conv_tol_grad=1e-8,
                  max_cycle=100, max_memory=60000., init_guess='hcore')
        mf = dft.RKS(mol, xc=xc, grids_level=cfg.get('grid_level', 2), **kw) if is_dft else scf.RHF(mol, **kw)
        if cfg.get('jk_backend') == 'direct':
            mf.jk_backend = 'direct'
        if method == 'dfmp2' or cfg.get('ri', False):
            mf.density_fit('weigend')
        timed('scf', lambda: mf.run().e_tot)
        nocc = int(np.count_nonzero(np.asarray(mf.mo_occ)))
        ints = mf._scf_inputs
        naux = None if getattr(ints, 'df_factors', None) is None else int(ints.df_factors.shape[0])
        result = dict(hf_energy=float(mf.e_tot), scf_iterations=int(mf.cycles),
                      converged=bool(mf.converged), naux=naux)
        if method in ('mp2', 'dfmp2'):
            from gradscf import mp
            post = timed('correlation', lambda: mp.MP2(mf, with_t2=True).run())
            ready(post.e_corr)
            result.update(correlation_energy=float(post.e_corr), energy=float(post.e_tot),
                          t2_shape=list(post.t2.shape), converged=result['converged'] and post.converged)
        elif method in ('ccsd', 'ccsdt'):
            from gradscf import cc
            post = timed('correlation', lambda: cc.CCSD(mf, conv_tol=1e-10, residual_tol=1e-9).run())
            ready(post.e_corr)
            result.update(correlation_energy=float(post.e_corr), energy=float(post.e_tot),
                          cc_iterations=int(post.result.iterations), cc_residual=float(post.result.residual_norm),
                          t1_shape=list(post.t1.shape), t2_shape=list(post.t2.shape),
                          converged=result['converged'] and post.converged)
            if method == 'ccsdt':
                result['triples_energy'] = float(timed('triples', post.ccsd_t))
        elif method.endswith(('_tda', '_td')):
            from gradscf import tdscf
            post = (tdscf.TDA if method.endswith('_tda') else tdscf.TDDFT)(mf,
                nstates=5, eigensolver='davidson', davidson_tol=1e-8, davidson_max_iter=100)
            timed('response', lambda: post.kernel().excitation_energies)
            result.update(excitations=plain(post.e), converged=result['converged'] and bool(np.all(post.converged)))
            result['oscillator_strengths'] = plain(timed('oscillator_strengths', post.oscillator_strength))
            for key in ('iterations', 'residual_norms', 'residual_norm', 'converged'):
                if hasattr(post.result, key):
                    result['response_'+key] = plain(getattr(post.result, key))
        elif method == 'gwbse':
            from gradscf import gw, bse
            post = gw.GW(mf, method='g0w0', nw=100, eta=1e-5, max_cycle=100, conv_tol=1e-10)
            timed('gw', lambda: post.run(orbs=cfg.get('qp_orbitals')).mo_energy)
            result.update(qp_energy=plain(post.mo_energy), qp_residual=plain(post.result.qp_residual),
                          gw_converged=bool(post.converged))
            for key in ('qp_candidate_count','qp_branch_valid','qp_weight','qp_on_shell','qp_linearized'):
                result[key] = plain(getattr(post.result,key))
            for tda in (True, False):
                label = 'bse_tda' if tda else 'bse_full'
                response = bse.BSE(post, nroots=5, tda=tda, singlet=True, solver='davidson', conv_tol=1e-8,
                    occupied=cfg.get('bse_occupied'), virtual=cfg.get('bse_virtual'))
                timed(label, lambda: response.run().e)
                result[label] = plain(response.e)
                result[label+'_residual'] = plain(response.result.residual_norms)
                result['converged'] &= bool(np.all(response.converged))
            result['converged'] &= result['gw_converged']
            if repeat == 0:
                ref = response.reference
                np.savez(folder/'screening.npz', qp_energy=ref.qp_energy, screening_energy=ref.screening_energy,
                         mo_factors=ref.mo_factors, nocc=ref.nocc)
        else:
            result['energy'] = float(mf.e_tot)
        stages['task_wall'] = time.perf_counter()-task_start
        s = np.asarray(mf.scf_result.overlap_matrix)
        h = np.asarray(mf.scf_result.hcore_matrix)
        density = np.asarray(mf.scf_result.density_matrix)
        fock = np.asarray(mf.scf_result.fock_matrix)
        result['ngrid'] = int(ints.grid_weights.shape[0]) if is_dft else None
    else:
        from pyscf import gto, scf, dft, mp, cc, gw, lib
        lib.num_threads(cfg['threads'])
        task_start = time.perf_counter()
        mol = timed('molecule', lambda: gto.M(**options, verbose=0, max_memory=60000))
        mf = dft.RKS(mol) if is_dft else scf.RHF(mol)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle, mf.init_guess = 1e-10, 1e-8, 100, 'hcore'
        if is_dft:
            mf.xc, mf.grids.level = xc, cfg.get('grid_level', 2)
            mf.small_rho_cutoff = 0
        if method == 'dfmp2' or cfg.get('ri', False):
            mf = mf.density_fit(auxbasis='weigend')
        timed('scf', mf.kernel)
        nocc = int(np.count_nonzero(mf.mo_occ))
        result = dict(hf_energy=float(mf.e_tot), scf_iterations=getattr(mf, 'cycles', None),
                      converged=bool(mf.converged), naux=None)
        if method in ('pbe_grad','mp2_grad'):
            post = mf
            if method == 'mp2_grad':
                post = mp.MP2(mf)
                timed('correlation',lambda: post.kernel(with_t2=True))
            derivative = post.nuc_grad_method()
            if method == 'pbe_grad':
                derivative.grid_response = True
            gradient = timed('gradient',derivative.kernel)
            result.update(energy=float(post.e_tot),gradient=plain(gradient),
                gradient_units='hartree/bohr',orbital_response=True,grid_response=method=='pbe_grad')
        elif method in ('mp2', 'dfmp2'):
            post = mp.MP2(mf)
            if method == 'dfmp2':
                post = post.density_fit(auxbasis='weigend')
            timed('correlation', lambda: post.kernel(with_t2=True))
            result.update(correlation_energy=float(post.e_corr), energy=float(post.e_tot), t2_shape=list(post.t2.shape))
            if method == 'dfmp2':
                result['naux'] = int(mf.with_df.get_naoaux())
        elif method in ('ccsd', 'ccsdt'):
            post = cc.CCSD(mf)
            post.conv_tol, post.conv_tol_normt, post.max_cycle = 1e-10, 1e-9, 100
            timed('correlation', post.kernel)
            result.update(correlation_energy=float(post.e_corr), energy=float(post.e_tot),
                          converged=result['converged'] and bool(post.converged),
                          t1_shape=list(post.t1.shape), t2_shape=list(post.t2.shape))
            if method == 'ccsdt':
                result['triples_energy'] = float(timed('triples', post.ccsd_t))
        elif method.endswith(('_tda', '_td')):
            post = mf.TDA() if method.endswith('_tda') else mf.TDDFT()
            post.nstates, post.conv_tol, post.max_cycle = 5, 1e-8, 100
            timed('response', post.kernel)
            result.update(excitations=plain(post.e), converged=result['converged'] and bool(np.all(post.converged)))
            result['oscillator_strengths'] = plain(timed('oscillator_strengths', post.oscillator_strength))
        elif method == 'gwbse':
            from pyscf.gw.gw_cd import GWCD
            post = GWCD(mf)
            post.eta, post.linearized = 1e-5, False
            timed('gw', lambda: post.kernel(nw=100, vhf_df=cfg.get('gw_vhf_df',False),orbs=cfg.get('qp_orbitals')))
            result.update(qp_energy=plain(post.mo_energy), gw_converged=bool(post.converged),
                          bse_status='unsupported', converged=result['converged'] and bool(post.converged))
        else:
            result['energy'] = float(mf.e_tot)
        stages['task_wall'] = time.perf_counter()-task_start
        s, h, density = mf.get_ovlp(), mf.get_hcore(), mf.make_rdm1()
        fock = mf.get_fock(dm=density)
        result['ngrid'] = len(mf.grids.weights) if is_dft else None
    result.update(nao=s.shape[0], nocc=nocc, nvir=s.shape[0]-nocc,
                  ntransition=nocc*(s.shape[0]-nocc), scf_commutator_max=float(np.max(abs(fock@density@s-s@density@fock))))
    if repeat == 0:
        np.savez(folder/'reference.npz', overlap=s, hcore=h, density=density, fock=fock,
                 mo_coeff=mf.mo_coeff, mo_energy=mf.mo_energy, mo_occ=mf.mo_occ)
        if engine == 'pyscf':
            (folder/'actual-basis.json').write_text(json.dumps(mol._basis, indent=2)+'\n')
        else:
            from gradscf.integrals.basis.data import load_basis_from_snapshot
            symbols = {line.strip().split()[0] for line in cfg['atom'].split(';')}
            basis = {symbol: load_basis_from_snapshot(cfg['basis'], symbol) for symbol in symbols}
            (folder/'actual-basis.json').write_text(json.dumps(basis, indent=2)+'\n')
            if naux is not None:
                np.save(folder/'df-factors.npy', np.asarray(ints.df_factors))
    return result, stages


def main(*, provenance_probes=()):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    input_bytes = args.config.read_bytes()
    cfg = json.loads(input_bytes)
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for repeat in range(cfg.get('repeats', 4)):
        t0 = time.perf_counter()
        row = dict(run_id=cfg['case_id']+f'-r{repeat}', repeat=repeat,
            start_monotonic=t0,
            timing_scope='independent_geometry_to_result_fresh_object',
            cache_state='new_process_first' if repeat == 0 else 'same_process_new_objects_compilation_allowed',
            final_result_reused=False,
            config=cfg, affinity=sorted(os.sched_getaffinity(0)))
        try:
            if args.profile:
                profiler = cProfile.Profile()
                profiler.enable()
            row['values'], row['stages'] = calculate(cfg, folder, repeat)
            row['status'] = 'ok' if row['values']['converged'] else 'nonconverged'
            if args.profile:
                profiler.disable()
                profiler.dump_stats(folder/f'profile-{repeat}.pstats')
        except Exception as exc:
            row.update(status='unsupported' if isinstance(exc, (NotImplementedError, ImportError)) else 'error',
                       exception=repr(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        row['wall_seconds'] = time.perf_counter()-t0
        row['end_monotonic'] = time.perf_counter()
        row['peak_rss_bytes'] = 1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        row['memory_method'] = 'Linux ru_maxrss process high-water mark including earlier repeats; controller samples group separately'
        records.append(row)
        (folder/'records.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(row), flush=True)
        if row['status'] != 'ok':
            break
    from provenance import collect
    provenance = collect((__file__, *provenance_probes))
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    for row in records:
        row.update(provenance=provenance, input_sha256=input_sha256)
    (folder/'records.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
