"""Independent orbital minimization / Newton comparisons for difficult open shells."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import hashlib
import importlib.metadata
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback
from ten_system_scf_matrix import SYSTEMS, XC_REFERENCE, prepare, json_safe
import numpy as np
from gradscf.scf.init_guess import orbital_rotation_guesses


def reference_object(method,xc,p):
    from pyscf import scf,dft
    mf=getattr(scf if xc=='hf' else dft,method.replace('_ncol',''))(p['mol'])
    mf._eri=p['eri_reference'];mf.direct_scf=False;mf.chkfile=None
    if xc!='hf':
        mf.xc=XC_REFERENCE[xc];mf.grids.coords=p['coords'];mf.grids.weights=p['weights'];mf.small_rho_cutoff=0.
        if method.startswith('GKS'):mf.collinear='ncol' if method.endswith('ncol') else 'col'
    return mf


def run_case(args,system,method,xc):
    import jax
    from gradscf.scf import orbital_optimization as opt
    p=prepare(SYSTEMS[system],args.basis,args.grid_level)
    out=args.output/(system+'_'+method+'_'+xc);out.mkdir(parents=True,exist_ok=True)
    oa=np.zeros(len(p['c']));ob=oa.copy();oa[:p['na']]=1;ob[:p['nb']]=1
    attempts=[]
    for index,c in enumerate(orbital_rotation_guesses(p['c'],amplitudes=args.amplitudes,seed=args.seed)):
        row=dict(index=index,amplitude=args.amplitudes[index])
        common=dict(overlap=p['s'],hcore=p['h'],eri=p['eri'],nuclear_repulsion=p['enuc'],ao=p['ao'],ao_deriv1=p['deriv'],grid_weights=p['weights'],xc_spec=xc,max_iterations=500,gradient_tolerance=1e-7,gradient_mode=args.gradient_mode)
        if method in {'UKS','UHF'}:
            fn=opt.minimize_uks_from_integrals;coeff=np.stack([c,c]);occ=np.stack([oa,ob]);ref_occ=occ
        elif method in {'ROKS','ROHF'}:
            fn=opt.minimize_roks_from_integrals;coeff=c;occ=np.stack([oa,ob]);ref_occ=oa+ob
        else:
            fn=opt.minimize_gks_from_integrals;z=np.zeros_like(c);coeff=np.block([[c,z],[z,c]]).astype(complex);occ=np.concatenate([oa,ob]);ref_occ=occ
            common['collinear']='ncol' if method.endswith('ncol') else 'col'
        mf=reference_object(method,xc,p)
        try:
            start=time.perf_counter();r=fn(**common,mo_coeff=coeff,mo_occ=occ)
            dm=np.asarray(r.density_matrix)
            cross_energy=float(mf.energy_tot(dm=dm))
            vf=mf.get_veff(dm=dm);fock=mf.get_fock(dm=dm,vhf=vf)
            nr_occ=np.asarray(r.mo_occ).sum(axis=0) if method in {'ROKS','ROHF'} else np.asarray(r.mo_occ)
            cross_gradient=float(np.linalg.norm(mf.get_grad(np.asarray(r.mo_coeff),nr_occ,fock)))
            row['gradscf']=dict(energy=r.total_energy,stationary=r.stationary,gradient_norm=r.gradient_norm,
                optimizer_success=r.optimizer_success,optimizer_message=r.optimizer_message,iterations=r.iterations,
                evaluations=r.evaluations,elapsed_seconds=time.perf_counter()-start,
                reference_energy_at_density=cross_energy,cross_energy_error=abs(r.total_energy-cross_energy),reference_gradient=cross_gradient,
                stage_history=r.stage_history,polishing_history=r.polishing_history,
                roundoff_energy_allowance=r.roundoff_energy_allowance)
            np.savez_compressed(out/f'native-{index}.npz',density=dm,mo_coeff=np.asarray(r.mo_coeff),mo_occ=np.asarray(r.mo_occ))
        except Exception as e:row['gradscf_error']=dict(error=str(e),traceback=traceback.format_exc())
        try:
            start=time.perf_counter()
            if args.reference_solver == 'orbital':
                from pyscf_orbital_minimizer import minimize_pyscf_orbitals
                reference=minimize_pyscf_orbitals(mf,coeff,occ,method=method,
                    max_iterations=500,gradient_tolerance=1e-7)
                row['pyscf']=dict(energy=reference.total_energy,converged=reference.stationary,
                    gradient_norm=reference.gradient_norm,optimizer_success=reference.optimizer_success,
                    optimizer_message=reference.optimizer_message,iterations=reference.iterations,
                    evaluations=reference.evaluations,elapsed_seconds=time.perf_counter()-start,
                    polishing_steps=reference.polish_steps,polishing_history=reference.polish_history)
                np.savez_compressed(out/f'reference-{index}.npz',density=reference.density_matrix,
                    mo_coeff=reference.mo_coeff,mo_occ=reference.mo_occ)
            else:
                newton=mf.newton();newton.conv_tol=1e-11;newton.conv_tol_grad=1e-7;newton.max_cycle=200;newton.conv_check=False
                newton.kernel(mo_coeff=coeff,mo_occ=ref_occ)
                row['pyscf']=dict(energy=float(newton.e_tot),converged=bool(newton.converged),elapsed_seconds=time.perf_counter()-start)
        except Exception as e:row['pyscf_error']=dict(error=str(e),traceback=traceback.format_exc())
        attempts.append(row)
        (out/'attempts.json').write_text(json.dumps(json_safe(attempts),indent=2)+'\n')
        print(system,method,xc,json.dumps(json_safe(row)),flush=True)
    good_native=[r for r in attempts if r.get('gradscf',{}).get('stationary') and r['gradscf']['cross_energy_error']<1e-8 and r['gradscf']['reference_gradient']<1e-7]
    good_reference=[r for r in attempts if r.get('pyscf',{}).get('converged')]
    g=min(good_native,key=lambda r:r['gradscf']['energy']) if good_native else None
    r=min(good_reference,key=lambda r:r['pyscf']['energy']) if good_reference else None
    error=abs(g['gradscf']['energy']-r['pyscf']['energy']) if g and r else None
    summary=dict(system=system,method=method,xc=xc,seed=args.seed,amplitudes=args.amplitudes,gradient_mode=args.gradient_mode,
        devices=[str(device) for device in jax.devices()],input_errors=p['input_errors'],
        native_optimizer_sha256=hashlib.sha256(Path(opt.__file__).read_bytes()).hexdigest(),
        reference_optimizer_sha256=hashlib.sha256(Path(__file__).with_name('pyscf_orbital_minimizer.py').read_bytes()).hexdigest(),
        gradient_tolerance=1e-7,reference_solver=args.reference_solver,gradscf=g,pyscf=r,energy_error=error,pass_comparison=error is not None and error<(1e-8 if xc=='hf' else 1e-6))
    (out/'summary.json').write_text(json.dumps(json_safe(summary),indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--systems',nargs='+',default=['OH','NO']);p.add_argument('--methods',nargs='+',default=['UKS','ROKS','GKS']);p.add_argument('--xcs',nargs='+',default=['svwn']);p.add_argument('--worker',nargs=3);p.add_argument('--jobs',type=int,default=2);p.add_argument('--basis',default='3-21g');p.add_argument('--grid-level',type=int,default=1);p.add_argument('--amplitudes',nargs='+',type=float,default=[0.,.1]);p.add_argument('--seed',type=int,default=20260913);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-solver',choices=['orbital','newton'],default='orbital')
    p.add_argument('--gradient-mode',choices=['implicit','unrolled'],default='implicit')
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if a.worker:return run_case(a,*a.worker)
    if any(a.output.iterdir()):
        raise ValueError('Use a fresh output directory; existing attempts must be preserved')
    started=time.perf_counter()
    root=Path(__file__).resolve().parents[2]
    sources=[Path(__file__), Path(__file__).with_name('pyscf_orbital_minimizer.py'),
             Path(__file__).with_name('ten_system_scf_matrix.py')]
    sources.extend((root/'src/gradscf').rglob('*.py'))
    metadata=dict(command=[sys.executable,*sys.argv],hostname=platform.node(),platform=platform.platform(),
        started_utc=datetime.now(timezone.utc).isoformat(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        thread_environment={key:os.environ.get(key) for key in
            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','XLA_FLAGS')},
        backend='cpu',dtype='float64/complex128',basis=a.basis,cartesian=True,grid_level=a.grid_level,
        systems={s:SYSTEMS[s] for s in a.systems},geometry_unit='Angstrom',energy_unit='Hartree',
        gradient_tolerance=1e-7,energy_tolerance_hf=1e-8,energy_tolerance_dft=1e-6,
        seed=a.seed,amplitudes=a.amplitudes,max_iterations=500,jobs=a.jobs,gradient_mode=a.gradient_mode,
        versions={name:importlib.metadata.version(name) for name in ('jax','jaxlib','numpy','scipy','pyscf','jax-xc')},
        source_hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        native_source_hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (root/'src/gradscf/integrals/_native/csrc').iterdir() if f.is_file()},
        vendor_manifest_sha256=hashlib.sha256((root/'src/gradscf/integrals/_native/vendor/manifest.json').read_bytes()).hexdigest())
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    cases=[(s,m,x) for s in a.systems for m in a.methods for x in a.xcs if not (m=='GKS_ncol' and x not in {'lda','svwn'})]
    def worker(case):
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',*case,'--basis',a.basis,'--grid-level',str(a.grid_level),'--seed',str(a.seed),'--amplitudes',*map(str,a.amplitudes),'--reference-solver',a.reference_solver,'--gradient-mode',a.gradient_mode,'--output',str(a.output)]
        with (a.output/('_'.join(case)+'.log')).open('w') as log:
            return subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:codes=list(pool.map(worker,cases))
    rows=[json.loads(f.read_text()) for f in a.output.glob('*/summary.json')]
    summary=dict(returncodes=codes,expected_cases=len(cases),rows=rows,passed=sum(r['pass_comparison'] for r in rows),
        elapsed_seconds=time.perf_counter()-started,complete=len(rows)==len(cases) and all(code==0 for code in codes))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Fallback comparisons:',summary['passed'],'/',len(cases),flush=True)
    if not summary['complete'] or summary['passed'] != len(cases):
        raise SystemExit(1)

if __name__=='__main__':main()
