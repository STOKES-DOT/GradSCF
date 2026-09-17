"""Gamma-point Si PBE-TDDFT velocity-gauge oscillator-strength spectrum."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import jax
import numpy as np
jax.config.update('jax_enable_x64',True)
from gradscf.pbc import gto,dft,tdscf,optics

EV=optics.HARTREE_TO_EV


def run(args):
    from pyscf.pbc import gto as pgto,dft as pdft,tdscf as ptd
    started=time.perf_counter()
    length=5.43
    settings=dict(atom=[('Si',(0,0,0)),('Si',(length/4,)*3)],
        a=np.array([[0,.5,.5],[.5,0,.5],[.5,.5,0]])*length,unit='Angstrom',
        basis='gth-szv',pseudo='gth-pbe',mesh=(args.mesh,)*3,precision=1e-10)
    cell=gto.M(**settings)
    mf=dft.RKS(cell,xc='pbe',max_cycle=150,conv_tol=1e-11,conv_tol_density=1e-9).run()
    if not mf.converged:raise RuntimeError('GradSCF reference did not converge')
    count=cell.nelec[0]*(cell.topology.nao-cell.nelec[0])
    td=tdscf.TDDFT(mf,nstates=count);td.conv_tol=1e-8;td.max_cycle=150;td.kernel()
    if not np.all(td.converged):raise RuntimeError('GradSCF response did not converge')
    e=np.asarray(td.e);f=np.asarray(td.oscillator_strength())
    print('GradSCF completed',len(e),'states; strength sum',f.sum(),flush=True)
    rcell=pgto.M(**settings,cart=True,verbose=0)
    ref=pdft.RKS(rcell);ref.xc='pbe';ref.conv_tol=1e-11;ref.conv_tol_grad=1e-7;ref.max_cycle=150;ref.kernel()
    if not ref.converged:raise RuntimeError('PySCF reference did not converge')
    rtd=ptd.TDDFT(ref);rtd.nstates=count;rtd.conv_tol=1e-8;rtd.max_cycle=150;rtd.kernel()
    if not np.all(rtd.converged):raise RuntimeError('PySCF response did not converge')
    er=np.asarray(rtd.e);fr=np.asarray(rtd.oscillator_strength(gauge='velocity'))
    if len(er)!=count:raise RuntimeError('Reference returned an incomplete transition space')
    order=np.argsort(e);e,f=e[order],f[order]
    order=np.argsort(er);er,fr=er[order],fr[order]
    if f.sum()<1e-5 or fr.sum()<1e-5:raise RuntimeError('No resolved bright transitions')
    # Individual vectors inside a degenerate manifold are not uniquely paired.
    boundaries=np.r_[0,np.flatnonzero(np.diff(er)*EV>1e-5)+1,len(er)]
    groups=[]
    for begin,end in zip(boundaries[:-1],boundaries[1:]):
        groups.append(dict(first=int(begin),stop=int(end),degeneracy=int(end-begin),
            gradscf_energy_ev=float(e[begin:end].mean()*EV),pyscf_energy_ev=float(er[begin:end].mean()*EV),
            gradscf_strength=float(f[begin:end].sum()),pyscf_strength=float(fr[begin:end].sum())))
    grid=np.linspace(0,max(e.max(),er.max())*EV+2,5001)
    sg=np.asarray(optics.broaden_spectrum(e,f,grid,fwhm_ev=args.fwhm))
    sr=np.asarray(optics.broaden_spectrum(er,fr,grid,fwhm_ev=args.fwhm))
    max_energy_error=float(np.max(abs(e-er))*EV)
    max_strength_error=max(abs(g['gradscf_strength']-g['pyscf_strength']) for g in groups)
    relative_spectrum_error=float(np.max(abs(sg-sr))/sr.max())
    summary=dict(system='silicon',lattice_constant_angstrom=length,method='PBE TDDFT',
        basis='gth-szv',pseudo='gth-pbe',fft_mesh=cell.mesh,k_sampling='Gamma',
        nstates=count,gauge='velocity',nonlocal_gth_correction=True,
        spectrum_quantity='oscillator strength per eV per cell at Gamma',fwhm_ev=args.fwhm,
        gradscf_total_energy_hartree=float(mf.e_tot),pyscf_total_energy_hartree=float(ref.e_tot),
        total_energy_error_hartree=float(abs(mf.e_tot-ref.e_tot)),
        max_excitation_error_ev=max_energy_error,max_group_strength_error=float(max_strength_error),
        max_spectrum_error=float(np.max(abs(sg-sr))),relative_peak_spectrum_error=relative_spectrum_error,
        gradscf_strength_sum=float(f.sum()),pyscf_strength_sum=float(fr.sum()),groups=groups,
        elapsed_seconds=time.perf_counter()-started,
        comparison_passed=bool(max_energy_error<1e-4 and max_strength_error<1e-5 and relative_spectrum_error<1e-4))
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'spectrum.npz',grid_ev=grid,gradscf_spectrum=sg,pyscf_spectrum=sr,
        gradscf_energy_hartree=e,pyscf_energy_hartree=er,gradscf_strength=f,pyscf_strength=fr)
    np.savetxt(args.output/'spectrum.csv',np.column_stack([grid,sg,sr,sg-sr]),delimiter=',',
        header='energy_ev,gradscf_strength_per_ev,pyscf_strength_per_ev,difference',comments='')
    np.savetxt(args.output/'transitions.csv',np.column_stack([np.arange(len(e)),e*EV,er*EV,f,fr]),delimiter=',',
        header='state_index,gradscf_energy_ev,pyscf_energy_ev,gradscf_strength,pyscf_strength',comments='')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    root=Path(__file__).resolve().parents[2]
    sources=[p for directory in ['src/gradscf/pbc','src/gradscf/integrals/periodic'] for p in (root/directory).glob('*.py')]+[Path(__file__).resolve()]
    metadata=dict(command=sys.argv,hostname=platform.node(),platform=platform.platform(),
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        backend=jax.default_backend(),dtype='float64/complex128',
        versions={n:importlib.metadata.version(n) for n in ['jax','jaxlib','numpy','scipy','pyscf','jax-xc']},
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=int,default=41)
    parser.add_argument('--fwhm',type=float,default=.3)
    parser.add_argument('--output',type=Path,required=True)
    summary=run(parser.parse_args())
    sys.exit(0 if summary['comparison_passed'] else 1)
