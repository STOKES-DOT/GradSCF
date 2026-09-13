"""Independent GradSCF/PySCF PBE bands for diamond-structure Si and C."""
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
from gradscf.pbc import gto,dft

EV=27.211386245988
SYSTEMS={'silicon':('Si',5.43),'diamond':('C',3.567)}
LABELS=['G','X','W','K','G','L']
# Conventional Cartesian reciprocal coordinates, in units of 2*pi/a.
VERTICES=np.array([[0,0,0],[0,1,0],[.5,1,0],[.75,.75,0],[0,0,0],[.5,.5,.5]])


def run(name,args):
    from pyscf.pbc import gto as pgto,dft as pdft,df as pdf
    started=time.perf_counter();symbol,length=SYSTEMS[name]
    lattice=np.array([[0,.5,.5],[.5,0,.5],[.5,.5,0]])*length
    atom=[(symbol,(0.,0.,0.)),(symbol,(length/4,)*3)]
    settings=dict(atom=atom,a=lattice,unit='Angstrom',basis='gth-szv',pseudo='gth-pbe',
                  mesh=(args.mesh,)*3,precision=1e-10)
    cell=gto.M(**settings);kpts=np.asarray(cell.make_kpts((args.scf_mesh,)*3))
    mf=dft.KRKS(cell,kpts=kpts,xc='pbe',max_cycle=150,conv_tol=1e-11,
                conv_tol_density=1e-9,conv_tol_grad=1e-7)
    mf.kernel()
    grad_scf_seconds=time.perf_counter()-started
    if not mf.converged:raise RuntimeError(f'{name}: GradSCF not converged')
    print(name,'GradSCF converged',float(mf.e_tot),'cycles',mf.cycles,flush=True)
    rcell=pgto.M(**settings,cart=True,verbose=0)
    ref=pdft.KRKS(rcell,kpts=kpts);ref.xc='pbe';ref.with_df=pdf.FFTDF(rcell,kpts=kpts)
    ref.conv_tol=1e-11;ref.conv_tol_grad=1e-7;ref.max_cycle=150
    checkpoint=time.perf_counter();ref.kernel();reference_scf_seconds=time.perf_counter()-checkpoint
    if not ref.converged:raise RuntimeError(f'{name}: PySCF not converged')
    nocc=cell.nelectron//2
    if not np.all(np.sum(np.asarray(ref.mo_occ)>0,axis=1)==nocc):
        raise RuntimeError('Reference occupation counts differ across k; fixed-band model invalid.')
    vertices=VERTICES*(2*np.pi/(length*1.8897261245650618))
    path=np.concatenate([np.linspace(a,b,args.points_per_segment,endpoint=False)
                         for a,b in zip(vertices[:-1],vertices[1:])]+[vertices[-1:]])
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    tick_indices=np.arange(len(LABELS))*args.points_per_segment
    checkpoint=time.perf_counter();grad_bands,_=mf.get_bands(path,chunk_size=args.chunk_size)
    grad_bands=np.asarray(grad_bands);grad_band_seconds=time.perf_counter()-checkpoint
    reference=[];checkpoint=time.perf_counter()
    for first in range(0,len(path),args.chunk_size):
        bands,_=ref.get_bands(path[first:first+args.chunk_size]);reference.extend(bands)
    reference=np.asarray(reference);reference_band_seconds=time.perf_counter()-checkpoint
    delta=(grad_bands-reference)*EV
    # Both engines use ONE reference zero. Do not independently align them.
    vbm=reference[:,:nocc].max()
    summary=dict(system=name,element=symbol,lattice_constant_angstrom=length,
        method='PBE',basis='gth-szv',pseudo='gth-pbe',fft_mesh=cell.mesh,
        scf_kmesh=[args.scf_mesh]*3,kpath=LABELS,n_path_points=len(path),n_bands=grad_bands.shape[1],
        nocc=nocc,energy_zero='PySCF valence maximum on the sampled path',energy_zero_hartree=float(vbm),
        gradscf_total_energy_hartree=float(mf.e_tot),pyscf_total_energy_hartree=float(ref.e_tot),
        total_energy_error_hartree=float(abs(mf.e_tot-ref.e_tot)),
        max_band_error_ev=float(np.max(np.abs(delta))),rms_band_error_ev=float(np.sqrt(np.mean(delta**2))),
        gradscf_sampled_gap_ev=float((grad_bands[:,nocc:].min()-grad_bands[:,:nocc].max())*EV),
        pyscf_sampled_gap_ev=float((reference[:,nocc:].min()-reference[:,:nocc].max())*EV),
        gradscf_cycles=mf.cycles,pyscf_cycles=int(ref.cycles),
        timing_seconds=dict(gradscf_scf=grad_scf_seconds,pyscf_scf=reference_scf_seconds,
            gradscf_bands=grad_band_seconds,pyscf_bands=reference_band_seconds,total=time.perf_counter()-started),
        comparison_passed=bool(np.max(np.abs(delta))<1e-3 and abs(mf.e_tot-ref.e_tot)<1e-6))
    out=args.output/name;out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'bands.npz',kpoints=path,distance=distance,tick_indices=tick_indices,
        gradscf_hartree=grad_bands,pyscf_hartree=reference,reference_zero_hartree=vbm,nocc=nocc)
    rows=np.array([[ik,ib,distance[ik],*path[ik],grad_bands[ik,ib],reference[ik,ib],delta[ik,ib]]
                   for ik in range(len(path)) for ib in range(grad_bands.shape[1])])
    np.savetxt(out/'bands.csv',rows,delimiter=',',header='k_index,band_index,distance_bohr_inv,kx,ky,kz,gradscf_hartree,pyscf_hartree,difference_ev',comments='')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--systems',nargs='+',choices=SYSTEMS,default=list(SYSTEMS))
    parser.add_argument('--mesh',type=int,default=41)
    parser.add_argument('--scf-mesh',type=int,default=2)
    parser.add_argument('--points-per-segment',type=int,default=12)
    parser.add_argument('--chunk-size',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    metadata=dict(command=sys.argv,hostname=platform.node(),platform=platform.platform(),
        backend=jax.default_backend(),dtype='float64/complex128',
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        versions={n:importlib.metadata.version(n) for n in ['jax','jaxlib','numpy','scipy','pyscf','jax-xc']},
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['src/gradscf/pbc','src/gradscf/integrals/periodic'] for p in (root/folder).glob('*.py')})
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    summaries=[run(name,args) for name in args.systems]
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    sys.exit(0 if all(s['comparison_passed'] for s in summaries) else 1)
