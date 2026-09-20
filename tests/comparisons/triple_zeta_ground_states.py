"""Large-ERI ground-state regression on fixed aromatic geometries.

Native GradSCF S/H/ERI; independent PySCF SCF; identical reference spherical
AO grid values for DFT to isolate SCF/ERI behavior from quadrature conventions.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_ENABLE_X64','1')
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh
from pyscf import gto,dft
from gradscf import integrals,scf
from gradscf.data.molecule import parse_molecule_spec
from ten_system_scf_matrix import run_gradscf,run_pyscf,XC_REFERENCE


def cases(charge):
    if charge==0:
        for method in ('RHF','UHF','ROHF','GHF'):yield method,'hf'
        for xc in ('pbe','pbe0'):
            for method in ('RKS','UKS','ROKS','GKS'):yield method,xc
    else:
        for method in ('UHF','ROHF','GHF'):yield method,'hf'
        for method in ('UKS','ROKS','GKS'):yield method,'pbe0'


def run(args):
    geo=json.loads(Path(args.geometry).read_text())
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    atom=list(zip(geo['symbols'],geo['coords_angstrom']))
    top,params=integrals.prepare_basis(atom,args.basis,cart=False)
    plan=integrals.make_plan(top)
    started=time.perf_counter()
    print('Native integral preparation',geo['name'],args.basis,top.nao,flush=True)
    s=plan.evaluate('overlap',params)
    h=plan.evaluate('kinetic',params)+plan.evaluate('nuclear',params)
    eri=plan.evaluate('eri',params,aosym='s8');eri.block_until_ready()
    # Keep one real AO ERI shared by all method cases. References calculate
    # their own integrals; do not materialize a second full PySCF ERI tensor.
    neutral=gto.M(atom=atom,basis=args.basis,cart=False,unit='Angstrom',verbose=0)
    np.testing.assert_allclose(s,neutral.intor('int1e_ovlp'),atol=1e-10,rtol=1e-11)
    np.testing.assert_allclose(h,neutral.intor('int1e_kin')+neutral.intor('int1e_nuc'),atol=1e-9,rtol=1e-11)
    print('Independent compressed PySCF ERI preparation',flush=True)
    eri_reference=neutral.intor('int2e',aosym='s8')
    rng=np.random.default_rng(311)
    ijkl=rng.integers(0,top.nao,size=(512,4),dtype=np.int64)
    hi=np.maximum(ijkl[:,0],ijkl[:,1]);lo=np.minimum(ijkl[:,0],ijkl[:,1])
    ij=hi*(hi+1)//2+lo
    hi=np.maximum(ijkl[:,2],ijkl[:,3]);lo=np.minimum(ijkl[:,2],ijkl[:,3])
    kl=hi*(hi+1)//2+lo
    hi=np.maximum(ij,kl);lo=np.minimum(ij,kl)
    native_samples=np.asarray(eri)[hi*(hi+1)//2+lo]
    reference_samples=eri_reference[hi*(hi+1)//2+lo]
    np.testing.assert_allclose(native_samples,reference_samples,atol=1e-10,rtol=1e-10)
    spec=parse_molecule_spec(atom)
    coords,weights=integrals.build_molecular_grid_from_spec(spec,level=args.grid_level)
    deriv=dft.numint.eval_ao(neutral,np.asarray(coords),deriv=1)
    metadata=dict(geometry=geo,basis=args.basis,spherical=True,nao=top.nao,eri_elements=int(eri.size),eri_layout='s8',
        grid_level=args.grid_level,ngrid=len(weights),max_cycle=args.max_cycle,
        tolerances=dict(hf_energy=1e-8,dft_energy=1e-6,electron_count=1e-6,residual=1e-6),
        grid_ao_source='PySCF spherical AO evaluation on the shared GradSCF quadrature; SCF/ERI regression only',
        jax=jax.__version__,host=platform.node(),affinity=sorted(os.sched_getaffinity(0)),
        integral_seconds=time.perf_counter()-started,
        native_eri_sample_error=float(np.max(np.abs(native_samples-reference_samples))),
        reference_eri_storage='independent PySCF s8, shared by method cases',
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src/gradscf').rglob('*.py') if 'build' not in p.parts and '__pycache__' not in p.parts})
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Integrals ready',metadata['integral_seconds'],flush=True)
    for charge in args.charges:
        spin=charge%2
        mol=gto.M(atom=atom,basis=args.basis,cart=False,unit='Angstrom',charge=charge,spin=spin,verbose=0)
        na,nb=mol.nelec;eps,c=eigh(np.asarray(h),np.asarray(s))
        da,db=c[:,:na]@c[:,:na].T,c[:,:nb]@c[:,:nb].T;z=np.zeros_like(da)
        p=dict(s=s,h=h,eri=eri,enuc=mol.energy_nuc(),ao=jnp.asarray(deriv[0]),deriv=jnp.asarray(deriv),
               coords=np.asarray(coords),weights=np.asarray(weights),mol=mol,eri_reference=eri_reference,
               c=c,da=da,db=db,dspin=np.block([[da,z],[z,db]]).astype(complex),na=na,nb=nb)
        for method,xc in cases(charge):
            if args.methods and method not in args.methods:continue
            row=dict(molecule=geo['name'],charge=charge,spin=spin,method=method,xc=xc)
            print('Start',row,flush=True)
            try:
                begin=time.perf_counter();g,dg=run_gradscf(method,xc,p,args.max_cycle)
                row.update(gradscf=g,gradscf_seconds=time.perf_counter()-begin)
                print('GradSCF',g,flush=True)
                begin=time.perf_counter();r,dr=run_pyscf(method,xc,p,args.max_cycle,args.grid_level)
                error=abs(g['energy']-r['energy']);tol=1e-8 if xc=='hf' else 1e-6
                dm=dg.sum(axis=0) if dg.ndim==3 else dg
                metric=np.kron(np.eye(2),np.asarray(s)) if dm.shape[0]==2*top.nao else np.asarray(s)
                electrons=float(np.trace(dm@metric).real)
                residual=g.get('orbital_gradient_norm',g.get('raw_commutator_max',np.inf))
                passed=(g['converged'] and r['converged'] and error<tol and abs(electrons-mol.nelectron)<1e-6 and residual<1e-6)
                row.update(pyscf=r,pyscf_seconds=time.perf_counter()-begin,energy_error=error,
                           electrons=electrons,status='pass' if passed else 'fail')
            except Exception:
                row.update(status='error',traceback=traceback.format_exc())
            with (out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('Result',row,flush=True)
            # Discard executable caches between spin/XC variants, retain ERIs.
            jax.clear_caches();gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('geometry');p.add_argument('--basis',default='def2-tzvp')
    p.add_argument('--output',required=True);p.add_argument('--grid-level',type=int,default=0)
    p.add_argument('--max-cycle',type=int,default=200);p.add_argument('--charges',nargs='+',type=int,default=[0,1])
    p.add_argument('--methods',nargs='+')
    run(p.parse_args())
