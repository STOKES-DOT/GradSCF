"""Compare spherical RHF bases using GradSCF native integrals and SCF.

The NNAO coefficients are the saved single-geometry optimized coefficients;
no network fitting or PySCF execution takes place in this benchmark.
"""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_ENABLE_X64','1')
import jax
import jax.numpy as jnp
import numpy as np
from gradscf import integrals
from gradscf.scf import nuclear_repulsion_energy
from gradscf.scf.rks import RKSConfig,run_rks_from_integrals_traceable
from nnao import prepare_direct_basis

STANDARD_BASES=('sto-3g','3-21g','6-31g','6-31g(d)','6-31g(d,p)',
                'def2-svp','cc-pvdz','def2-tzvp','cc-pvtz')
CUSTOM='NNAO-core6-4s4p2d'
CUSTOM_FAMILIES={CUSTOM:'szp442_direct','NNAO-core6-6s6p3d-H6s2p':'szp663_direct'}


def prepare_case(name,trained):
    atom=list(zip(trained['symbols'],trained['coords_angstrom']))
    if name in CUSTOM_FAMILIES:
        assert trained['basis']==CUSTOM_FAMILIES[name]
        layout=prepare_direct_basis(atom,core_primitives=6,basis_family=CUSTOM_FAMILIES[name])
        blocks=[block for shells in trained['final_basis'] for block in shells]
        assert tuple(b[0] for b in blocks)==layout.topology.angular_momenta
        arrays=[jnp.asarray(b[1:],dtype=jnp.float64) for b in blocks]
        params=replace(layout.parameters,exponents=tuple(a[:,0] for a in arrays),
                       coefficients=tuple(a[:,1:] for a in arrays))
        top=layout.topology
        counts=[sum((2*l+1)*nc for owner,l,nc in zip(layout.shell_atoms,top.angular_momenta,top.contraction_counts)
                    if owner==i) for i in range(len(atom))]
    else:
        top,params=integrals.prepare_basis(atom,name,cart=False)
        counts=[integrals.prepare_basis([(s,(0.,0.,0.))],name,cart=False)[0].nao for s in trained['symbols']]
    assert sum(counts)==top.nao
    return top,params,counts


def calculate(name,trained,max_eri_gib=2.):
    started=time.perf_counter()
    top,params,counts=prepare_case(name,trained)
    if not np.isfinite(max_eri_gib) or max_eri_gib<=0:raise ValueError('max_eri_gib must be finite and positive.')
    if 8*top.nao**4/2**30>max_eri_gib:
        raise MemoryError(f'{name} contracted ERI alone requires {8*top.nao**4/2**30:.2f} GiB; limit is {max_eri_gib:.2f} GiB.')
    plan=integrals.make_plan(top,backend='native')
    overlap=plan.evaluate('overlap',params)
    hcore=plan.evaluate('kinetic',params)+plan.evaluate('nuclear',params)
    eri=plan.evaluate('eri',params);eri.block_until_ready()
    integral_seconds=time.perf_counter()-started
    enuc=nuclear_repulsion_energy(params.nuclear_coords,jnp.asarray(top.nuclear_charges))
    n=top.nao;nelectron=sum(top.nuclear_charges)-trained['charge']
    result=run_rks_from_integrals_traceable(overlap=overlap,hcore=hcore,eri=eri,nelectron=nelectron,
        nuclear_repulsion=enuc,ao=jnp.zeros((0,n)),ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0),
        config=RKSConfig(xc_spec='hf',max_cycle=200,conv_tol=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9))
    energy=float(result.total_energy)
    s=np.asarray(overlap);f=np.asarray(result.fock_matrix);d=np.asarray(result.density_matrix)
    c=np.asarray(result.mo_coeff);epsilon=np.asarray(result.mo_energy)
    residual=float(np.linalg.norm(f@d@s-s@d@f))
    eigen_residual=float(np.linalg.norm(f@c-(s@c)*epsilon[None,:]))
    smin=float(np.linalg.eigvalsh(s)[0])
    assert bool(result.converged) and residual<1e-7 and eigen_residual<1e-7,(name,residual,eigen_residual)
    assert np.isfinite(energy) and smin>1e-8
    assert f.shape==hcore.shape==(n,n) and c.shape==(n,n)
    if name in CUSTOM_FAMILIES:np.testing.assert_allclose(energy,trained['final_energy_hartree'],atol=1e-8,rtol=0)
    per_element={symbol:{count for s,count in zip(trained['symbols'],counts) if s==symbol} for symbol in set(trained['symbols'])}
    assert all(len(values)==1 for values in per_element.values())
    element_count={symbol:next(iter(values)) for symbol,values in per_element.items()}
    row=dict(basis=name,carbon_ao=element_count.get('C',0),hydrogen_ao=element_count.get('H',0),nao=n,
        fock_dimension=n,fock_elements=n*n,primitive_nao=sum((2*l+1)*p for l,p in zip(top.angular_momenta,top.primitive_counts)),
        nelectron=nelectron,occupied_orbitals=nelectron//2,virtual_orbitals=n-nelectron//2,
        energy_hartree=energy,converged=bool(result.converged),scf_cycles=int(result.cycles),
        orbital_residual=residual,generalized_eigen_residual=eigen_residual,min_overlap_eigenvalue=smin,
        integral_seconds=integral_seconds,scf_seconds=time.perf_counter()-started-integral_seconds,
        elapsed_seconds=time.perf_counter()-started)
    if 'N' in element_count:row['nitrogen_ao']=element_count['N']
    print(f'{name:22s} AO={n:3d} Fock={n}x{n} E={energy:.12f} SCF={row["scf_cycles"]} |FDS-SDF|={residual:.2e}',flush=True)
    return row


def compare(summary,output,bases=None,max_eri_gib=2.):
    summary=Path(summary);trained=json.loads(summary.read_text())
    assert trained['charge']==trained['spin']==0
    assert all(n==6 for n in trained['core_primitive_counts']) and not trained['cartesian'] and not trained['ecp']
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    custom=next(name for name,family in CUSTOM_FAMILIES.items() if family==trained['basis'])
    cases=list(bases) if bases else [*STANDARD_BASES,custom]
    results=[];started=time.perf_counter()
    for name in cases:
        results.append(calculate(name,trained,max_eri_gib))
        (output/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
        jax.clear_caches()
    if 'cc-pvtz' in cases:
        anchor=next(r['energy_hartree'] for r in results if r['basis']=='cc-pvtz')
        for r in results:r['above_cc_pvtz_millihartree']=1000*(r['energy_hartree']-anchor)
    cpuinfo=Path('/proc/cpuinfo')
    report=dict(method='RHF',backend='GradSCF native CPU integrals + JAX SCF',cartesian=False,
        molecule=trained.get('molecule','CH4'),geometry_metadata=trained.get('geometry_metadata'),
        symbols=trained['symbols'],coords_angstrom=trained['coords_angstrom'],bond_angstrom=trained.get('bond_angstrom'),
        charge=0,spin=0,dtype='float64',energy_reference='cc-pVTZ finite basis, not CBS',
        custom_basis_note='NNAO coefficients are specific to this geometry; no transferability claim.',
        settings=dict(max_cycle=200,conv_tol=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9,init_guess='hcore'),
        source_summary=str(summary),summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),jax_version=jax.__version__,platform=platform.platform(),
        cpu_model=next((line.split(':',1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith('model name')),'unknown') if cpuinfo.exists() else platform.processor(),
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        devices=[str(d) for d in jax.devices()],elapsed_seconds=time.perf_counter()-started,results=results)
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bases',nargs='+',choices=(*STANDARD_BASES,*CUSTOM_FAMILIES))
    parser.add_argument('--max-eri-gib',type=float,default=2.,help='Allocation guard for contracted ERI alone.')
    args=parser.parse_args();compare(args.summary,args.output,args.bases,args.max_eri_gib)
