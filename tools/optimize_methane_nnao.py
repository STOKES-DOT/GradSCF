"""Fixed-geometry molecular RHF minimization of MACE basis parameters.

Uses native fixed primitive integrals and a stationary first-order RHF
Lagrangian (including Pulay overlap response). This experiment is not a
higher-order SCF differentiation interface or a transferable trained model.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_ENABLE_X64','1')
import jax
import jax.numpy as jnp
import numpy as np
from gradscf import integrals, scf
from gradscf.integrals.contraction import primitive_basis, contraction_matrix, contract_integrals
from gradscf.scf.rks import RKSConfig,run_rks_from_integrals_traceable
from nnao import prepare_basis


class MethaneRHF:
    """RHF experiment, defaulting to methane; geometry may specify another molecule."""
    def __init__(self,bond=1.09,basis_family="szp442_direct",core_primitives=None,
                 geometry=None,max_primitive_eri_gib=2.):
        if core_primitives is not None and basis_family not in {'szp3_direct','szp442_direct','szp663_direct'}:
            raise ValueError('core_primitives applies only to all-electron direct basis families.')
        if not np.isfinite(bond) or bond<=0:raise ValueError('bond must be positive.')
        self.bond=float(bond)
        self.coords=np.vstack((np.zeros((1,3)),bond/np.sqrt(3)*np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])))
        self.symbols=('C','H','H','H','H')
        self.molecule='CH4'
        if geometry is not None:
            if geometry.get('charge',0)!=0 or geometry.get('spin',0)!=0:
                raise ValueError('This RHF experiment currently requires a neutral closed-shell molecule.')
            self.symbols=tuple(geometry['symbols']);self.coords=np.asarray(geometry['coords_angstrom'],dtype=float)
            if self.coords.shape!=(len(self.symbols),3) or not np.isfinite(self.coords).all():
                raise ValueError('Invalid molecular geometry.')
            self.molecule=geometry['name'];self.bond=None
        if basis_family=='qvszps':
            from nnao import prepare_grimme_basis
            self.layout=prepare_grimme_basis(list(zip(self.symbols,self.coords)),unit='Angstrom')
        elif basis_family in {'szp3_direct','szp442_direct','szp663_direct'}:
            from nnao import prepare_direct_basis
            self.layout=prepare_direct_basis(list(zip(self.symbols,self.coords)),unit='Angstrom',basis_family=basis_family,core_primitives=core_primitives)
        elif basis_family=='szp3':
            self.layout=prepare_basis(list(zip(self.symbols,self.coords)),unit='Angstrom')
        else:raise ValueError('Unknown basis family')
        self.basis_family=basis_family
        self.nelectron=sum(self.layout.topology.nuclear_charges)
        if self.nelectron%2:raise ValueError('RHF requires an even electron count.')
        pt,pp=primitive_basis(self.layout.topology,self.layout.parameters)
        self.primitive_topology=pt
        self.primitive_eri_gib=8*pt.nao**4/2**30
        if not np.isfinite(max_primitive_eri_gib) or max_primitive_eri_gib<=0:
            raise ValueError('max_primitive_eri_gib must be finite and positive.')
        if self.primitive_eri_gib>max_primitive_eri_gib:
            raise MemoryError(f'Primitive ERI alone requires {self.primitive_eri_gib:.2f} GiB; '
                              f'limit is {max_primitive_eri_gib:.2f} GiB. Training peak is larger.')
        plan=integrals.make_plan(pt,backend='native')
        self.ps=plan.evaluate('overlap',pp)
        self.ph=plan.evaluate('kinetic',pp)+plan.evaluate('nuclear',pp)
        if basis_family=='qvszps':self.ph=self.ph+plan.evaluate('ecp',pp,ecps=self.layout.ecps)
        self.peri=plan.evaluate('eri',pp)
        self.peri.block_until_ready()
        self.enuc=scf.nuclear_repulsion_energy(pp.nuclear_coords,jnp.asarray(pt.nuclear_charges))
        self._value_grad=jax.jit(jax.value_and_grad(self._stationary_value,argnums=0,has_aux=True))

    def _stationary_value(self,outputs,ps,ph,peri):
        t=contraction_matrix(self.layout.topology,self.layout.bind(outputs))
        ints=contract_integrals({'overlap':ps,'hcore':ph,'eri':peri},t)
        n=ints['overlap'].shape[0]
        result=run_rks_from_integrals_traceable(overlap=ints['overlap'],hcore=ints['hcore'],eri=ints['eri'],
            nelectron=self.nelectron,nuclear_repulsion=self.enuc,ao=jnp.zeros((0,n)),
            ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0),
            config=RKSConfig(xc_spec='hf',max_cycle=150,conv_tol=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9))
        d=jax.lax.stop_gradient(result.density_matrix)
        w=jax.lax.stop_gradient((result.mo_coeff*(result.mo_occ*result.mo_energy)[None,:])@result.mo_coeff.T)
        dp=t@d@t.T
        j=jnp.einsum('pqrs,rs->pq',peri,dp)
        k=jnp.einsum('prqs,rs->pq',peri,dp)
        energy=jnp.sum(dp*ph)+.5*jnp.sum(dp*j)-.25*jnp.sum(dp*k)+self.enuc
        lagrangian=energy-jnp.sum((t@w@t.T)*ps)
        # Preserve the actual energy as the value, and only the stationary
        # Lagrangian's basis derivative as the first-order gradient.
        value=lagrangian+jax.lax.stop_gradient(result.total_energy-lagrangian)
        residual=result.fock_matrix@d@ints['overlap']-ints['overlap']@d@result.fock_matrix
        info=dict(converged=result.converged,scf_cycles=result.cycles,
                  orbital_residual=jnp.linalg.norm(residual),
                  min_overlap_eigenvalue=jnp.linalg.eigvalsh(ints['overlap'])[0],
                  reconstruction_error=jnp.abs(energy-result.total_energy))
        return value,info

    def evaluate(self,outputs):
        (energy,info),gradient=self._value_grad(jnp.asarray(outputs),self.ps,self.ph,self.peri)
        row={key:bool(value) if key=='converged' else int(value) if key=='scf_cycles' else float(value)
             for key,value in info.items()}
        if not row['converged'] or row['orbital_residual']>1e-7:
            raise RuntimeError(f'SCF is not stationary: {row}')
        if row['min_overlap_eigenvalue']<1e-8 or row['reconstruction_error']>1e-9:
            raise RuntimeError(f'Invalid basis or primitive reconstruction: {row}')
        if not np.isfinite(energy) or not np.isfinite(gradient).all():
            raise RuntimeError('Nonfinite energy or gradient')
        return float(energy),gradient,row


def embedded_outputs(layout,atom_shells):
    """Embed saved raw contractions in a larger nested primitive pool."""
    if len(atom_shells)!=len(layout.symbols):raise ValueError('Warm-start atom count differs.')
    blocks=[s for shells in atom_shells for s in shells]
    if len(blocks)!=len(layout.roles):raise ValueError('Warm-start shell count differs.')
    out=layout.reference_outputs()
    for i,(block,atom,slot,l,a,c) in enumerate(zip(blocks,layout.shell_atoms,layout.slots,
            layout.topology.angular_momenta,layout.parameters.exponents,layout.parameters.coefficients)):
        rows=np.asarray(block[1:]);n=len(rows)
        if block[0]!=l or rows.shape[1]!=1+c.shape[1] or n>len(a):raise ValueError('Warm-start shell topology differs.')
        np.testing.assert_allclose(rows[:,0],np.asarray(a[:n]),atol=0,rtol=0)
        if slot<0:
            np.testing.assert_allclose(rows[:,1:],c,atol=1e-14,rtol=0)
        else:
            vector=jnp.pad(jnp.asarray(rows[:,1]),(0,layout.max_primitives-n))
            out=out.at[atom,slot].set(vector/jnp.linalg.norm(vector))
    return out


def optimize(*,bond=1.09,maxiter=80,basis_family='szp442_direct',core_primitives=None,
             initial_summary=None,geometry=None,max_primitive_eri_gib=2.,
             output_dir=Path('artifacts/methane-nnao')):
    from flax import nnx,serialization
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import minimize
    from nnao import MACEBasisModel,build_graph
    jax.config.update('jax_enable_x64',True)
    start=time.perf_counter();output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    from gradscf.data.molecule import atomic_number
    experiment=MethaneRHF(bond,basis_family,core_primitives,geometry,max_primitive_eri_gib)
    atomic_numbers=[atomic_number(s) for s in experiment.symbols]
    model_config=dict(elements=tuple(sorted(set(atomic_numbers))),channels=8,num_interactions=2,max_ell=1,correlation=2,zero_init=True,basis_family=basis_family)
    model=MACEBasisModel(**model_config,rngs=nnx.Rngs(0))
    warm=None
    if initial_summary is not None:
        warm=json.loads(Path(initial_summary).read_text())
        if basis_family not in {'szp3_direct','szp442_direct','szp663_direct'}:
            raise ValueError('Warm start requires an all-electron direct basis family.')
        assert tuple(warm['symbols'])==experiment.symbols and not warm['cartesian'] and not warm['ecp']
        assert warm['charge']==warm['spin']==0
        np.testing.assert_allclose(warm['coords_angstrom'],experiment.coords,atol=1e-14,rtol=0)
        outputs=embedded_outputs(experiment.layout,warm['final_basis'])
        from gradscf.data.molecule import atomic_number
        bias=np.asarray(model.head_bias[...]).copy();seen={}
        for i,symbol in enumerate(experiment.symbols):
            z=atomic_number(symbol);vector=np.asarray(outputs[i]).ravel()
            if z in seen:np.testing.assert_allclose(vector,seen[z],atol=1e-10,rtol=0)
            seen[z]=vector;bias[model.elements.index(z)]=vector
        model.head_bias[...]=jnp.asarray(bias)
    graph=build_graph(atomic_numbers,experiment.coords,element_order=model.elements)
    graphdef,parameters,other=nnx.split(model,nnx.Param,...)
    initial,unravel=ravel_pytree(parameters)
    @jax.jit
    def predict(vector):
        return nnx.merge(graphdef,unravel(vector),other)(graph)
    @jax.jit
    def backward(vector,cotangent):
        return jax.vjp(predict,vector)[1](cotangent)[0]
    cache={};history=[]
    def evaluate(vector):
        key=np.asarray(vector).tobytes()
        if key not in cache:
            outputs=predict(jnp.asarray(vector))
            energy,go,info=experiment.evaluate(outputs)
            gradient=np.asarray(backward(jnp.asarray(vector),go))
            row=dict(energy_hartree=energy,max_parameter_gradient=float(np.max(np.abs(gradient))),
                     max_output_gradient=float(np.max(np.abs(go))),
                     max_abs_output=float(np.max(np.abs(np.asarray(outputs)))),**info)
            if basis_family=='szp3':row['max_tanh_output']=float(np.max(np.abs(np.tanh(np.asarray(outputs)))))
            cache[key]=(energy,gradient,row,np.asarray(outputs))
        return cache[key]
    def record(vector):
        _,_,row,_=evaluate(vector)
        row=dict(iteration=len(history),**row);history.append(row)
        (output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print(f"iter={row['iteration']:3d} E={row['energy_hartree']:.12f} "
              f"max|dE/dtheta|={row['max_parameter_gradient']:.3e} SCF={row['scf_cycles']} "
              f"Smin={row['min_overlap_eigenvalue']:.3e}",flush=True)
    def finite_difference(vector):
        energy,g,_,_=evaluate(vector)
        direction=g/np.linalg.norm(g) if np.linalg.norm(g)>1e-12 else np.ones_like(g)/np.sqrt(g.size)
        step=1e-4
        def e(scale):return evaluate(np.asarray(vector)+scale*direction)[0]
        fd=(8*(e(step)-e(-step))-e(2*step)+e(-2*step))/(12*step)
        ad=float(g@direction)
        np.testing.assert_allclose(ad,fd,atol=2e-6,rtol=2e-5)
        return dict(ad=ad,finite_difference=fd,absolute_error=abs(ad-fd),step=step)
    print(f'{experiment.molecule} RHF {basis_family}: AO={experiment.layout.topology.nao}, '
          f'primitive AO={experiment.primitive_topology.nao}, MACE parameters={initial.size}',flush=True)
    record(initial)
    if warm is not None:np.testing.assert_allclose(history[0]['energy_hartree'],warm['final_energy_hartree'],atol=1e-9,rtol=0)
    initial_fd=finite_difference(np.asarray(initial))
    print('Initial gradient check:',initial_fd,flush=True)
    result=minimize(lambda v:evaluate(v)[:2],np.asarray(initial),jac=True,method='L-BFGS-B',callback=record,
                    options=dict(maxiter=maxiter,gtol=5e-7,ftol=1e-14,maxls=30))
    final_e,final_gradient,final_row,final_outputs=evaluate(result.x)
    if history[-1]['energy_hartree']!=final_e:record(result.x)
    final_fd=finite_difference(result.x)
    def state_norm_difference(first,last):
        a=jax.tree_util.tree_leaves(nnx.to_pure_dict(first));b=jax.tree_util.tree_leaves(nnx.to_pure_dict(last))
        return float(np.sqrt(sum(float(jnp.sum((x-y)**2)) for x,y in zip(a,b))))
    final_parameters=unravel(jnp.asarray(result.x))
    summary=dict(molecule=experiment.molecule,method='RHF',basis=basis_family,cartesian=experiment.layout.topology.cart,nelectron=experiment.nelectron,
        geometry_metadata=geometry,primitive_eri_gib=experiment.primitive_eri_gib,
        initial_summary=str(initial_summary) if initial_summary is not None else None,
        initial_summary_sha256=hashlib.sha256(Path(initial_summary).read_bytes()).hexdigest() if initial_summary is not None else None,
        core_primitive_counts=[n for n,r in zip(experiment.layout.topology.primitive_counts,experiment.layout.roles) if r=='core'],
        ecp=[e.as_raw() if e else None for e in experiment.layout.ecps] if basis_family=='qvszps' else None,bond_angstrom=experiment.bond,
        symbols=experiment.symbols,coords_angstrom=experiment.coords.tolist(),charge=0,spin=0,
        optimized='all MACE and basis-head trainable parameters',fixed='geometry, primitive exponents and ECP' if basis_family=='qvszps' else 'geometry, primitive exponents, core and single-primitive polarization shells' if basis_family in {'szp442_direct','szp663_direct'} else 'geometry, primitive exponents, core and polarization shells',
        gradient='native fixed primitive integrals + JAX contraction + stationary RHF Lagrangian with Pulay term',
        optimizer='L-BFGS-B',optimizer_success=bool(result.success),message=str(result.message),
        iterations=int(result.nit),evaluations=int(result.nfev),parameter_count=int(initial.size),
        model_config=model_config,seed=0,initial_energy_hartree=history[0]['energy_hartree'],final_energy_hartree=final_e,
        energy_decrease_hartree=history[0]['energy_hartree']-final_e,initial_fd=initial_fd,final_fd=final_fd,
        backbone_parameter_change_norm=state_norm_difference(parameters.backbone,final_parameters.backbone),
        final=final_row,initial_basis=experiment.layout.atom_shells(experiment.layout.bind(predict(initial))),
        final_basis=experiment.layout.atom_shells(experiment.layout.bind(final_outputs)),
        hydrogen_output_symmetry_error=float(np.max(np.abs(final_outputs[1:]-final_outputs[1]))) if geometry is None else None,
        history=history,elapsed_seconds=time.perf_counter()-start,jax_version=jax.__version__,
        devices=[str(x) for x in jax.devices()],dtype='float64',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output_dir/'optimization.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    np.savez(output_dir/'checkpoint.npz',initial_parameters=np.asarray(initial),final_parameters=result.x,
             initial_outputs=np.asarray(predict(initial)),final_outputs=final_outputs)
    (output_dir/'parameters.msgpack').write_bytes(serialization.to_bytes(nnx.to_pure_dict(final_parameters)))
    print('Final:',result.message,'energy decrease / Ha:',summary['energy_decrease_hartree'],flush=True)
    print('Final gradient check:',final_fd,flush=True)
    print('Backbone parameter change:',summary['backbone_parameter_change_norm'],flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bond',type=float,default=1.09)
    parser.add_argument('--maxiter',type=int,default=80)
    parser.add_argument('--basis-family',choices=('szp663_direct','szp442_direct','szp3_direct','szp3','qvszps'),default='szp442_direct')
    parser.add_argument('--core-primitives',type=int,choices=(3,6),default=None)
    parser.add_argument('--initial-summary',type=Path,default=None,help='Initialize trainable output bias from saved nested contractions; backbone uses seed 0.')
    parser.add_argument('--geometry',type=Path,help='JSON with name, symbols and coords_angstrom; default is methane.')
    parser.add_argument('--max-primitive-eri-gib',type=float,default=2.,help='Allocation guard for the ERI tensor alone; training peak is larger.')
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/methane-nnao'))
    args=parser.parse_args()
    optimize(bond=args.bond,maxiter=args.maxiter,basis_family=args.basis_family,core_primitives=args.core_primitives,
             initial_summary=args.initial_summary,geometry=json.loads(args.geometry.read_text()) if args.geometry else None,
             max_primitive_eri_gib=args.max_primitive_eri_gib,output_dir=args.output_dir)
