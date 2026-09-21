"""Fixed-geometry molecular RHF minimization of MACE basis parameters.

Uses native fixed primitive integrals and implicit differentiation of the
converged RHF density fixed point. This is not a transferable trained model.
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
from gradscf.integrals.contraction import primitive_basis, contraction_matrix
from gradscf.integrals.backends.native_compact import NativeDirectBasis,ProjectedNativeDirectBasis
from gradscf.df import build_jk_from_df
from gradscf.solvers.nonlinear import ImplicitFixedPointConfig, implicit_fixed_point_solution
from gradscf.scf.core import _build_density_from_occ, _diagonalize_fock, _orthogonalizer
from gradscf.scf.rks import RKSConfig,run_rks_from_integrals_traceable
from gradscf.model.nnao import prepare_basis


class MethaneRHF:
    """RHF experiment, defaulting to methane; geometry may specify another molecule."""
    def __init__(self,bond=1.09,basis_family="szp442_direct",core_primitives=None,
                 geometry=None,jk_backend='direct',auxbasis='def2-universal-jkfit',
                 integral_cache=None):
        if jk_backend not in {'direct','df'}:raise ValueError('jk_backend must be direct or df.')
        if jk_backend=='df' and basis_family=='qvszps':raise NotImplementedError('DF experiment currently covers all-electron families.')
        self.jk_backend=jk_backend;self.auxbasis=auxbasis
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
            from gradscf.model.nnao import prepare_grimme_basis
            self.layout=prepare_grimme_basis(list(zip(self.symbols,self.coords)),unit='Angstrom')
        elif basis_family in {'szp3_direct','szp442_direct','szp663_direct'}:
            from gradscf.model.nnao import prepare_direct_basis
            self.layout=prepare_direct_basis(list(zip(self.symbols,self.coords)),unit='Angstrom',basis_family=basis_family,core_primitives=core_primitives)
        elif basis_family=='szp3':
            self.layout=prepare_basis(list(zip(self.symbols,self.coords)),unit='Angstrom')
        else:raise ValueError('Unknown basis family')
        self.basis_family=basis_family
        self.nelectron=sum(self.layout.topology.nuclear_charges)
        if self.nelectron%2:raise ValueError('RHF requires an even electron count.')
        pt,pp=primitive_basis(self.layout.topology,self.layout.parameters)
        self.primitive_topology=pt
        self.integral_signature=self._build_integral_signature(pp)
        self.primitive_direct=None;self.rep=None
        if integral_cache is not None:
            if jk_backend!='df':raise ValueError('Integral caches currently support jk_backend="df" only.')
            with np.load(integral_cache,allow_pickle=False) as cache:
                signature=str(cache['signature'].item())
                if signature!=self.integral_signature:raise ValueError('Integral cache does not match this geometry and basis.')
                self.ps=jnp.asarray(cache['overlap']);self.ph=jnp.asarray(cache['hcore'])
                self.rep=jnp.asarray(cache['df_factors'])
        else:
            plan=integrals.make_plan(pt,backend='native')
            if jk_backend=='direct':self.primitive_direct=NativeDirectBasis(plan,pp)
            self.ps=plan.evaluate('overlap',pp)
            self.ph=plan.evaluate('kinetic',pp)+plan.evaluate('nuclear',pp)
            if basis_family=='qvszps':self.ph=self.ph+plan.evaluate('ecp',pp,ecps=self.layout.ecps)
            if jk_backend=='df':
                from gradscf.integrals.density_fitting import make_auxiliary_plan
                at,ap=integrals.prepare_basis(list(zip(self.symbols,self.coords)),auxbasis,cart=self.layout.topology.cart)
                self.rep=make_auxiliary_plan(pt,at).factors(pp,ap)
        self.ps.block_until_ready();self.ph.block_until_ready()
        if self.rep is not None:self.rep.block_until_ready()
        self.enuc=scf.nuclear_repulsion_energy(pp.nuclear_coords,jnp.asarray(pt.nuclear_charges))
        self._value_grad=jax.jit(jax.value_and_grad(self._implicit_value,argnums=0,has_aux=True))

    def _build_integral_signature(self,primitive_parameters):
        payload=dict(symbols=self.symbols,coords=self.coords.tolist(),basis=self.basis_family,
                     auxbasis=self.auxbasis,cart=self.layout.topology.cart,
                     angular_momenta=self.primitive_topology.angular_momenta,
                     primitive_counts=self.primitive_topology.primitive_counts,
                     exponents=[np.asarray(x).tolist() for x in primitive_parameters.exponents])
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def write_integral_cache(self,path):
        if self.jk_backend!='df':raise ValueError('Only the DF backend has a reusable integral cache.')
        path=Path(path)
        if not str(path).endswith('.npz'):path=Path(str(path)+'.npz')
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,signature=np.asarray(self.integral_signature),overlap=np.asarray(self.ps),
                 hcore=np.asarray(self.ph),df_factors=np.asarray(self.rep))
        return path

    def _contracted_one_electron(self,outputs,ps,ph):
        t=contraction_matrix(self.layout.topology,self.layout.bind(outputs))
        s=t.T@ps@t;h=t.T@ph@t
        return t,s,h

    @staticmethod
    def _project_df_factors(t,rep):
        from gradscf.integrals.density_fitting import project_factors
        return project_factors(rep,t)

    def _jk(self,t,density,rep,df_factors):
        if self.jk_backend=='df':
            return build_jk_from_df(df_factors,density)
        return ProjectedNativeDirectBasis(self.primitive_direct,t).get_jk(density)

    def _implicit_value(self,outputs,ps,ph,rep):
        t,s,h=self._contracted_one_electron(outputs,ps,ph)
        df_factors=self._project_df_factors(t,rep) if self.jk_backend=='df' else None
        direct_basis=(ProjectedNativeDirectBasis(self.primitive_direct,t)
                      if self.jk_backend=='direct' else None)
        n=s.shape[0]
        cfg=RKSConfig(xc_spec='hf',jk_backend=self.jk_backend,max_cycle=150,
                      conv_tol=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9)
        result=run_rks_from_integrals_traceable(overlap=s,hcore=h,eri=None,df_factors=df_factors,direct_basis=direct_basis,
            nelectron=self.nelectron,nuclear_repulsion=self.enuc,ao=jnp.zeros((0,n)),
            ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0),
            config=cfg)
        mo_occ=jnp.zeros(n,dtype=s.dtype).at[:self.nelectron//2].set(2.)

        def density_fixed_point(density,outputs_local):
            t_local,s_local,h_local=self._contracted_one_electron(outputs_local,ps,ph)
            df_local=self._project_df_factors(t_local,rep) if self.jk_backend=='df' else None
            j_local,k_local=self._jk(t_local,density,rep,df_local)
            x_local=_orthogonalizer(s_local,cfg.orthogonalization_eps)
            _,coeff_local=_diagonalize_fock(h_local+j_local-.5*k_local,x_local)
            return _build_density_from_occ(coeff_local,mo_occ)

        density=implicit_fixed_point_solution(
            outputs,
            solution=result.density_matrix,
            fixed_point=density_fixed_point,
            config=ImplicitFixedPointConfig(tolerance=1e-10,max_iter=20,restart=40),
            converged=result.converged,
            require_converged=True,
        )
        j,k=self._jk(t,density,rep,df_factors)
        fock=h+j-.5*k
        energy=jnp.sum(density*h)+.5*jnp.sum(density*j)-.25*jnp.sum(density*k)+self.enuc
        residual=fock@density@s-s@density@fock
        info=dict(converged=result.converged,scf_cycles=result.cycles,
                  orbital_residual=jnp.linalg.norm(residual),
                  min_overlap_eigenvalue=jnp.linalg.eigvalsh(s)[0],
                  reconstruction_error=jnp.abs(energy-result.total_energy))
        return energy,info

    def evaluate(self,outputs):
        (energy,info),gradient=self._value_grad(jnp.asarray(outputs),self.ps,self.ph,self.rep)
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
             initial_summary=None,geometry=None,jk_backend='direct',auxbasis='def2-universal-jkfit',
             integral_cache=None,output_dir=Path('artifacts/methane-nnao')):
    from flax import nnx,serialization
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import minimize
    from gradscf.model.nnao import MACEBasisModel,build_graph
    jax.config.update('jax_enable_x64',True)
    start=time.perf_counter();output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    from gradscf.data.molecule import atomic_number
    experiment=MethaneRHF(bond,basis_family,core_primitives,geometry,jk_backend,auxbasis,integral_cache)
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
    same_hamiltonian = (warm is not None and warm.get('jk_backend')==jk_backend
                        and (jk_backend!='df' or warm.get('auxbasis')==auxbasis))
    if same_hamiltonian:
        np.testing.assert_allclose(history[0]['energy_hartree'],warm['final_energy_hartree'],atol=1e-9,rtol=0)
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
        geometry_metadata=geometry,jk_backend=jk_backend,auxbasis=auxbasis if jk_backend=='df' else None,
        integral_cache=None if integral_cache is None else str(integral_cache),
        rep_shape=None if experiment.rep is None else experiment.rep.shape,
        rep_bytes=0 if experiment.rep is None else int(experiment.rep.size*experiment.rep.dtype.itemsize),
        initial_summary=str(initial_summary) if initial_summary is not None else None,
        initial_summary_sha256=hashlib.sha256(Path(initial_summary).read_bytes()).hexdigest() if initial_summary is not None else None,
        core_primitive_counts=[n for n,r in zip(experiment.layout.topology.primitive_counts,experiment.layout.roles) if r=='core'],
        ecp=[e.as_raw() if e else None for e in experiment.layout.ecps] if basis_family=='qvszps' else None,bond_angstrom=experiment.bond,
        symbols=experiment.symbols,coords_angstrom=experiment.coords.tolist(),charge=0,spin=0,
        optimized='all MACE and basis-head trainable parameters',fixed='geometry, primitive exponents and ECP' if basis_family=='qvszps' else 'geometry, primitive exponents, core and single-primitive polarization shells' if basis_family in {'szp442_direct','szp663_direct'} else 'geometry, primitive exponents, core and polarization shells',
        gradient='native fixed primitive '+('DF factors' if jk_backend=='df' else 'shell-direct J/K')+' + JAX contraction + implicit RHF density fixed point',
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
    parser.add_argument('--jk-backend',choices=('direct','df'),default='direct')
    parser.add_argument('--auxbasis',default='def2-universal-jkfit')
    parser.add_argument('--integral-cache',type=Path,help='Load a CPU-prepared DF cache; required to keep native setup out of a GPU process.')
    parser.add_argument('--prepare-integral-cache',type=Path,help='Build a DF cache and exit without training.')
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/methane-nnao'))
    args=parser.parse_args()
    geometry=json.loads(args.geometry.read_text()) if args.geometry else None
    if args.prepare_integral_cache is not None:
        experiment=MethaneRHF(args.bond,args.basis_family,args.core_primitives,geometry,args.jk_backend,args.auxbasis)
        print(experiment.write_integral_cache(args.prepare_integral_cache),flush=True)
    else:
        optimize(bond=args.bond,maxiter=args.maxiter,basis_family=args.basis_family,core_primitives=args.core_primitives,
                 initial_summary=args.initial_summary,geometry=geometry,jk_backend=args.jk_backend,
                 auxbasis=args.auxbasis,integral_cache=args.integral_cache,output_dir=args.output_dir)
