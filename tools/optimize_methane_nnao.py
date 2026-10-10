"""Fixed-geometry molecular RHF minimization of MACE basis parameters.

Uses native fixed primitive integrals and implicit differentiation of the
converged RHF density fixed point. This is not a transferable trained model.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import zipfile

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_ENABLE_X64','1')
import jax
import jax.numpy as jnp
import numpy as np
from gradscf import integrals, scf
from gradscf.integrals.basis.contraction import primitive_basis, contraction_matrix
from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis
from gradscf.integrals.molecular.jk import build_jk_from_df
from gradscf.solvers.nonlinear import ImplicitFixedPointConfig, implicit_fixed_point_solution
from gradscf.scf.core import _build_density_from_occ, _diagonalize_fock, _orthogonalizer
from gradscf.scf.rks import RKSConfig,run_rks_from_integrals_traceable
from gradscf.model.nnao import prepare_basis


def _run_scf_with_rescue(solver,kwargs,config,*,rescue_level_shift):
    """Use shifted iterations only after failure, then solve the raw problem."""
    primary=solver(**kwargs,config=config)
    unchanged=(primary,jnp.asarray(False),primary.cycles)
    if rescue_level_shift==0.:return unchanged
    def rescue(previous):
        shifted=solver(**kwargs,config=replace(config,level_shift=rescue_level_shift))
        polished=solver(**dict(kwargs,init_density=shifted.density_matrix,
            init_mo_coeff=shifted.mo_coeff,init_mo_energy=shifted.mo_energy),config=config)
        return polished,jnp.asarray(True),previous.cycles+shifted.cycles+polished.cycles
    return jax.lax.cond(primary.converged,lambda _:unchanged,rescue,primary)


def map_npz_numeric(path,name):
    """Read-only map a numeric ZIP_STORED NPY member of the original NPZ file."""
    with zipfile.ZipFile(path) as archive,Path(path).open('rb') as stream:
        member=archive.getinfo(name+'.npy')
        if member.compress_type!=zipfile.ZIP_STORED or member.flag_bits&1:
            raise ValueError('NPZ mmap requires unencrypted ZIP_STORED members; compressed arrays cannot be mapped.')
        stream.seek(member.header_offset)
        header=struct.unpack('<4s5H3I2H',stream.read(30))
        if header[0]!=b'PK\x03\x04' or header[3]!=zipfile.ZIP_STORED or header[2]&1:
            raise ValueError('Invalid ZIP_STORED local header for NPZ mmap.')
        start=member.header_offset+30+header[-2]+header[-1]
        stream.seek(start)
        version=np.lib.format.read_magic(stream)
        readers={(1,0):np.lib.format.read_array_header_1_0,
                 (2,0):np.lib.format.read_array_header_2_0}
        if version not in readers:raise ValueError(f'Unsupported NPY version for NPZ mmap: {version}.')
        shape,fortran,dtype=readers[version](stream)
        if dtype.hasobject or dtype.kind not in 'biufc':
            raise ValueError('NPZ mmap supports only numeric arrays without object dtype.')
        offset=stream.tell();size=math.prod(shape)*dtype.itemsize
        if offset+size!=start+member.file_size:
            raise ValueError('NPZ numeric payload size differs from its NPY header.')
    return np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=shape,
                     order='F' if fortran else 'C')


class NonfiniteGradientError(RuntimeError):
    """Keep the failed backward arrays alongside its already validated primal."""
    def __init__(self,outputs,gradient,energy,info):
        super().__init__('Nonfinite energy or gradient')
        self.failure_arrays=dict(basis_outputs=np.array(outputs),coefficient_gradient=np.array(gradient))
        self.failure_details=dict(energy_hartree=float(energy),scf=info)


@lru_cache(maxsize=32)
def _df_rhf_solver(nelectron,implicit_tolerance,rescue_level_shift):
    """Cache SCF/response by algorithm configuration; integrals are operands."""
    cfg=RKSConfig(xc_spec='hf',jk_backend='df',max_cycle=150,
                  conv_tol=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9)

    def value(inputs,nuclear_energy):
        s,h,factors=inputs;n=s.shape[0]
        mo_occ=jnp.zeros(n,dtype=s.dtype).at[:nelectron//2].set(2.)
        kwargs=dict(overlap=s,hcore=h,eri=None,df_factors=factors,
            nelectron=nelectron,nuclear_repulsion=nuclear_energy,
            ao=jnp.zeros((0,n)),ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0))
        result,rescued,cycles=_run_scf_with_rescue(run_rks_from_integrals_traceable,
            kwargs,cfg,rescue_level_shift=rescue_level_shift)

        def fixed_point(density,physical):
            overlap,core,df=physical
            j,k=build_jk_from_df(df,density)
            _,coeff=_diagonalize_fock(core+j-.5*k,
                                     _orthogonalizer(overlap,cfg.orthogonalization_eps))
            return _build_density_from_occ(coeff,mo_occ)

        density=implicit_fixed_point_solution(inputs,solution=result.density_matrix,
            fixed_point=fixed_point,
            config=ImplicitFixedPointConfig(tolerance=implicit_tolerance,max_iter=100,restart=40),
            converged=result.converged,require_converged=True)
        j,k=build_jk_from_df(factors,density);fock=h+j-.5*k
        energy=jnp.sum(density*h)+.5*jnp.sum(density*j)-.25*jnp.sum(density*k)+nuclear_energy
        _,coeff=_diagonalize_fock(fock,_orthogonalizer(s,cfg.orthogonalization_eps))
        info=dict(converged=result.converged,scf_cycles=result.cycles,
            rescue_used=rescued,total_scf_cycles=cycles,
            fixed_point_residual=jnp.linalg.norm(_build_density_from_occ(coeff,mo_occ)-density),
            orbital_residual=jnp.linalg.norm(fock@density@s-s@density@fock),
            min_overlap_eigenvalue=jnp.linalg.eigvalsh(s)[0],
            reconstruction_error=jnp.abs(energy-result.total_energy))
        return energy,info

    return jax.jit(value),jax.jit(jax.value_and_grad(value,argnums=0,has_aux=True))


class MethaneRHF:
    """RHF experiment, defaulting to methane; geometry may specify another molecule."""
    def __init__(self,bond=1.09,basis_family="szp442_direct",core_primitives=None,
                 geometry=None,jk_backend='direct',auxbasis='def2-universal-jkfit',
                 integral_cache=None,implicit_tolerance=1e-9,cache_storage='device',
                 scf_rescue_level_shift=0.0,*,log_exponent_scales=None,df_metric_factor=None):
        if not np.isfinite(implicit_tolerance) or implicit_tolerance<=0:
            raise ValueError('implicit_tolerance must be finite and positive.')
        self.implicit_tolerance=float(implicit_tolerance)
        if not np.isfinite(scf_rescue_level_shift) or scf_rescue_level_shift<0:
            raise ValueError('scf_rescue_level_shift must be finite and nonnegative.')
        self.scf_rescue_level_shift=float(scf_rescue_level_shift)
        if cache_storage not in {'device','host','mmap'}:
            raise ValueError('cache_storage must be device, host or mmap.')
        if cache_storage in {'host','mmap'} and (jk_backend!='df' or integral_cache is None):
            raise ValueError('Host/mmap cache_storage requires a prepared DF integral_cache.')
        self.cache_storage=cache_storage
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
        if log_exponent_scales is not None:
            if not hasattr(self.layout,'with_log_exponent_scales'):
                raise ValueError('Log exponent scales require a direct-contraction NNAO basis.')
            self.layout=self.layout.with_log_exponent_scales(log_exponent_scales)
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
                if cache_storage=='mmap':
                    self.ps=map_npz_numeric(integral_cache,'overlap')
                    self.ph=map_npz_numeric(integral_cache,'hcore')
                    self.rep=map_npz_numeric(integral_cache,'df_factors')
                else:
                    array=np.asarray if cache_storage=='host' else jnp.asarray
                    self.ps=array(cache['overlap']);self.ph=array(cache['hcore'])
                    self.rep=array(cache['df_factors'])
        else:
            plan=integrals.make_plan(pt,backend='native')
            if jk_backend=='direct':self.primitive_direct=NativeDirectBasis(plan,pp)
            self.ps=plan.evaluate('overlap',pp)
            self.ph=plan.evaluate('kinetic',pp)+plan.evaluate('nuclear',pp)
            if basis_family=='qvszps':self.ph=self.ph+plan.evaluate('ecp',pp,ecps=self.layout.ecps)
            if jk_backend=='df':
                from gradscf.integrals.molecular.density_fitting import make_auxiliary_plan
                at,ap=integrals.prepare_basis(list(zip(self.symbols,self.coords)),auxbasis,cart=self.layout.topology.cart)
                kwargs={} if df_metric_factor is None else {'metric_factor':df_metric_factor}
                self.rep=make_auxiliary_plan(pt,at).factors(pp,ap,**kwargs)
        for array in (self.ps,self.ph,self.rep):
            if hasattr(array,'block_until_ready'):array.block_until_ready()
        self.enuc=scf.nuclear_repulsion_energy(pp.nuclear_coords,jnp.asarray(pt.nuclear_charges))
        if self.jk_backend=='df':
            self._projection=jax.jit(self._df_inputs)
            self._df_value,self._df_value_grad=_df_rhf_solver(
                self.nelectron,self.implicit_tolerance,self.scf_rescue_level_shift)
            self._value=self._staged_df_value
            self._value_grad=self._staged_df_value_grad
        else:
            self._value=jax.jit(self._implicit_value)
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
        from gradscf.integrals.molecular.density_fitting import project_factors
        return project_factors(rep,t)

    def _df_inputs(self,outputs,ps,ph,rep):
        t,s,h=self._contracted_one_electron(outputs,ps,ph)
        return s,h,self._project_df_factors(t,rep)

    def _staged_df_value(self,outputs,ps,ph,rep):
        return self._df_value(self._projection(outputs,ps,ph,rep),self.enuc)

    def _staged_df_value_grad(self,outputs,ps,ph,rep):
        physical,pullback=jax.vjp(lambda c:self._projection(c,ps,ph,rep),outputs)
        value,cotangent=self._df_value_grad(physical,self.enuc)
        return value,pullback(cotangent)[0]

    def _jk(self,t,density,rep,df_factors):
        if self.jk_backend=='df':
            return build_jk_from_df(df_factors,density)
        return ProjectedNativeDirectBasis(self.primitive_direct,t).get_jk(density)

    def _implicit_value(self,outputs,ps,ph,rep):
        if self.jk_backend=='df':
            return self._df_value(self._df_inputs(outputs,ps,ph,rep),self.enuc)
        t,s,h=self._contracted_one_electron(outputs,ps,ph)
        df_factors=self._project_df_factors(t,rep) if self.jk_backend=='df' else None
        direct_basis=(ProjectedNativeDirectBasis(self.primitive_direct,t)
                      if self.jk_backend=='direct' else None)
        n=s.shape[0]
        cfg=RKSConfig(xc_spec='hf',jk_backend=self.jk_backend,max_cycle=150,
                      conv_tol=1e-12,conv_tol_density=1e-10,conv_tol_grad=1e-9)
        solve_kwargs=dict(overlap=s,hcore=h,eri=None,df_factors=df_factors,direct_basis=direct_basis,
            nelectron=self.nelectron,nuclear_repulsion=self.enuc,ao=jnp.zeros((0,n)),
            ao_deriv1=jnp.zeros((4,0,n)),grid_weights=jnp.zeros(0))
        result,rescue_used,total_cycles=_run_scf_with_rescue(run_rks_from_integrals_traceable,
            solve_kwargs,cfg,rescue_level_shift=self.scf_rescue_level_shift)
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
            config=ImplicitFixedPointConfig(tolerance=self.implicit_tolerance,max_iter=100,restart=40),
            converged=result.converged,
            require_converged=True,
        )
        j,k=self._jk(t,density,rep,df_factors)
        fock=h+j-.5*k
        energy=jnp.sum(density*h)+.5*jnp.sum(density*j)-.25*jnp.sum(density*k)+self.enuc
        residual=fock@density@s-s@density@fock
        _,raw_coeff=_diagonalize_fock(fock,_orthogonalizer(s,cfg.orthogonalization_eps))
        fixed_point_residual=jnp.linalg.norm(_build_density_from_occ(raw_coeff,mo_occ)-density)
        info=dict(converged=result.converged,scf_cycles=result.cycles,
                  rescue_used=rescue_used,total_scf_cycles=total_cycles,
                  fixed_point_residual=fixed_point_residual,
                  orbital_residual=jnp.linalg.norm(residual),
                  min_overlap_eigenvalue=jnp.linalg.eigvalsh(s)[0],
                  reconstruction_error=jnp.abs(energy-result.total_energy))
        return energy,info

    @staticmethod
    def _checked_info(energy,info):
        row={key:bool(value) if key in {'converged','rescue_used'} else
             int(value) if key in {'scf_cycles','total_scf_cycles'} else float(value)
             for key,value in info.items()}
        if not np.isfinite(energy) or not np.isfinite([
            row['orbital_residual'],row['min_overlap_eigenvalue'],row['reconstruction_error'],
            row['fixed_point_residual']
        ]).all():
            raise RuntimeError('Nonfinite energy or SCF diagnostics')
        if not row['converged'] or row['orbital_residual']>1e-7:
            raise RuntimeError(f'SCF is not stationary: {row}')
        if row['fixed_point_residual']>1e-7:
            raise RuntimeError(f'SCF density is not a raw Aufbau fixed point: {row}')
        if row['min_overlap_eigenvalue']<1e-8 or row['reconstruction_error']>1e-9:
            raise RuntimeError(f'Invalid basis or primitive reconstruction: {row}')
        return row

    def evaluate_value(self,outputs):
        """Reconverge SCF and validate its energy without executing backward."""
        energy,info=self._value(jnp.asarray(outputs),self.ps,self.ph,self.rep)
        return float(energy),self._checked_info(energy,info)

    def evaluate(self,outputs):
        (energy,info),gradient=self._value_grad(jnp.asarray(outputs),self.ps,self.ph,self.rep)
        row=self._checked_info(energy,info)
        if not np.isfinite(gradient).all():
            raise NonfiniteGradientError(outputs,gradient,energy,row)
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


def _run_adam_steps(initial,evaluate,record,*,steps,initial_learning_rate,
                    final_learning_rate,gradient_clip_norm,max_retries=12):
    """Run accepted Adam steps, shrinking the learning rate at invalid trials."""
    import optax
    if steps<1:raise ValueError('Adam steps must be positive.')
    if min(initial_learning_rate,final_learning_rate,gradient_clip_norm)<=0:
        raise ValueError('Adam learning rates and gradient clip norm must be positive.')
    if max_retries<1:raise ValueError('Adam max_retries must be positive.')
    vector=jnp.asarray(initial)
    transform=optax.chain(optax.clip_by_global_norm(float(gradient_clip_norm)),optax.scale_by_adam())
    state=transform.init(vector)
    energy,gradient,_,_=evaluate(np.asarray(vector));evaluations=1;rejected_total=0
    record(np.asarray(vector),learning_rate=0.,rejected_trials=0)
    multiplier=1.
    for step in range(steps):
        updates,next_state=transform.update(jnp.asarray(gradient),state,vector)
        fraction=step/max(steps-1,1)
        base_lr=initial_learning_rate*(final_learning_rate/initial_learning_rate)**fraction
        last_error=None
        for retry in range(max_retries):
            learning_rate=base_lr*multiplier
            candidate=vector-learning_rate*updates
            try:
                candidate_result=evaluate(np.asarray(candidate));evaluations+=1
            except RuntimeError as error:
                if str(error)!='Nonfinite energy or gradient':raise
                evaluations+=1;rejected_total+=1;multiplier*=.5;last_error=error
                continue
            vector=candidate;state=next_state
            energy,gradient=candidate_result[:2]
            record(np.asarray(vector),learning_rate=float(learning_rate),rejected_trials=retry)
            break
        else:
            raise RuntimeError('Adam failed to find a finite trial point.') from last_error
    return np.asarray(vector),dict(iterations=steps,evaluations=evaluations,
                                   rejected_trials=rejected_total)


def optimize(*,bond=1.09,maxiter=80,basis_family='szp442_direct',core_primitives=None,
             initial_summary=None,initial_checkpoint=None,geometry=None,jk_backend='direct',auxbasis='def2-universal-jkfit',
             integral_cache=None,optimizer='lbfgs',learning_rate=1e-4,
             final_learning_rate=1e-5,gradient_clip_norm=1.,
             output_dir=Path('artifacts/methane-nnao'),implicit_tolerance=1e-9,
             scf_rescue_level_shift=0.0):
    from flax import nnx,serialization
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import minimize
    from gradscf.model.nnao import MACEBasisModel,build_graph
    jax.config.update('jax_enable_x64',True)
    start=time.perf_counter();output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    from gradscf.data.molecule import atomic_number
    experiment=MethaneRHF(bond,basis_family,core_primitives,geometry,jk_backend,auxbasis,integral_cache,
                         implicit_tolerance=implicit_tolerance,scf_rescue_level_shift=scf_rescue_level_shift)
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
    if initial_checkpoint is not None:
        with np.load(initial_checkpoint,allow_pickle=False) as checkpoint:
            key='parameters' if 'parameters' in checkpoint else 'final_parameters'
            restored=np.asarray(checkpoint[key])
        if restored.shape!=initial.shape:raise ValueError('Checkpoint parameter shape does not match the model.')
        initial=jnp.asarray(restored,dtype=initial.dtype)
    initial_parameter_tree=unravel(initial)
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
    def record(vector,**metadata):
        _,_,row,outputs=evaluate(vector)
        row=dict(iteration=len(history),**row,**metadata);history.append(row)
        (output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        temporary=output_dir/'checkpoint-latest.tmp.npz'
        np.savez(temporary,parameters=np.asarray(vector),outputs=outputs,iteration=row['iteration'])
        os.replace(temporary,output_dir/'checkpoint-latest.npz')
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
    same_hamiltonian = (warm is not None and warm.get('jk_backend')==jk_backend
                        and (jk_backend!='df' or warm.get('auxbasis')==auxbasis))
    if optimizer=='lbfgs':
        record(initial)
        if same_hamiltonian:
            np.testing.assert_allclose(history[0]['energy_hartree'],warm['final_energy_hartree'],atol=1e-9,rtol=0)
        initial_fd=finite_difference(np.asarray(initial))
        print('Initial gradient check:',initial_fd,flush=True)
        result=minimize(lambda v:evaluate(v)[:2],np.asarray(initial),jac=True,method='L-BFGS-B',callback=record,
                        options=dict(maxiter=maxiter,gtol=5e-7,ftol=1e-14,maxls=30))
        final_vector=np.asarray(result.x);optimizer_success=bool(result.success);message=str(result.message)
        iterations=int(result.nit);evaluations=int(result.nfev);rejected_trials=0
        final_e,final_gradient,final_row,final_outputs=evaluate(final_vector)
        if history[-1]['energy_hartree']!=final_e:record(final_vector)
        final_fd=finite_difference(final_vector)
    elif optimizer=='adam':
        initial_fd=final_fd=None
        final_vector,stats=_run_adam_steps(
            np.asarray(initial),evaluate,record,steps=maxiter,
            initial_learning_rate=learning_rate,final_learning_rate=final_learning_rate,
            gradient_clip_norm=gradient_clip_norm,
        )
        final_e,final_gradient,final_row,final_outputs=evaluate(final_vector)
        optimizer_success=True;message=f'Completed {maxiter} Adam steps'
        iterations=stats['iterations'];evaluations=stats['evaluations'];rejected_trials=stats['rejected_trials']
    else:raise ValueError("optimizer must be 'lbfgs' or 'adam'.")
    def state_norm_difference(first,last):
        a=jax.tree_util.tree_leaves(nnx.to_pure_dict(first));b=jax.tree_util.tree_leaves(nnx.to_pure_dict(last))
        return float(np.sqrt(sum(float(jnp.sum((x-y)**2)) for x,y in zip(a,b))))
    final_parameters=unravel(jnp.asarray(final_vector))
    summary=dict(molecule=experiment.molecule,method='RHF',basis=basis_family,cartesian=experiment.layout.topology.cart,nelectron=experiment.nelectron,
        geometry_metadata=geometry,jk_backend=jk_backend,auxbasis=auxbasis if jk_backend=='df' else None,
        integral_cache=None if integral_cache is None else str(integral_cache),
        implicit_config=dict(tolerance=experiment.implicit_tolerance,max_iter=100,restart=40),
        scf_rescue_level_shift=experiment.scf_rescue_level_shift,
        rep_shape=None if experiment.rep is None else experiment.rep.shape,
        rep_bytes=0 if experiment.rep is None else int(experiment.rep.size*experiment.rep.dtype.itemsize),
        initial_summary=str(initial_summary) if initial_summary is not None else None,
        initial_summary_sha256=hashlib.sha256(Path(initial_summary).read_bytes()).hexdigest() if initial_summary is not None else None,
        initial_checkpoint=str(initial_checkpoint) if initial_checkpoint is not None else None,
        initial_checkpoint_sha256=hashlib.sha256(Path(initial_checkpoint).read_bytes()).hexdigest() if initial_checkpoint is not None else None,
        core_primitive_counts=[n for n,r in zip(experiment.layout.topology.primitive_counts,experiment.layout.roles) if r=='core'],
        ecp=[e.as_raw() if e else None for e in experiment.layout.ecps] if basis_family=='qvszps' else None,bond_angstrom=experiment.bond,
        symbols=experiment.symbols,coords_angstrom=experiment.coords.tolist(),charge=0,spin=0,
        optimized='all MACE and basis-head trainable parameters',fixed='geometry, primitive exponents and ECP' if basis_family=='qvszps' else 'geometry, primitive exponents, core and single-primitive polarization shells' if basis_family in {'szp442_direct','szp663_direct'} else 'geometry, primitive exponents, core and polarization shells',
        gradient='native fixed primitive '+('DF factors' if jk_backend=='df' else 'shell-direct J/K')+' + JAX contraction + implicit RHF density fixed point',
        optimizer='L-BFGS-B' if optimizer=='lbfgs' else 'Adam',optimizer_success=optimizer_success,message=message,
        learning_rate=learning_rate if optimizer=='adam' else None,
        final_learning_rate=final_learning_rate if optimizer=='adam' else None,
        gradient_clip_norm=gradient_clip_norm if optimizer=='adam' else None,
        rejected_trials=rejected_trials,iterations=iterations,evaluations=evaluations,parameter_count=int(initial.size),
        model_config=model_config,seed=0,initial_energy_hartree=history[0]['energy_hartree'],final_energy_hartree=final_e,
        energy_decrease_hartree=history[0]['energy_hartree']-final_e,initial_fd=initial_fd,final_fd=final_fd,
        backbone_parameter_change_norm=state_norm_difference(initial_parameter_tree.backbone,final_parameters.backbone),
        final=final_row,initial_basis=experiment.layout.atom_shells(experiment.layout.bind(predict(initial))),
        final_basis=experiment.layout.atom_shells(experiment.layout.bind(final_outputs)),
        hydrogen_output_symmetry_error=float(np.max(np.abs(final_outputs[1:]-final_outputs[1]))) if geometry is None else None,
        history=history,elapsed_seconds=time.perf_counter()-start,jax_version=jax.__version__,
        devices=[str(x) for x in jax.devices()],dtype='float64',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output_dir/'optimization.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    np.savez(output_dir/'checkpoint.npz',initial_parameters=np.asarray(initial),final_parameters=final_vector,
             initial_outputs=np.asarray(predict(initial)),final_outputs=final_outputs)
    (output_dir/'parameters.msgpack').write_bytes(serialization.to_bytes(nnx.to_pure_dict(final_parameters)))
    print('Final:',message,'energy decrease / Ha:',summary['energy_decrease_hartree'],flush=True)
    if final_fd is not None:print('Final gradient check:',final_fd,flush=True)
    print('Backbone parameter change:',summary['backbone_parameter_change_norm'],flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bond',type=float,default=1.09)
    parser.add_argument('--maxiter',type=int,default=80)
    parser.add_argument('--basis-family',choices=('szp663_direct','szp442_direct','szp3_direct','szp3','qvszps'),default='szp442_direct')
    parser.add_argument('--core-primitives',type=int,choices=(3,6),default=None)
    parser.add_argument('--initial-summary',type=Path,default=None,help='Initialize trainable output bias from saved nested contractions; backbone uses seed 0.')
    parser.add_argument('--initial-checkpoint',type=Path,default=None,help='Resume the complete flattened MACE parameter vector.')
    parser.add_argument('--geometry',type=Path,help='JSON with name, symbols and coords_angstrom; default is methane.')
    parser.add_argument('--jk-backend',choices=('direct','df'),default='direct')
    parser.add_argument('--auxbasis',default='def2-universal-jkfit')
    parser.add_argument('--integral-cache',type=Path,help='Load a CPU-prepared DF cache; required to keep native setup out of a GPU process.')
    parser.add_argument('--prepare-integral-cache',type=Path,help='Build a DF cache and exit without training.')
    parser.add_argument('--optimizer',choices=('lbfgs','adam'),default='lbfgs')
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--final-learning-rate',type=float,default=1e-5)
    parser.add_argument('--gradient-clip-norm',type=float,default=1.)
    parser.add_argument('--implicit-tolerance',type=float,default=1e-9)
    parser.add_argument('--scf-rescue-level-shift',type=float,default=0.0)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/methane-nnao'))
    args=parser.parse_args()
    geometry=json.loads(args.geometry.read_text()) if args.geometry else None
    if args.prepare_integral_cache is not None:
        experiment=MethaneRHF(args.bond,args.basis_family,args.core_primitives,geometry,args.jk_backend,args.auxbasis,
                             implicit_tolerance=args.implicit_tolerance,
                             scf_rescue_level_shift=args.scf_rescue_level_shift)
        print(experiment.write_integral_cache(args.prepare_integral_cache),flush=True)
    else:
        optimize(bond=args.bond,maxiter=args.maxiter,basis_family=args.basis_family,core_primitives=args.core_primitives,
                 initial_summary=args.initial_summary,initial_checkpoint=args.initial_checkpoint,
                 geometry=geometry,jk_backend=args.jk_backend,
                 auxbasis=args.auxbasis,integral_cache=args.integral_cache,optimizer=args.optimizer,
                 learning_rate=args.learning_rate,final_learning_rate=args.final_learning_rate,
                 gradient_clip_norm=args.gradient_clip_norm,output_dir=args.output_dir,
                 implicit_tolerance=args.implicit_tolerance,
                 scf_rescue_level_shift=args.scf_rescue_level_shift)
