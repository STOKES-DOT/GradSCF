"""Gamma-point periodic SCF using shared iterations and explicit FFTDF inputs."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from ..integrals.periodic.fft import build_inputs,get_jk
from ..scf.core import _orthogonalizer,_diagonalize_fock,_build_density_from_occ
from ..scf.uks import run_unrestricted_scf_scan,_unrestricted_xc_energy_and_potential_on_grid
from ..scf.rks import _vxc_matrix_from_grid_potential
from ..scf.energy import XCContribution,unrestricted_energy,unrestricted_fock
from ..scf.autodiff import SCFDifferentiationConfig,attach_scf_backward
from ..xc_backend.jax_libxc import hybrid_coeff,xc_type


class PeriodicSCFResult(NamedTuple):
    total_energy: object
    density_spin: object       # (2,nk,nao,nao)
    mo_coeff_spin: object
    mo_energy_spin: object
    mo_occ_spin: object
    fock_spin: object
    converged: object
    cycles: object


def gamma_energy_and_fock(density,inputs,*,xc='hf',exxdiv='ewald'):
    alpha=hybrid_coeff(xc);kind=xc_type(xc)
    j,k=get_jk(inputs,density,exxdiv=exxdiv,with_k=alpha!=0)
    ao=inputs.ao[0,0];deriv=inputs.ao[0]
    exc,va,vb,ga,gb=_unrestricted_xc_energy_and_potential_on_grid(
        ao=ao,ao_deriv1=deriv,weights=inputs.weights,density_a=density[0],density_b=density[1],
        xc_spec=xc,density_floor=1e-12,potential_clip=None,xc_kind=kind)
    potentials=jnp.stack([_vxc_matrix_from_grid_potential(ao=ao,ao_deriv1=deriv,
        ao_laplacian=jnp.zeros_like(ao),weights=inputs.weights,vxc_rho=v,vxc_grad=g,
        vxc_tau=jnp.zeros_like(v),vxc_lapl=jnp.zeros_like(v),xc_kind=kind)
        for v,g in ((va,ga),(vb,gb))])
    contribution=XCContribution(exc,potentials,alpha)
    return (unrestricted_energy(density,inputs.hcore[0],j,k,contribution,
                nuclear_repulsion=inputs.nuclear_repulsion),
            unrestricted_fock(inputs.hcore[0],j,k,contribution))


def run_gamma_scf(inputs,*,nelec,xc='hf',exxdiv='ewald',max_cycle=100,
                  conv_tol=1e-10,conv_tol_density=1e-8,conv_tol_grad=1e-7,
                  damping=0.,level_shift=0.,init_density=None,
                  differentiation=None):
    """Array-valued SCF with implicit/unrolled derivatives on fixed FFT topology."""
    if inputs.overlap.shape[0]!=1:
        raise NotImplementedError('This solver currently accepts Gamma only.')
    if xc_type(xc) not in ('HF','LDA','GGA'):
        raise NotImplementedError('Periodic XC currently supports LDA/GGA/global hybrids.')
    n=inputs.overlap.shape[-1]
    if any(v<0 or v>n for v in nelec) or sum(nelec)==0 or max_cycle<1:
        raise ValueError('Invalid occupations or max_cycle.')
    config=differentiation or SCFDifferentiationConfig(mode='unrolled')
    s=inputs.overlap[0];x=_orthogonalizer(s,1e-10)
    eps,c=_diagonalize_fock(inputs.hcore[0],x)
    occ=jnp.stack([(jnp.arange(n)<count).astype(s.dtype) for count in nelec])
    coeff=jnp.stack([c,c]);energies=jnp.stack([eps,eps])
    density=jax.vmap(_build_density_from_occ)(coeff,occ) if init_density is None else jnp.asarray(init_density)
    if density.shape!=(2,n,n):raise ValueError('Initial spin density must have shape (2,nao,nao).')
    def evaluate(d,args):return gamma_energy_and_fock(d,args,xc=xc,exxdiv=exxdiv)
    def builder(d,*_):
        e,f=evaluate(d,inputs)
        return f,f,e
    out=run_unrestricted_scf_scan(fock_builder=builder,density_spin=density,
        mo_coeff_spin=coeff,mo_occ_spin=occ,mo_energy_spin=energies,overlap=s,
        max_cycle=max_cycle,damping=damping,conv_tol=conv_tol,conv_tol_density=conv_tol_density,
        conv_tol_grad=conv_tol_grad,orthogonalization_eps=1e-10,level_shift=level_shift)
    density,coeff,energies,_,converged,cycles,*_=out
    if config.mode=='implicit':
        def residual(d,args):
            _,f=evaluate(d,args)
            transform=_orthogonalizer(args.overlap[0],1e-10)
            _,cnew=jax.vmap(lambda f:_diagonalize_fock(f,transform))(f)
            return jax.vmap(_build_density_from_occ)(cnew,occ)-d
        density=attach_scf_backward(inputs,solution=density,residual=residual,
                                    converged=converged,config=config)
        _,f=evaluate(density,inputs)
        energies,coeff=jax.vmap(lambda f:_diagonalize_fock(f,x))(f)
    energy,fock=evaluate(density,inputs)
    return PeriodicSCFResult(energy,density[:,None],coeff[:,None],energies[:,None],
                             occ[:,None],fock[:,None],converged,cycles)


class _SCF:
    unrestricted=False
    default_xc='hf'
    k_sampling=False

    def __init__(self,cell,*,kpts=None,xc=None,exxdiv='ewald',**controls):
        self.cell=cell if hasattr(cell,'topology') else cell.build()
        self.kpts=jnp.zeros((1,3)) if kpts is None else jnp.asarray(kpts).reshape(-1,3)
        if not self.k_sampling and (self.kpts.shape!=(1,3) or np.any(np.asarray(self.kpts)!=0)):
            raise NotImplementedError('First periodic SCF release supports Gamma only.')
        if not self.unrestricted and self.cell.spin!=0:
            raise ValueError('Restricted periodic SCF requires spin=0.')
        self.xc=self.default_xc if xc is None else xc
        if self.default_xc=='hf' and self.xc!='hf':
            raise ValueError('Use a periodic KS class for a DFT functional.')
        self.exxdiv=exxdiv;self.controls=controls;self.inputs=None;self.result=None

    def kernel(self,dm0=None):
        if not jax.config.x64_enabled:raise ValueError('Periodic SCF requires JAX float64.')
        if self.default_xc=='hf' and self.xc!='hf':
            raise ValueError('Use a periodic KS class for a DFT functional.')
        self.inputs=build_inputs(self.cell,kpts=self.kpts)
        density=None
        if dm0 is not None:
            d=jnp.asarray(dm0)
            density=d if self.unrestricted else jnp.stack([d/2,d/2])
        if self.kpts.shape==(1,3) and not np.any(np.asarray(self.kpts)):
            self.result=run_gamma_scf(self.inputs,nelec=self.cell.nelec,xc=self.xc,exxdiv=self.exxdiv,
                init_density=None if density is None else density[:,0],**self.controls)
        else:
            from ._kpoint import run_kpoint_scf
            self.result=run_kpoint_scf(self.inputs,mesh=self.cell.mesh,nelec=self.cell.nelec,
                xc=self.xc,exxdiv=self.exxdiv,init_density=density,**self.controls)
        self._computed_xc=self.xc;self._computed_mesh=self.cell.mesh;self._cell_version=self.cell._version
        self.e_tot=self.result.total_energy
        self.converged=bool(self.result.converged);self.cycles=int(self.result.cycles)
        self.mo_coeff=self.result.mo_coeff_spin if self.unrestricted else self.result.mo_coeff_spin[0]
        self.mo_energy=self.result.mo_energy_spin if self.unrestricted else self.result.mo_energy_spin[0]
        self.mo_occ=self.result.mo_occ_spin if self.unrestricted else self.result.mo_occ_spin.sum(axis=0)
        return self.e_tot

    def run(self,**controls):
        self.controls.update(controls);self.kernel();return self

    def get_bands(self,kpts,*,chunk_size=4):
        from .bands import get_bands
        return get_bands(self,kpts,chunk_size=chunk_size)

    def make_rdm1(self):
        if self.result is None:raise RuntimeError('Run kernel() before requesting density.')
        return self.result.density_spin if self.unrestricted else self.result.density_spin.sum(axis=0)


class RHF(_SCF):
    pass


class UHF(_SCF):
    unrestricted=True


class KRHF(RHF):
    k_sampling=True


class KUHF(UHF):
    k_sampling=True
