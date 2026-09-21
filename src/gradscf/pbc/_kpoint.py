"""Complex k-mesh SCF; common DIIS, convergence, XC and AD infrastructure."""
import jax
import jax.numpy as jnp
from ..integrals.periodic.fft import get_kpoint_jk
from ..solvers.nonlinear.diis import diis_extrapolate,DIIS_SPACE
from ..scf.convergence import convergence_reached
from ..scf.uks import _point_unrestricted_xc_value_and_grad_kernel
from ..scf.autodiff import SCFDifferentiationConfig,attach_scf_backward
from ..dft.libxc_jax.jax_libxc import hybrid_coeff,xc_type


def energy_and_fock(density,inputs,*,mesh,xc='hf',exxdiv='ewald'):
    nk=density.shape[1];alpha=hybrid_coeff(xc);kind=xc_type(xc)
    j,k=get_kpoint_jk(inputs,density,mesh=mesh,exxdiv=exxdiv,with_k=alpha!=0)
    exc,potential=grid_xc_potential(density,inputs,xc=xc)
    vxc=project_xc_potential(inputs.ao,inputs.weights,potential,kind)
    h=inputs.hcore
    energy=jnp.einsum('skpq,kqp->',density,h+.5*j).real/nk
    energy-=.5*alpha*jnp.einsum('skpq,skqp->',density,k).real/nk
    return energy+exc+inputs.nuclear_repulsion,h+j-alpha*k+vxc


def grid_xc_potential(density,inputs,*,xc):
    """XC fields on the SCF grid, reused for non-self-consistent band queries."""
    kind=xc_type(xc);nk=density.shape[1]
    if kind=='HF':return jnp.array(0.),None
    ao=inputs.ao[:,0];deriv=inputs.ao[:,1:4]
    rho=jnp.einsum('kgp,skpq,kgq->sg',ao,density,ao.conj()).real/nk
    grad=2*jnp.einsum('kgp,skpq,kxgq->sgx',ao,density,deriv.conj()).real/nk
    variables=rho.T if kind=='LDA' else jnp.concatenate([rho.T,grad[0],grad[1]],axis=1)
    energies,potential=_point_unrestricted_xc_value_and_grad_kernel(xc,kind)(variables)
    return jnp.dot(inputs.weights,energies),potential


def project_xc_potential(ao_channels,weights,potential,kind):
    ao=ao_channels[:,0];deriv=ao_channels[:,1:4]
    if potential is None:return jnp.zeros((2,ao.shape[0],ao.shape[-1],ao.shape[-1]),dtype=ao.dtype)
    blocks=[]
    for spin in range(2):
        vr=potential[:,spin]
        block=jnp.einsum('g,kgp,kgq->kpq',weights*vr,ao.conj(),ao)
        if kind=='GGA':
            vg=potential[:,2+3*spin:5+3*spin]
            term=jnp.einsum('gx,kxgp,kgq->kpq',weights[:,None]*vg,deriv.conj(),ao)
            block=block+term+term.conj().swapaxes(-1,-2)
        blocks.append(block)
    return jnp.stack(blocks)


def run_kpoint_scf(inputs,*,mesh,nelec,xc='hf',exxdiv='ewald',max_cycle=100,
                   conv_tol=1e-10,conv_tol_density=1e-8,conv_tol_grad=1e-7,
                   damping=0.,level_shift=0.,init_density=None,differentiation=None):
    from .scf import PeriodicSCFResult
    if xc_type(xc) not in ('HF','LDA','GGA'):raise NotImplementedError('Unsupported periodic XC.')
    nk,n,_=inputs.overlap.shape
    if max_cycle<1 or any(v<0 or v>n for v in nelec):raise ValueError('Invalid occupations or cycles.')
    config=differentiation or SCFDifferentiationConfig(mode='unrolled')
    s=inputs.overlap
    def transform(s):
        return jnp.linalg.inv(jnp.linalg.cholesky(s)).conj().swapaxes(-1,-2)
    x=transform(s)
    occ=jnp.stack([jnp.broadcast_to((jnp.arange(n)<count).astype(float),(nk,n)) for count in nelec])
    def diagonalize(f,metric=x):
        orth=metric.conj().swapaxes(-1,-2)@f@metric
        eps,c=jnp.linalg.eigh(orth)
        return eps,metric@c
    def density_from(c):return jnp.einsum('skpi,ski,skqi->skpq',c,occ,c.conj())
    evaluate=lambda d,args:energy_and_fock(d,args,mesh=mesh,xc=xc,exxdiv=exxdiv)
    eps,c=diagonalize(jnp.broadcast_to(inputs.hcore,(2,nk,n,n)))
    d=density_from(c) if init_density is None else jnp.asarray(init_density,dtype=c.dtype)
    if d.shape!=(2,nk,n,n):raise ValueError('Spin density must have shape (2,nk,nao,nao).')
    e,f=evaluate(d,inputs)
    def error(f,d):return x.conj().swapaxes(-1,-2)@(f@d@s-s@d@f)@x
    def pack(a):return jnp.stack([a.real,a.imag])
    history=jnp.zeros((DIIS_SPACE,2,2,nk,n,n))
    errors=jnp.zeros((DIIS_SPACE,2*f.size))
    def step(state,index):
        def advance(state):
            d,e,f,c,eps,fh,eh,head,count,done,cycles=state
            feff,fh,eh,head,count=diis_extrapolate(pack(f),pack(error(f,d)).reshape(-1),fh,eh,head,count)
            feff=feff[0]+1j*feff[1]+level_shift*(s-s@d@s)
            eps,c=diagonalize(feff)
            dn=(1-damping)*density_from(c)+damping*d
            en,fn=evaluate(dn,inputs)
            rms=jnp.sqrt(jnp.mean(jnp.abs(dn-d)**2))
            residual=jnp.linalg.norm(error(fn,dn))/jnp.sqrt(2*nk)
            done=convergence_reached(jnp.abs(en-e),rms,residual,conv_tol=conv_tol,
                conv_tol_density=conv_tol_density,conv_tol_grad=conv_tol_grad,has_prior_cycle=index>0)
            return dn,en,fn,c,eps,fh,eh,head,count,done,index+1
        state=jax.lax.cond(state[-2],lambda s:s,advance,state)
        return state,None
    initial=(d,e,f,c,eps,history,errors,jnp.array(0),jnp.array(0),jnp.array(False),jnp.array(0))
    (d,e,f,c,eps,*rest),_=jax.lax.scan(step,initial,jnp.arange(max_cycle))
    converged,cycles=rest[-2:]
    if config.mode=='implicit':
        def residual(d,args):
            _,f=evaluate(d,args)
            _,c=diagonalize(f,transform(args.overlap))
            return density_from(c)-d
        d=attach_scf_backward(inputs,solution=d,residual=residual,config=config,converged=converged)
        e,f=evaluate(d,inputs);eps,c=diagonalize(f)
    return PeriodicSCFResult(e,d,c,eps,occ,f,converged,cycles)
