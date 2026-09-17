"""3D Ewald electrostatics with a uniform compensating background."""
import itertools
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import erfc


def _integer_box(n):
    return jnp.asarray(list(itertools.product(range(-n,n+1),repeat=3)),dtype=float)


def ewald_energy(lattice,coords,charges,*,precision=1e-9,reference_lattice=None):
    # Truncation topology is fixed from the reference lattice during AD.
    ref=np.asarray(lattice if reference_lattice is None else reference_lattice)
    length=abs(np.linalg.det(ref))**(1/3)
    eta=2.0/length
    cutoff=np.sqrt(-np.log(precision*1e-2))
    nr=int(np.ceil((cutoff/eta+np.linalg.norm(ref,axis=1).sum())/np.linalg.svd(ref)[1].min()))
    reciprocal=2*np.pi*np.linalg.inv(ref).T
    ng=int(np.ceil(2*eta*cutoff/np.linalg.svd(reciprocal)[1].min()))
    lattice=jnp.asarray(lattice);coords=jnp.asarray(coords);charges=jnp.asarray(charges)
    volume=jnp.linalg.det(lattice)
    # Wrap arbitrary positions before applying the fixed image list.
    fractional=coords@jnp.linalg.inv(lattice)
    positions=(fractional-jnp.floor(fractional))@lattice
    images=_integer_box(nr)@lattice
    diff=positions[:,None,None,:]-positions[None,:,None,:]+images[None,None,:,:]
    r2=jnp.sum(diff*diff,axis=-1)
    distance=jnp.sqrt(jnp.where(r2>1e-24,r2,1.))
    real=.5*jnp.sum(charges[:,None,None]*charges[None,:,None]*
        jnp.where(r2>1e-24,erfc(eta*distance)/distance,0.))
    integers=_integer_box(ng)
    gv=integers@(2*jnp.pi*jnp.linalg.inv(lattice).T)
    g2=jnp.sum(gv*gv,axis=-1);safe=jnp.where(g2>0,g2,1.)
    structure=jnp.exp(-1j*gv@positions.T)@charges
    recip=2*jnp.pi/volume*jnp.sum(jnp.where(g2>0,jnp.exp(-g2/(4*eta*eta))/safe,0.)*jnp.abs(structure)**2)
    return real+recip-eta/jnp.sqrt(jnp.pi)*jnp.sum(charges**2)-jnp.pi*jnp.sum(charges)**2/(2*eta*eta*volume)


def coulomb_kernel(gvectors):
    g2=jnp.sum(gvectors*gvectors,axis=-1)
    return jnp.where(g2>1e-24,4*jnp.pi/jnp.where(g2>1e-24,g2,1.),0.)
