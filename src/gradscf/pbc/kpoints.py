"""Reciprocal coordinates use radians per Bohr; lattice vectors are rows."""
import numpy as np
import jax.numpy as jnp


def reciprocal_vectors(lattice):
    return 2*jnp.pi*jnp.linalg.inv(jnp.asarray(lattice)).T


def uniform_kpoints(lattice, mesh):
    mesh=np.asarray(mesh)
    if mesh.shape!=(3,) or np.any(mesh<1) or np.any(mesh!=mesh.astype(int)):
        raise ValueError('k mesh must contain three positive integers.')
    scaled=np.stack(np.meshgrid(*[np.arange(int(n))/n for n in mesh],indexing='ij'),axis=-1).reshape(-1,3)
    scaled=np.where(scaled>=.5,scaled-1,scaled)
    return jnp.asarray(scaled)@reciprocal_vectors(lattice)


def kmesh_shape(lattice,kpoints):
    scaled=np.asarray(kpoints)@np.linalg.inv((2*np.pi*np.linalg.inv(np.asarray(lattice)).T))
    scaled=np.mod(np.round(scaled,10),1.)
    axes=[np.unique(scaled[:,i]) for i in range(3)]
    shape=tuple(len(v) for v in axes)
    if np.prod(shape)!=len(scaled) or len(np.unique(scaled,axis=0))!=len(scaled):
        raise ValueError('Use a complete uniform Cartesian-product k mesh.')
    for values in axes:
        if not np.allclose(np.diff(np.r_[values,values[0]+1]),1/len(values),atol=1e-8):
            raise ValueError('k points must be uniformly spaced.')
    return shape
