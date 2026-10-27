"""Public one-electron products from the single libcint interface."""
import jax
import jax.numpy as jnp
from ..libcint import bind_cartesian


def overlap_matrix(basis):return bind_cartesian(basis).evaluate('overlap')
def kinetic_matrix(basis):return bind_cartesian(basis).evaluate('kinetic')


def nuclear_attraction_matrix(basis,*,atom_coords=None,atom_charges=None):
    if (basis.atom_coords is None and atom_coords is None) or (basis.atom_charges is None and atom_charges is None):
        raise ValueError('Nuclear coordinates and charges are required.')
    return bind_cartesian(basis,atom_coords=atom_coords,
                          atom_charges=atom_charges).evaluate('nuclear')


def build_hcore(basis,*,atom_coords=None,atom_charges=None):
    return kinetic_matrix(basis)+nuclear_attraction_matrix(basis,atom_coords=atom_coords,atom_charges=atom_charges)


def overlap_hcore_matrices(basis,*,atom_coords=None,atom_charges=None):
    return overlap_matrix(basis),build_hcore(basis,atom_coords=atom_coords,atom_charges=atom_charges)


def dipole_matrix(basis,*,origin=None):return bind_cartesian(basis).evaluate('dipole',origin=origin)
def rinv_matrix(basis,*,origin,zeta=None):return bind_cartesian(basis).evaluate('rinv',origin=origin,zeta=zeta)


def rinv_matrices(basis,origins,*,zeta=None,grid_chunk_size=128):
    """Bound each grid tile; libcint products execute on CPU."""
    origins=jnp.asarray(origins,dtype=jnp.float64)
    if origins.ndim!=2 or origins.shape[1]!=3 or grid_chunk_size<1:
        raise ValueError('Expected grid origins (ngrid,3) and a positive tile size.')
    if not origins.shape[0]:return jnp.zeros((0,basis.nao,basis.nao),dtype=origins.dtype)
    binding=bind_cartesian(basis)
    return jax.lax.map(lambda point:binding.evaluate('rinv',origin=point,zeta=zeta),
                       origins,batch_size=min(int(grid_chunk_size),origins.shape[0]))


def overlap_element(basis,i,j):return overlap_matrix(basis)[i,j]
def kinetic_element(basis,i,j):return kinetic_matrix(basis)[i,j]
def nuclear_attraction_element(basis,i,j,**kwargs):return nuclear_attraction_matrix(basis,**kwargs)[i,j]
def dipole_element(basis,i,j,*,axis,origin=None):return dipole_matrix(basis,origin=origin)[axis,i,j]
def rinv_element(basis,i,j,*,origin,zeta=None):return rinv_matrix(basis,origin=origin,zeta=zeta)[i,j]


__all__=['overlap_matrix','kinetic_matrix','nuclear_attraction_matrix','build_hcore',
         'overlap_hcore_matrices','dipole_matrix','rinv_matrix','rinv_matrices',
         'overlap_element','kinetic_element','nuclear_attraction_element','dipole_element','rinv_element']
