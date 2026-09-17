"""Molecular MACE graphs; neighbor topology is prepared outside autodiff.

Positions exposed to MACE are Angstrom. Rebuild neighbors after geometry
changes exceeding the skin; with_positions preserves gradients on fixed edges.
"""
from collections.abc import Mapping
from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.data.molecule import ANGSTROM_TO_BOHR


def _scale(unit):
    name=unit.lower()
    if name.startswith('angs'):return 1.
    if name.startswith('bohr'):return 1./ANGSTROM_TO_BOHR
    raise ValueError('unit must be Angstrom or Bohr.')


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MolecularGraph(Mapping):
    data: dict
    element_order: tuple[int,...]
    cutoff: float
    skin: float

    def __getitem__(self,key):return self.data[key]
    def __iter__(self):return iter(self.data)
    def __len__(self):return len(self.data)
    def tree_flatten(self):return (self.data,),(self.element_order,self.cutoff,self.skin)
    @classmethod
    def tree_unflatten(cls,aux,children):return cls(children[0],*aux)

    def with_positions(self,positions,*,unit='Angstrom'):
        r=jnp.asarray(positions,dtype=self.data['positions'].dtype)*_scale(unit)
        if r.shape!=self.data['positions'].shape:raise ValueError('Position shape changed; rebuild graph.')
        return replace(self,data={**self.data,'positions':r})


def build_graph(atomic_numbers,positions,*,element_order,cutoff=5.,skin=.5,
                unit='Angstrom',batch=None,total_charge=None,spin=None):
    """Build a host neighbor list for one molecule or contiguous disconnected batches.

    There are no padding atoms in this API. Element order must match the model.
    The initial host neighbor search is quadratic; model evaluation uses edges.
    """
    z=np.asarray(atomic_numbers);r=np.asarray(positions,dtype=float)*_scale(unit)
    order=tuple(element_order)
    if not order or len(set(order))!=len(order) or any(int(e)!=e or e<1 or e>54 for e in order):
        raise ValueError('Invalid element_order; use unique atomic numbers in 1..54.')
    if z.ndim!=1 or not len(z) or r.shape!=(len(z),3) or not np.isfinite(r).all():
        raise ValueError('Use atomic_numbers [natom] and finite positions [natom,3].')
    if any(e not in order for e in z):raise ValueError('Graph contains an unsupported element.')
    if not np.isfinite(cutoff) or cutoff<=0 or not np.isfinite(skin) or skin<0:
        raise ValueError('cutoff must be positive and skin nonnegative.')
    b=np.zeros(len(z),dtype=int) if batch is None else np.asarray(batch)
    if b.shape!=z.shape or b.dtype.kind not in 'iu' or b[0]!=0 or np.any(np.diff(b)<0) or np.any(np.diff(b)>1):
        raise ValueError('batch must be contiguous sorted graph indices starting at zero.')
    ng=int(b[-1])+1
    diff=r[:,None,:]-r[None,:,:];distance=np.linalg.norm(diff,axis=-1)
    same=b[:,None]==b[None,:];offdiag=~np.eye(len(z),dtype=bool)
    if np.any(same & offdiag & (distance<1e-8)):raise ValueError('Atoms are coincident within a graph.')
    send,recv=np.where(same & offdiag & (distance<cutoff+skin))
    species=np.asarray([order.index(int(e)) for e in z],dtype=np.int32)
    def states(value,name):
        values=np.zeros(ng) if value is None else np.atleast_1d(np.asarray(value,dtype=float))
        if values.shape!=(ng,) or not np.isfinite(values).all():raise ValueError(f'{name} must have one finite value per graph.')
        return jnp.asarray(values)
    data=dict(positions=jnp.asarray(r),node_attrs=jnp.eye(len(order))[species],
              node_type=jnp.asarray(species),batch=jnp.asarray(b,dtype=jnp.int32),
              ptr=jnp.asarray(np.r_[0,np.cumsum(np.bincount(b))],dtype=jnp.int32),
              edge_index=jnp.asarray(np.stack((send,recv)),dtype=jnp.int32),
              shifts=jnp.zeros((len(send),3)),unit_shifts=jnp.zeros((len(send),3)),
              cell=jnp.zeros((ng,3,3)),total_charge=states(total_charge,'total_charge'),spin=states(spin,'spin'))
    return MolecularGraph(data,tuple(map(int,order)),float(cutoff),float(skin))


__all__=['MolecularGraph','build_graph']
