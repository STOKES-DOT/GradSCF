"""AO-pair layouts and explicit Cartesian-reference ERI entry points."""
from functools import lru_cache
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array

__all__ = ['packed_eri_shape','eri_element','eri_tensor','eri_pair_matrix_packed']


def eri_tensor(basis):
    from ..libcint import bind_cartesian
    return bind_cartesian(basis).evaluate('eri')


def eri_pair_matrix_packed(basis):
    from ..libcint import bind_cartesian
    return bind_cartesian(basis).evaluate('eri',aosym='s4')


def eri_element(basis,i,j,k,l):
    from ..libcint import bind_cartesian
    binding=bind_cartesian(basis)
    sizes=[(a+1)*(a+2)//2*n for a,n in zip(binding.plan.topology.angular_momenta,
                                         binding.plan.topology.contraction_counts)]
    offsets=np.r_[0,np.cumsum(sizes)]
    indices=[binding.selection[x] for x in (i,j,k,l)]
    shells=[int(np.searchsorted(offsets,x,side='right')-1) for x in indices]
    bounds=tuple(v for s in shells for v in (s,s+1))
    block=binding.evaluate('eri',shell_slice=bounds)
    return block[tuple(x-offsets[s] for x,s in zip(indices,shells))]


def packed_eri_shape(nao: int, ndim: int):
    npair=nao*(nao+1)//2
    return (npair*(npair+1)//2,) if ndim==1 else (npair,npair)


@lru_cache(maxsize=64)
def _pair_metadata(nao: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.tril_indices(int(nao))
    pair_index = np.empty((int(nao), int(nao)), dtype=np.int32)
    pair_ids = np.arange(rows.size, dtype=np.int32)
    pair_index[rows, cols] = pair_ids
    pair_index[cols, rows] = pair_ids
    multiplicity = np.where(rows == cols, 1.0, 2.0)
    return (
        rows.astype(np.int32),
        cols.astype(np.int32),
        pair_index,
        multiplicity.astype(np.float64),
    )


def _metadata_arrays(nao: int, dtype) -> tuple[Array, Array, Array, Array]:
    rows, cols, pair_index, multiplicity = _pair_metadata(int(nao))
    return (
        jnp.asarray(rows, dtype=jnp.int32),
        jnp.asarray(cols, dtype=jnp.int32),
        jnp.asarray(pair_index, dtype=jnp.int32),
        jnp.asarray(multiplicity, dtype=dtype),
    )
