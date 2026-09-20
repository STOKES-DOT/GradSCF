from __future__ import annotations

from functools import lru_cache, partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array


def packed_eri_shape(nao: int, ndim: int):
    npair=nao*(nao+1)//2
    return (npair*(npair+1)//2,) if ndim==1 else (npair,npair)


@jax.jit
def _build_jk_from_packed_jax(eri: Array, density: Array) -> tuple[Array,Array]:
    """J/K from s4 or s8 real spatial ERIs, including arbitrary complex D.

    Bounded pair/row gathers avoid expanding the packed buffer to N^4.
    Unlike a Hermitian-only shortcut, this preserves all spin-offdiagonal
    exchange blocks. The entire contraction is JAX differentiable.
    """
    eri,density=jnp.asarray(eri),jnp.asarray(density);n=density.shape[-1]
    if eri.ndim not in (1,2) or eri.shape!=packed_eri_shape(n,eri.ndim) or density.shape[-2:]!=(n,n):
        raise ValueError('Packed ERI/density shapes do not match.')
    rows,cols,pair,_=_pair_metadata(n)
    rows,cols,pair=jnp.asarray(rows),jnp.asarray(cols),jnp.asarray(pair,dtype=jnp.int64)
    npair=len(rows)
    def fetch(a,b):
        if eri.ndim==2:return eri[a,b]
        hi=jnp.maximum(a,b);lo=jnp.minimum(a,b)
        return eri[hi*(hi+1)//2+lo]
    d= density[...,rows,cols]+jnp.where(rows!=cols,density[...,cols,rows],0.)
    if eri.ndim==2 and eri.size<2**31:
        jp=jnp.einsum('ab,...b->...a',eri,d,precision=Precision.HIGHEST)
    else:
        width=min(128,npair);ids=jnp.arange(npair,dtype=jnp.int64)
        jp=jnp.zeros(d.shape,dtype=jnp.result_type(eri,density))
        def step(i,out):
            start=i*width
            index=start+jnp.arange(width,dtype=jnp.int64)
            values=fetch(index[:,None],ids[None,:])
            value=jnp.einsum('ab,...b->...a',values,d,precision=Precision.HIGHEST)
            return jax.lax.dynamic_update_slice(out,value,(0,)*(out.ndim-1)+(start,))
        jp=jax.lax.fori_loop(0,npair//width,step,jp)
        start=(npair//width)*width
        if start<npair:
            value=jnp.einsum('ab,...b->...a',fetch(ids[start:,None],ids[None,:]),d,precision=Precision.HIGHEST)
            jp=jp.at[...,start:].set(value)
    j=jnp.zeros(density.shape,dtype=jp.dtype)
    j=j.at[...,rows,cols].set(jp);j=j.at[...,cols,rows].set(jp)
    def k_row(p):
        block=fetch(pair[p,:][None,:,None],pair[:,None,:])
        return jnp.einsum('qrs,...rs->...q',block,density,precision=Precision.HIGHEST)
    k=jnp.moveaxis(jax.lax.map(k_row,jnp.arange(n)),0,-2)
    return j,k


@jax.jit
def build_jk_from_packed(eri: Array, density: Array) -> tuple[Array,Array]:
    """Consume s4/s8 directly; s8 uses a streaming native CPU J/K kernel.

    S4 and non-float64 inputs use the portable bounded JAX path. GPU lowering
    also uses that path. Density and ERI derivatives remain available.
    """
    eri,density=jnp.asarray(eri),jnp.asarray(density)
    if eri.ndim==1 and eri.dtype==jnp.float64 and density.dtype in (jnp.float64,jnp.complex128):
        n=density.shape[-1]
        if eri.shape!=packed_eri_shape(n,1) or density.shape[-2:]!=(n,n):raise ValueError('Packed ERI/density shapes do not match.')
        from .backends.native_compact import packed_jk
        return packed_jk(eri,density)
    return _build_jk_from_packed_jax(eri,density)


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


@jax.jit
def build_j_from_eri_pair_matrix(eri_pair_matrix: Array, density: Array) -> Array:
    """Build exact Coulomb matrix from an AO-pair packed ERI matrix."""

    pair = jnp.asarray(eri_pair_matrix)
    density = 0.5 * (jnp.asarray(density) + jnp.asarray(density).T)
    nao = int(density.shape[0])
    rows, cols, _, multiplicity = _metadata_arrays(nao, density.dtype)
    density_pair = density[rows, cols] * multiplicity
    j_pair = pair @ density_pair
    j_mat = jnp.zeros_like(density)
    j_mat = j_mat.at[rows, cols].set(j_pair)
    j_mat = j_mat.at[cols, rows].set(j_pair)
    return 0.5 * (j_mat + j_mat.T)


@jax.jit
def build_jk_from_eri_pair_matrix(eri_pair_matrix: Array, density: Array) -> tuple[Array, Array]:
    """Build exact Coulomb and exchange matrices from packed no-DF ERIs.

    ``eri_pair_matrix`` follows PySCF's ``aosym='s4'`` lower-triangle AO-pair
    layout: ``M[pair(p,q), pair(r,s)] = (pq|rs)``.
    """

    return build_jk_from_packed(eri_pair_matrix,density)


def _mo_pair_products(left: Array, right: Array, rows: Array, cols: Array) -> Array:
    left_rows = left[rows]
    left_cols = left[cols]
    right_rows = right[rows]
    right_cols = right[cols]
    products = jnp.einsum(
        "Pi,Pa->iaP",
        left_rows,
        right_cols,
        precision=Precision.HIGHEST,
    )
    swapped = jnp.einsum(
        "Pi,Pa->iaP",
        left_cols,
        right_rows,
        precision=Precision.HIGHEST,
    )
    offdiag = (rows != cols).astype(products.dtype)
    return products + swapped * offdiag[None, None, :]


@partial(jax.jit, static_argnames=("nocc", "include_oovv"))
def eri_pair_matrix_to_mo_eri_slices(
    eri_pair_matrix: Array,
    mo_coeff: Array,
    *,
    nocc: int,
    include_oovv: bool = True,
) -> tuple[Array, Array, Array | None]:
    """Transform packed no-DF AO ERIs into restricted TDDFT MO slices."""

    pair = jnp.asarray(eri_pair_matrix)
    coeff = jnp.asarray(mo_coeff)
    nocc_int = int(nocc)
    rows, cols, _, _ = _metadata_arrays(int(coeff.shape[0]), coeff.dtype)
    orbo = coeff[:, :nocc_int]
    orbv = coeff[:, nocc_int:]
    ov = _mo_pair_products(orbo, orbv, rows, cols)
    vo = _mo_pair_products(orbv, orbo, rows, cols)
    eri_ovov = jnp.einsum("iaP,PQ,jbQ->iajb", ov, pair, ov, precision=Precision.HIGHEST)
    eri_ovvo = jnp.einsum("iaP,PQ,bjQ->iabj", ov, pair, vo, precision=Precision.HIGHEST)
    if not include_oovv:
        return eri_ovov, eri_ovvo, None
    oo = _mo_pair_products(orbo, orbo, rows, cols)
    vv = _mo_pair_products(orbv, orbv, rows, cols)
    eri_oovv = jnp.einsum("ijP,PQ,abQ->ijab", oo, pair, vv, precision=Precision.HIGHEST)
    return eri_ovov, eri_ovvo, eri_oovv
