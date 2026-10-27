"""JAX density contractions and libcint shell-direct products."""
from functools import partial
from dataclasses import dataclass
import math
from numbers import Integral
import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import Precision
from jaxtyping import Array
from .eri import packed_eri_shape, _metadata_arrays, _pair_metadata


@dataclass(frozen=True)
class DirectJKResult:
    j: Array
    k: Array


def build_direct_jk_from_basis(basis,density,*,screening_threshold=None,workers=1):
    from ..libcint import bind_cartesian
    provider=basis if callable(getattr(basis,'get_jk',None)) else bind_cartesian(basis).direct()
    density=jnp.asarray(density)
    density=.5*(density+density.T)
    j,k=provider.get_jk(density,screening_threshold=0. if screening_threshold is None else screening_threshold,
                        workers=workers)
    return DirectJKResult(j,k)


def build_direct_jk_incremental(basis,density,*,density_last=None,j_last=None,k_last=None,
                                 screening_threshold=None,workers=1):
    if density_last is None:
        return build_direct_jk_from_basis(basis,density,screening_threshold=screening_threshold,workers=workers)
    if j_last is None or k_last is None:
        raise ValueError('Previous J/K are required with a previous density.')
    result=build_direct_jk_from_basis(basis,density-density_last,
        screening_threshold=screening_threshold,workers=workers)
    return DirectJKResult(result.j+j_last,result.k+k_last)

@jax.jit
def build_jk_from_df(df_factors: Array, density: Array) -> tuple[Array, Array]:
    """Build Coulomb and exchange matrices from DF factors.

    Uses a vmapped contraction over auxiliary channels so the J/K path stays in
    pure JAX and remains jit-friendly on CPU/GPU.
    """

    factors = jnp.asarray(df_factors)
    density = jnp.asarray(density)
    if factors.shape[0] == 0:
        zeros = jnp.zeros_like(density)
        return zeros, zeros

    projected = jnp.matmul(factors, density, precision=Precision.HIGHEST)
    rho_aux = jnp.einsum("Qpq,pq->Q", factors, density, precision=Precision.HIGHEST)
    j_mat = jnp.einsum(
        "Q,Qpq->pq",
        rho_aux,
        factors,
        precision=Precision.HIGHEST,
    )
    k_mat = jnp.einsum(
        "Qps,Qqs->pq",
        projected,
        factors,
        precision=Precision.HIGHEST,
    )
    return 0.5 * (j_mat + j_mat.T), 0.5 * (k_mat + k_mat.T)

@partial(jax.jit, static_argnames=("nocc",))
def build_jk_from_df_orbitals(
    df_factors: Array,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    *,
    nocc: int,
) -> tuple[Array, Array]:
    """Build DF J/K using occupied orbitals for the exchange contraction.

    For a density matrix assembled from orbitals, the K contraction can use
    ``B_Q C_occ`` instead of ``B_Q D``. This
    reduces the large exchange intermediate from ``naux * nao * nao`` to
    ``naux * nao * nocc`` for closed-shell RKS.
    """

    factors = jnp.asarray(df_factors)
    density = jnp.asarray(density)
    coeff = jnp.asarray(mo_coeff, dtype=factors.dtype)
    occ = jnp.asarray(mo_occ, dtype=factors.dtype)
    nocc_int = int(nocc)
    if factors.shape[0] == 0:
        zeros = jnp.zeros_like(density)
        return zeros, zeros
    if nocc_int <= 0 or nocc_int > coeff.shape[-1]:
        raise ValueError("nocc must be in the range [1, nmo].")

    rho_aux = jnp.einsum("Qpq,pq->Q", factors, density, precision=Precision.HIGHEST)
    j_mat = jnp.einsum(
        "Q,Qpq->pq",
        rho_aux,
        factors,
        precision=Precision.HIGHEST,
    )
    coeff_occ = coeff[:, :nocc_int]
    occ_scale = jnp.sqrt(jnp.maximum(occ[:nocc_int], 0.0))
    weighted_occ = coeff_occ * occ_scale[None, :]
    b_occ = jnp.einsum(
        "Qpr,ri->Qpi",
        factors,
        weighted_occ,
        precision=Precision.HIGHEST,
    )
    k_mat = jnp.einsum(
        "Qpi,Qqi->pq",
        b_occ,
        b_occ,
        precision=Precision.HIGHEST,
    )
    return 0.5 * (j_mat + j_mat.T), 0.5 * (k_mat + k_mat.T)

@jax.jit
def build_j_from_df(df_factors: Array, density: Array) -> Array:
    """Build Coulomb matrix only from DF factors.

    Useful for pure semilocal functionals where HF exchange is absent.
    """

    factors = jnp.asarray(df_factors)
    density = jnp.asarray(density)
    if factors.shape[0] == 0:
        return jnp.zeros_like(density)

    rho_aux = jnp.einsum("Qpq,pq->Q", factors, density, precision=Precision.HIGHEST)
    j_mat = jnp.einsum(
        "Q,Qpq->pq",
        rho_aux,
        factors,
        precision=Precision.HIGHEST,
    )
    return 0.5 * (j_mat + j_mat.T)


def exchange_matrix(eri, density, *, block_size=None):
    """K[...,p,q] = sum_rs (pr|qs) D[...,r,s], with bounded transposes.

    Some CPU XLA versions silently return zero for the full-tensor exchange
    transpose when the ERI exceeds 32-bit indexing. Slice its first AO axis
    before contracting; all arithmetic remains in JAX, including higher AD.
    ``block_size`` can force the bounded path on small regression inputs.
    """
    eri,density=jnp.asarray(eri),jnp.asarray(density)
    n=eri.shape[0]
    if eri.shape!=(n,)*4 or density.shape[-2:]!=(n,n):
        raise ValueError('Exchange requires (n,n,n,n) ERI and (...,n,n) density.')
    if block_size is None and math.prod(eri.shape)<2**31:
        return jnp.einsum('prqs,...rs->...pq',eri,density,precision=jax.lax.Precision.HIGHEST)
    width=max(1,min(n,(2**26)//n**3)) if block_size is None else int(block_size)
    if width<1:raise ValueError('block_size must be positive.')
    width=min(width,n)
    result=jnp.zeros(density.shape[:-2]+(n,n),dtype=jnp.result_type(eri,density))
    def contract(block):
        return jnp.einsum('prqs,...rs->...pq',block,density,precision=jax.lax.Precision.HIGHEST)
    def advance(i,out):
        block=jax.lax.dynamic_slice_in_dim(eri,i*width,width,axis=0)
        return jax.lax.dynamic_update_slice(out,contract(block),(0,)*(density.ndim-2)+(i*width,0))
    result=jax.lax.fori_loop(0,n//width,advance,result)
    start=(n//width)*width
    if start<n:result=result.at[...,start:,:].set(contract(eri[start:]))
    return result


def _packed_jk_workspace(eri, density, with_j, with_k, max_memory, block_size):
    """Plan gather tiles from extra workspace, excluding the packed inputs."""
    from gradscf._memory import _requested_workspace_bytes

    workspace = _requested_workspace_bytes(64. if max_memory is None else max_memory)
    if (isinstance(block_size, bool) or not isinstance(block_size, Integral)
            or block_size < 1):
        raise ValueError('block_size must be a positive integer.')
    n = density.shape[-1]
    pairs = n*(n+1)//2
    itemsize = np.dtype(jnp.result_type(eri, density)).itemsize
    # Results, pair metadata and the symmetrized Coulomb density are live in
    # addition to gather tiles. Promotion to complex and int64 gather indices
    # are charged even when XLA can fuse some of those buffers away.
    fixed = 2*density.size*itemsize + (n*n+3*pairs)*8
    if with_j:
        fixed += math.prod(density.shape[:-2])*pairs*itemsize
    available = workspace-fixed
    gather_itemsize = itemsize+3*8
    j_row = pairs*gather_itemsize if with_j and (eri.ndim == 1 or eri.size >= 2**31) else 0
    k_cell = n*n*gather_itemsize if with_k else 0
    if available < j_row+k_cell:
        raise ValueError('max_memory is too small for the packed J/K workspace.')
    j_width = min(128, pairs)
    if j_row:
        j_budget = available if not with_k else max(j_row, available//4)
        j_width = min(j_width, j_budget//j_row)
    k_budget = available-j_width*j_row
    q_width = min(n, k_budget//k_cell) if with_k else n
    # Never gather every AO row together. For a budget below one whole row,
    # subdivide q as well, keeping the same contraction and AD operations.
    p_width = min(int(block_size), max(1, n-1), k_budget//(q_width*k_cell)) if with_k else 1
    return int(j_width), int(p_width), int(q_width)


@partial(jax.jit, static_argnames=("with_j", "with_k", "max_memory", "block_size"))
def _build_jk_from_packed_jax(eri: Array, density: Array, *, with_j=True, with_k=True,
                             max_memory=None, block_size=1) -> tuple[Array,Array]:
    """J/K from s4 or s8 real spatial ERIs, including arbitrary complex D.

    Bounded pair/row gathers avoid expanding the packed buffer to N^4.
    Unlike a Hermitian-only shortcut, this preserves all spin-offdiagonal
    exchange blocks. The entire contraction is JAX differentiable.
    """
    eri,density=jnp.asarray(eri),jnp.asarray(density)
    n=density.shape[-1] if density.ndim>=2 else 0
    if n<1 or eri.ndim not in (1,2) or eri.shape!=packed_eri_shape(n,eri.ndim) or density.shape[-2:]!=(n,n):
        raise ValueError('Packed ERI/density shapes do not match.')
    widths=_packed_jk_workspace(eri,density,with_j,with_k,max_memory,block_size)
    zeros=jnp.zeros(density.shape,dtype=jnp.result_type(eri,density))
    if not (with_j or with_k):return zeros,zeros
    rows,cols,pair,_=_pair_metadata(n)
    rows,cols,pair=jnp.asarray(rows),jnp.asarray(cols),jnp.asarray(pair,dtype=jnp.int64)
    npair=len(rows)
    def fetch(a,b):
        if eri.ndim==2:return eri[a,b]
        hi=jnp.maximum(a,b);lo=jnp.minimum(a,b)
        return eri[hi*(hi+1)//2+lo]
    j=zeros
    if with_j:
        d=density[...,rows,cols]+jnp.where(rows!=cols,density[...,cols,rows],0.)
        if eri.ndim==2 and eri.size<2**31:
            jp=jnp.einsum('ab,...b->...a',eri,d,precision=Precision.HIGHEST)
        else:
            width=widths[0];ids=jnp.arange(npair,dtype=jnp.int64)
            jp=jnp.zeros(d.shape,dtype=zeros.dtype)
            @jax.checkpoint
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
        j=j.at[...,rows,cols].set(jp);j=j.at[...,cols,rows].set(jp)
    k=zeros
    if with_k:
        p_width,q_width=widths[1:]
        def contract_rows(start,count,out):
            pr=jax.lax.dynamic_slice_in_dim(pair,start,count,axis=0)
            # Reverse scans recompute the gather instead of saving every
            # tile's indices/values, which would recover N^4 workspace.
            @partial(jax.checkpoint,static_argnums=(1,))
            def tile(q_start,q_count,value):
                qs=jax.lax.dynamic_slice_in_dim(pair,q_start,q_count,axis=0)
                block=fetch(pr[:,None,:,None],qs[None,:,None,:])
                result=jnp.einsum('pqrs,...rs->...pq',block,density,precision=Precision.HIGHEST)
                return jax.lax.dynamic_update_slice(value,result,(0,)*(density.ndim-2)+(start,q_start))
            out=jax.lax.fori_loop(0,n//q_width,lambda q,value:tile(q*q_width,q_width,value),out)
            tail=(n//q_width)*q_width
            return tile(tail,n-tail,out) if tail<n else out
        k=jax.lax.fori_loop(0,n//p_width,lambda p,out:contract_rows(p*p_width,p_width,out),k)
        tail=(n//p_width)*p_width
        if tail<n:k=contract_rows(tail,n-tail,k)
    return j,k

@partial(jax.jit, static_argnames=("with_j", "with_k", "max_memory", "block_size"))
def build_jk_from_packed(eri: Array, density: Array, *, with_j=True, with_k=True,
                         max_memory=None, block_size=None) -> tuple[Array,Array]:
    """Consume s4/s8 ERIs and arbitrary ``(..., n, n)`` complex densities.

    Unselected operators return zeros. CPU s8 float64 uses the streaming
    native kernel; other CPU inputs use scalar rows. GPU contracts multiple
    bounded AO rows at a time. ``block_size`` overrides the AO-row ceiling
    and selects the portable path on every device, for explicit tuning.

    ``max_memory`` is extra contraction workspace in decimal MB, excluding
    the input ERI/density and the caller's other live arrays. The default is
    64 MB; 20% is reserved before charging outputs, metadata and gather tiles.
    This array-planning estimate is not a process/allocator memory limit.
    Density and ERI JVP/VJP, including geometry products, remain available.
    """
    eri,density=jnp.asarray(eri),jnp.asarray(density)
    n=density.shape[-1] if density.ndim>=2 else 0
    if n<1 or eri.ndim not in (1,2) or eri.shape!=packed_eri_shape(n,eri.ndim) or density.shape[-2:]!=(n,n):
        raise ValueError('Packed ERI/density shapes do not match.')
    if not all(isinstance(flag,(bool,np.bool_)) for flag in (with_j,with_k)):
        raise TypeError('with_j and with_k must be boolean.')
    # Validate static options even for an unselected pair of operators.
    from gradscf._memory import _requested_workspace_bytes
    workspace=_requested_workspace_bytes(64. if max_memory is None else max_memory)
    if block_size is not None and (isinstance(block_size,bool)
            or not isinstance(block_size,Integral) or block_size<1):
        raise ValueError('block_size must be a positive integer.')
    if not (with_j or with_k):
        zeros=jnp.zeros(density.shape,dtype=jnp.result_type(eri,density))
        return zeros,zeros
    if block_size is not None:
        return _build_jk_from_packed_jax(eri,density,with_j=with_j,with_k=with_k,
                                         max_memory=max_memory,block_size=block_size)
    def portable(width):
        return _build_jk_from_packed_jax(eri,density,with_j=with_j,with_k=with_k,
                                         max_memory=max_memory,block_size=width)
    def cpu():
        native_workspace=(8*(int(with_j)+int(with_k))+2)*density.size*np.dtype(density.dtype).itemsize
        if (eri.ndim==1 and eri.dtype==jnp.float64
                and density.dtype in (jnp.float64,jnp.complex128) and native_workspace<=workspace):
            from gradscf.integrals.backends.native.jk import packed_jk
            return packed_jk(eri,density,with_j=with_j,with_k=with_k,max_memory=max_memory)
        return portable(1)
    return jax.lax.platform_dependent(cpu=cpu,default=lambda:portable(16))

@jax.jit
def build_j_from_eri_pair_matrix(eri_pair_matrix: Array, density: Array) -> Array:
    """Build exact Coulomb matrix from an AO-pair packed ERI matrix."""

    pair = jnp.asarray(eri_pair_matrix)
    density = 0.5 * (jnp.asarray(density) + jnp.asarray(density).T)
    nao = int(density.shape[0])
    rows, cols, _, multiplicity = _metadata_arrays(nao, density.dtype)
    density_pair = density[rows, cols] * multiplicity
    j_pair = pair @ density_pair
    # Each triangle has unique indices. Halve the diagonal before adding both
    # triangles: equivalent to assignment, with a Taylor-compatible primitive.
    values = j_pair*jnp.where(rows==cols, .5, 1.)
    j_mat = jnp.zeros_like(density)
    j_mat = j_mat.at[rows, cols].add(values)
    j_mat = j_mat.at[cols, rows].add(values)
    return 0.5 * (j_mat + j_mat.T)

@jax.jit
def build_jk_from_eri_pair_matrix(eri_pair_matrix: Array, density: Array) -> tuple[Array, Array]:
    """Build exact Coulomb and exchange matrices from packed no-DF ERIs.

    ``eri_pair_matrix`` follows PySCF's ``aosym='s4'`` lower-triangle AO-pair
    layout: ``M[pair(p,q), pair(r,s)] = (pq|rs)``.
    """

    return build_jk_from_packed(eri_pair_matrix,density)
