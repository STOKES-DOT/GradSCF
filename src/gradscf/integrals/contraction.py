"""JAX contraction of fixed primitive integrals, with basis normalization.

Native integrals can be evaluated outside AD when exponents/centers are fixed.
These transforms expose coefficient derivatives without a native basis JVP.
Full primitive ERIs are intended for small-system validation, not a scalable cache.
"""
from dataclasses import replace
import math
import jax
import jax.numpy as jnp
from .normalization import normalized_shell_coefficients, radial_primitive_norm


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


def primitive_basis(topology, parameters):
    """Expand each contraction into independent normalized primitives."""
    return replace(topology, contraction_counts=topology.primitive_counts), replace(
        parameters, coefficients=tuple(jnp.eye(n, dtype=c.dtype) for n,c in
            zip(topology.primitive_counts,parameters.coefficients)))


def contraction_matrix(topology, parameters):
    """Matrix T in chi_contracted = chi_primitive @ T; supports Cartesian/real spherical."""
    from .plan import _check_shapes
    _check_shapes(topology,parameters)
    angular = tuple((l+1)*(l+2)//2 if topology.cart else 2*l+1 for l in topology.angular_momenta)
    nprim = sum(n*m for n,m in zip(topology.primitive_counts,angular))
    t = jnp.zeros((nprim,topology.nao),dtype=parameters.coefficients[0].dtype)
    pi=ci=0
    for l,m,a,c in zip(topology.angular_momenta,angular,parameters.exponents,parameters.coefficients):
        normalized=normalized_shell_coefficients(l,a,c)/radial_primitive_norm(l,a)[:,None]
        block=jnp.kron(normalized,jnp.eye(m,dtype=c.dtype))
        t=t.at[pi:pi+block.shape[0],ci:ci+block.shape[1]].set(block)
        pi+=block.shape[0];ci+=block.shape[1]
    return t


def contract_integrals(tensors, transform):
    """Contract overlap/kinetic/nuclear/dipole/ERI tensors with the same T."""
    t=jnp.asarray(transform)
    result={}
    for name,value in tensors.items():
        if name=='eri':
            if value.shape != (t.shape[0],)*4:
                raise ValueError('Primitive ERI dimensions do not match transform.')
            result[name]=jnp.einsum('pqrs,pi,qj,rk,sl->ijkl',value,t,t,t,t,optimize=True)
        elif name in {'overlap','kinetic','nuclear','dipole','hcore','ecp'}:
            if value.shape[-2:] != (t.shape[0],)*2:
                raise ValueError('Primitive matrix dimensions do not match transform.')
            result[name]=jnp.einsum('pi,...pq,qj->...ij',t,value,t)
        else:
            raise ValueError(f'Unknown integral operator {name!r}.')
    return result


__all__=['primitive_basis','contraction_matrix','contract_integrals','exchange_matrix']
