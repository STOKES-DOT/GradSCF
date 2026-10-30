"""Grid-density objective in a fixed real local-potential representation.

This is squared-density inversion, not the Wu-Yang potential functional or
RI/DF integral approximation. AD supplies both the state response and its HVP.
"""
import jax
import jax.numpy as jnp

from gradscf._memory import workspace_bytes


def grid_density_loss(grid, orthogonalizer, target, *, rank, max_memory, resident_bytes):
    """Cache a budget-checked AO grid, then checkpoint scalar block reductions."""
    n, g = grid.nao, grid.ngrids
    width = min(512, g)
    if not g:
        raise ValueError('Density matching requires a nonempty quadrature')
    size = orthogonalizer.dtype.itemsize
    estimate = (4*g*n+12*rank*width*n+16*rank*n*n+8*rank*rank)*size
    workspace_bytes(max_memory, resident_bytes=resident_bytes+estimate)
    ao = grid.map(lambda a, da, lap, coords, weights:a, deriv=0)@orthogonalizer
    weights = grid.weights
    if not bool(jnp.all(jnp.isfinite(weights)) & jnp.all(weights >= 0) & (jnp.sum(weights) > 0)):
        raise ValueError('Density matching requires a finite positive quadrature')
    rho = jnp.einsum('gi,cij,gj->cg', ao, target, ao)
    norm2 = jnp.sum(weights[None,:]*rho**2)
    if not bool(jnp.isfinite(norm2) & (norm2 > 0)):
        raise ValueError('Target density must have a finite positive norm')
    padding = (-g)%width
    blocks = jnp.pad(ao, ((0,padding),(0,0))).reshape(-1,width,n)
    wblocks = jnp.pad(weights, (0,padding)).reshape(-1,width)
    target_blocks = jnp.pad(rho, ((0,0),(0,padding))).reshape(rho.shape[0],-1,width).swapaxes(0,1)

    def measure(states):
        def add(total, inputs):
            a,w,reference = inputs
            difference = jnp.einsum('gi,cij,gj->cg',a,states,a)-reference
            return total+jnp.sum(w[None,:]*difference**2), None
        error = jax.lax.scan(jax.checkpoint(add), jnp.asarray(0., ao.dtype),
                             (blocks,wblocks,target_blocks))[0]
        return .5*error/norm2
    return measure
