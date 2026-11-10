"""Cached windows preserve exact bounds, tails and array derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.integrals.grids.blocks import _CachedAOGrid


WEIGHTS = np.linspace(.1,.3,11)

def grid(values):
    return _CachedAOGrid(jnp.arange(33,dtype=values.dtype).reshape(11,3),jnp.asarray(WEIGHTS,dtype=values.dtype),values,3)


@pytest.mark.parametrize('start,size',[(-2,3),(0,3),(4,3),(9,3),(12,3),(0,17)])
@pytest.mark.parametrize('deriv',[0,1,2])
def test_cached_windows_and_array_ad_match_independent_index_oracle(start,size,deriv):
    values=jnp.sin(jnp.arange(165,dtype=jnp.float64).reshape(11,5,3))
    direction=jnp.cos(values)
    def actual(v):
        ao,da,lap,_,w=grid(v).block(jnp.asarray(start),size,deriv=deriv)
        return (ao,da,lap,w)
    def expected(v):
        rows=jnp.stack([v[i] if 0<=i<11 else jnp.zeros((5,3)) for i in range(start,start+size)])
        w=jnp.stack([jnp.asarray(WEIGHTS,dtype=v.dtype)[i] if 0<=i<11 else jnp.asarray(0.) for i in range(start,start+size)])
        return rows[:,0],rows[:,:4].transpose(1,0,2) if deriv else None,rows[:,4] if deriv==2 else None,w
    for a,b in zip(jax.tree.leaves(jax.jit(actual)(values)),jax.tree.leaves(expected(values))):
        np.testing.assert_array_equal(a,b)
    def loss(fn,v):return sum(jnp.sum(a*a) for a in jax.tree.leaves(fn(v)))
    for fn in (lambda f:jax.jvp(f,(values,),(direction,)),lambda f:jax.jvp(jax.grad(lambda v:loss(f,v)),(values,),(direction,))):
        a,b=fn(actual),fn(expected)
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_allclose(x,y,atol=1e-12,rtol=1e-12)
