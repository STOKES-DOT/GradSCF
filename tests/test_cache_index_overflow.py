import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.integrals.grids.blocks import _CachedAOGrid

@pytest.mark.parametrize('dtype',[jnp.int32,jnp.int64])
def test_far_outside_signed_index_never_wraps_into_interior(dtype):
    values=jnp.arange(165,dtype=jnp.float64).reshape(11,5,3)+1
    grid=_CachedAOGrid(jnp.ones((11,3)),jnp.ones(11),values,3)
    result=jax.jit(lambda start:grid.block(start,3,deriv=2))(jnp.asarray(jnp.iinfo(dtype).max,dtype))
    for value in result:np.testing.assert_array_equal(value,jnp.zeros_like(value))
