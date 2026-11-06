"""Exact low-order Cartesian monomials: value/AD and compiled-operation gate."""
import re
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.integrals.grids.ao import _coordinate_power


@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64])
def test_common_angular_powers_do_not_use_generic_floating_power(dtype):
    powers=jnp.arange(5,dtype=jnp.int32)[None,:]
    base=jnp.linspace(-2.,2.,15,dtype=dtype).reshape(3,5)
    executable=jax.jit(lambda x:_coordinate_power(x,powers)).lower(base).compile()
    expected=jnp.stack([jnp.ones(3,dtype),base[:,1],base[:,2]*base[:,2],
                       base[:,3]*base[:,3]*base[:,3],(base[:,4]*base[:,4])**2],axis=1)
    np.testing.assert_allclose(executable(base),expected,atol=1e-6 if dtype==jnp.float32 else 1e-14,rtol=1e-6 if dtype==jnp.float32 else 1e-14)
    assert re.search(r'\bpower\(',executable.as_text()) is None, 'Integer s/p/d/f monomials still lower to a general floating power'


@pytest.mark.parametrize('power',range(7))
def test_integer_power_value_and_first_three_derivatives_at_zero(power):
    def reference(x):
        value=jnp.ones_like(x)
        for _ in range(power):value=value*x
        return value
    actual=lambda x:_coordinate_power(x,jnp.asarray(power,jnp.int32))
    for order in range(4):
        points=jnp.array([-1.3,0.,.7])
        np.testing.assert_allclose(jax.jit(jax.vmap(actual))(points),jax.vmap(reference)(points),atol=1e-12,rtol=1e-12)
        actual,reference=jax.grad(actual),jax.grad(reference)
