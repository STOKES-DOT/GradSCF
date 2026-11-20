"""Declared archive occupation uncertainty never repairs the supplied density."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft.iks import Target


def test_rounded_reference_needs_an_explicit_occupation_tolerance():
    density=jnp.diag(jnp.array([2.+3e-9,2.-3e-9,0.]))
    with pytest.raises(ValueError): Target(density,jnp.eye(3),4)
    target=Target(density,jnp.eye(3),4,occupation_tolerance=1e-8)
    np.testing.assert_array_equal(target.density,density)
    assert target.valid
    assert jax.jit(lambda t:t.valid)(target)
    leaves,tree=jax.tree.flatten(target)
    restored=jax.tree.unflatten(tree,leaves)
    assert restored.occupation_tolerance==1e-8 and restored.valid


def test_occupation_tolerance_does_not_relax_number_or_hermiticity():
    with pytest.raises(ValueError):
        Target(jnp.diag(jnp.array([2.+3e-9,2.,0.])),jnp.eye(3),4,occupation_tolerance=1e-8)
    with pytest.raises(ValueError):
        Target(jnp.array([[2.,3e-9],[0.,0.]]),jnp.eye(2),2,occupation_tolerance=1e-8)


@pytest.mark.parametrize('value',[-1.,jnp.nan,jnp.inf])
def test_bad_declared_uncertainty_is_rejected(value):
    with pytest.raises(ValueError): Target(jnp.eye(2),jnp.eye(2),2,occupation_tolerance=value)
