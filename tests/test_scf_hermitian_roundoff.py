"""Generalized SCF validates Hermiticity relative to the matrix scale."""
import jax.numpy as jnp
import pytest

from gradscf.scf.gks import _validate_hermitian


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_hermitian_validation_accepts_scaled_integral_roundoff(dtype):
    # Native Cl/TZVP hcore has scale147Ha and about1.1e-13Ha asymmetry.
    matrix = jnp.asarray([[147., 2.], [2.+1.1e-13, -3.]], dtype=dtype)
    _validate_hermitian(matrix, (2, 2), "hcore")


@pytest.mark.parametrize("scale", [1e-3, 1., 147.])
def test_hermitian_validation_still_rejects_nonhermitian_input(scale):
    matrix = scale*jnp.asarray([[1., 2.], [2.+1e-7, -3.]])
    with pytest.raises(ValueError, match="finite and Hermitian"):
        _validate_hermitian(matrix, (2, 2), "hcore")


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf])
def test_hermitian_validation_rejects_nonfinite_input(value):
    matrix = jnp.eye(2).at[0, 0].set(value)
    with pytest.raises(ValueError, match="finite and Hermitian"):
        _validate_hermitian(matrix, (2, 2), "hcore")
