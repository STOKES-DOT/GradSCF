"""DIIS must resolve small commutators without an absolute residual floor."""
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.rks import _PYSCF_LIKE_DIIS_SPACE, _diis_solve


@pytest.mark.parametrize("scale", [1., 1e-6, 1e-9])
@pytest.mark.parametrize("complex_fock", [False, True])
def test_diis_solution_is_invariant_to_common_error_scale(scale, complex_fock):
    dtype = jnp.complex128 if complex_fock else jnp.float64
    focks = jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, 1, 1), dtype=dtype)
    focks = focks.at[0, 0, 0].set(1.).at[1, 0, 0].set(4.)
    errors = jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, 1), dtype=jnp.float64)
    errors = errors.at[0, 0].set(scale).at[1, 0].set(-2. * scale)
    # c0+c1=1 and c0-2*c1=0 => extrapolate=2, independent of units.
    result = _diis_solve(focks, errors, jnp.asarray(2))
    np.testing.assert_allclose(result, [[2.]], atol=1e-10, rtol=0)


def test_diis_zero_residual_history_stays_finite():
    focks = jnp.ones((_PYSCF_LIKE_DIIS_SPACE, 1, 1))
    errors = jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, 1))
    np.testing.assert_allclose(_diis_solve(focks, errors, jnp.asarray(2)), [[1.]])


def test_diis_zero_residual_history_has_finite_gradient():
    import jax
    focks = jnp.arange(float(_PYSCF_LIKE_DIIS_SPACE)).reshape(-1, 1, 1)
    errors = jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, 1))
    derivative = jax.grad(lambda e: _diis_solve(focks, e, jnp.asarray(2)).sum())(errors)
    assert np.all(np.isfinite(derivative))


def test_diis_single_precision_collinear_history_stays_finite():
    focks = jnp.arange(_PYSCF_LIKE_DIIS_SPACE, dtype=jnp.float32).reshape(-1, 1, 1)
    errors = jnp.zeros((_PYSCF_LIKE_DIIS_SPACE, 1), dtype=jnp.float32)
    errors = errors.at[:3, 0].set(jnp.asarray(1., dtype=jnp.float32))
    result = _diis_solve(focks, errors, jnp.asarray(3))
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, [[1.]], atol=3e-3, rtol=0)
