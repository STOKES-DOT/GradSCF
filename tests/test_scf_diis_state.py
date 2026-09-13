"""DIIS ring history and RKS compatibility boundaries."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.diis import DIIS_SPACE, diis_push, diis_extrapolate
from gradscf.scf import rks


@pytest.mark.parametrize('shape', [(2, 2), (2, 2, 2)])
def test_history_wraparound_preserves_recent_entries_under_jit(shape):
    focks = jnp.zeros((DIIS_SPACE, *shape))
    errors = jnp.zeros((DIIS_SPACE, int(np.prod(shape))))
    state = focks, errors, jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32)
    push = jax.jit(diis_push)
    for i in range(DIIS_SPACE + 3):
        state = push(jnp.full(shape, float(i)), jnp.full(shape, -float(i)), *state)
    fh, eh, head, count = state
    assert int(head) == 3
    assert int(count) == DIIS_SPACE
    expected = np.concatenate([np.arange(DIIS_SPACE, DIIS_SPACE + 3), np.arange(3, DIIS_SPACE)])
    np.testing.assert_array_equal(fh.reshape(DIIS_SPACE, -1)[:, 0], expected)
    np.testing.assert_array_equal(eh[:, 0], -expected)


def test_start_cadence_and_rks_solve_patchpoint(monkeypatch):
    fock = jnp.eye(2)
    state = (jnp.zeros((DIIS_SPACE, 2, 2)), jnp.zeros((DIIS_SPACE, 4)),
             jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32))
    monkeypatch.setattr(rks, '_diis_solve', lambda *_: jnp.full((2, 2), 7.))
    first, *state = rks._diis_extrapolate(fock, fock, *state)
    np.testing.assert_array_equal(first, fock)
    second, *state = rks._diis_extrapolate(fock, -fock, *state)
    np.testing.assert_array_equal(second, jnp.full((2, 2), 7.))
    assert int(state[-1]) == 2


def test_complex_focks_real_residual_packing_preserved():
    fock = jnp.array([[1., 1.j], [-1.j, 2.]], dtype=jnp.complex128)
    state = (jnp.zeros((DIIS_SPACE, 2, 2), dtype=fock.dtype),
             jnp.zeros((DIIS_SPACE, 8), dtype=jnp.float64),
             jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32))
    error = jnp.arange(8, dtype=jnp.float64)
    first = diis_extrapolate(fock, error, *state)
    second = diis_extrapolate(2 * fock, -2 * error, *first[1:])
    np.testing.assert_allclose(second[0], (4 / 3) * fock, atol=1e-12, rtol=0)
    assert second[1].dtype == jnp.complex128
    assert second[2].dtype == jnp.float64
