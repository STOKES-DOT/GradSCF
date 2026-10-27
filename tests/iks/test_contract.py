"""Physical AO target contracts for inverse RKS, UKS and spinor GKS."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft


def api():
    assert hasattr(dft, 'iks'), 'Missing public gradscf.dft.iks infrastructure'
    return dft.iks


@pytest.mark.parametrize('layout,nelec,density', [
    ('rks', 2, [[1.2, 0.], [0., .8]]),
    ('uks', (1, 0), [[[.6, 0.], [0., .4]], [[0., 0.], [0., 0.]]]),
    ('gks', 1, [[.5, -.5j], [.5j, .5]]),
])
def test_target_preserves_correlated_occupations_and_spin_layout(layout, nelec, density):
    module = api()
    density = jnp.asarray(density)
    n = density.shape[-1]//2 if layout == 'gks' else density.shape[-1]
    target = module.Target(density, jnp.eye(n), nelec, layout=layout, energy=-1.2)
    assert bool(target.valid)
    np.testing.assert_array_equal(target.density, density)
    np.testing.assert_allclose(target.electron_numbers,
                               (1, 0) if layout == 'uks' else [sum(nelec) if isinstance(nelec, tuple) else nelec])
    assert float(target.energy) == -1.2
    assert jax.jit(lambda t: t.valid)(target)


def test_spinor_pauli_components_preserve_transverse_magnetization():
    target = api().Target(jnp.array([[.5, -.5j], [.5j, .5]]), jnp.eye(1), 1, layout='gks')
    # |alpha> + i|beta> has positive sigma_y, not sigma_x magnetization.
    np.testing.assert_allclose(target.components[:, 0, 0], [1., 0., 1., 0.], atol=1e-14)


@pytest.mark.parametrize('kind', ['shape', 'number', 'hermitian', 'negative', 'overfilled', 'overlap', 'energy'])
def test_invalid_targets_are_rejected_without_normalization(kind):
    module = api()
    density, overlap, number, energy = jnp.eye(2), jnp.eye(2), 2, None
    if kind == 'shape': density = jnp.ones((3, 2))
    if kind == 'number': number = 4
    if kind == 'hermitian': density = density.at[0, 1].set(.2)
    if kind == 'negative': density = jnp.diag(jnp.array([-1., 3.]))
    if kind == 'overfilled': density = jnp.diag(jnp.array([2.1, -.1]))
    if kind == 'overlap': overlap = jnp.diag(jnp.array([1., -1.]))
    if kind == 'energy': energy = jnp.nan
    with pytest.raises(ValueError):
        module.Target(density, overlap, number, energy=energy)


def test_traced_invalid_target_has_an_explicit_validity_flag():
    module = api()
    check = jax.jit(lambda d: module.Target(d, jnp.eye(2), 2).valid)
    assert check(jnp.eye(2))
    assert not check(jnp.full((2, 2), jnp.nan))


def test_nonorthogonal_target_and_live_second_response():
    module = api()
    overlap = jnp.diag(jnp.array([2., .5]))
    def value(x):
        density = jnp.diag(jnp.array([(.8+x)/2, (1.2-x)/.5]))
        target = module.Target(density, overlap, 2)
        return jnp.sum(target.components**2)
    point = jnp.asarray(.02)
    gradient = jax.jit(jax.grad(value))(point)
    hvp = jax.jit(lambda x: jax.jvp(jax.grad(value), (x,), (jnp.ones_like(x),))[1])(point)
    assert np.isfinite([gradient, hvp]).all()
    np.testing.assert_allclose(hvp, 8.5, atol=1e-12, equal_nan=False)
