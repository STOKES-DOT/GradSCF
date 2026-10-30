"""Reference orbitals are distinct from a correlated target 1-RDM."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks


def reference(layout='rks', *, fractional=False):
    s = jnp.diag(jnp.array([1.2, .8]))
    c = jnp.diag(1/jnp.sqrt(jnp.diag(s)))
    occ = jnp.array([1.7, .3]) if fractional else jnp.array([2., 0.])
    eps = jnp.array([-.4, .6])
    if layout == 'uks':
        c = jnp.stack((c, c))
        occ = jnp.stack((occ/2, occ/2))
        eps = jnp.stack((eps, eps+.1))
    elif layout == 'gks':
        c = c.astype(complex) @ jnp.array([[1., 1j], [1j, 1.]])/jnp.sqrt(2.)
        c = jnp.kron(jnp.eye(2), c)
        occ = jnp.array([.8, .2, .8, .2]) if fractional else jnp.array([1., 0., 1., 0.])
        eps = jnp.array([-.4, .6, -.3, .7])
    # Target natural occupations differ deliberately from the KS occupations.
    d = jnp.diag(jnp.array([1.6, .4])/jnp.diag(s))
    if layout == 'uks': d = jnp.stack((d/2, d/2))
    elif layout == 'gks': d = jnp.kron(jnp.eye(2), d/2)
    target = iks.Target(d, s, (1, 1) if layout == 'uks' else 2, layout=layout, energy=-1.)
    cls = getattr(iks, 'OrbitalReference', None)
    assert cls is not None, 'Missing orbital reference contract'
    return cls(target, c, occ, eps, method='supplied_orbitals')


@pytest.mark.parametrize('layout', ['rks', 'uks', 'gks'])
def test_reference_keeps_target_and_ks_density_distinct(layout):
    ref = reference(layout)
    assert ref.valid and ref.method == 'supplied_orbitals'
    assert np.linalg.norm(ref.density-ref.target.density) > .01
    s = ref.target.overlap
    if layout == 'gks': s = jnp.kron(jnp.eye(2), s)
    c = ref.coefficients
    expected = (c*ref.occupations[..., None, :])@c.conj().swapaxes(-1, -2)
    np.testing.assert_allclose(ref.density, expected, atol=2e-13)
    np.testing.assert_allclose(c.conj().swapaxes(-1, -2)@s@c,
                               jnp.broadcast_to(jnp.eye(c.shape[-1]), c.shape), atol=2e-13)
    leaves, tree = jax.tree.flatten(ref)
    rebuilt = jax.tree.unflatten(tree, leaves)
    np.testing.assert_allclose(jax.jit(lambda r: r.density)(rebuilt), expected, atol=2e-13)


def test_reference_rejects_bad_orbitals_and_electron_count():
    ref = reference()
    with pytest.raises(ValueError, match='orthonormal|reference'):
        replace(ref, coefficients=ref.coefficients*1.1)
    with pytest.raises(ValueError, match='electron|reference'):
        replace(ref, occupations=jnp.array([1., 0.]))


@pytest.mark.parametrize('layout', ['rks', 'uks', 'gks'])
def test_fractional_reference_is_valid_without_a_hidden_integer_chart(layout):
    ref = reference(layout, fractional=True)
    assert ref.valid and not ref.integer_occupations


def test_nonfinite_reference_acceptance_flag_is_not_true():
    ref = reference()
    with pytest.raises(ValueError, match='valid|reference'):
        replace(ref, reference_valid=jnp.asarray(jnp.nan))
