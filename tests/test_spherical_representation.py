"""Canonical real spherical embedding independently checks angular integrals."""
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.basis import BasisTopology, cartesian_angular_tuples


def transform(topology):
    from gradscf.integrals.basis import cartesian_to_spherical
    return np.asarray(cartesian_to_spherical(topology))


def test_d_shell_transform_is_orthonormal_in_exact_sphere_metric():
    top = BasisTopology((2,), (1,), (1,), (6,))
    t = transform(top)
    powers = [(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
    def moment(a):
        if any(k%2 for k in a): return 0.
        return 2*math.prod(math.gamma((k+1)/2) for k in a)/math.gamma((sum(a)+3)/2)
    metric = np.array([[moment(tuple(x+y for x,y in zip(a,b))) for b in powers] for a in powers])
    np.testing.assert_allclose(t.T@metric@t, np.eye(5), atol=3e-15)
    # Real harmonics have the libcint order xy,yz,z2,xz,x2-y2.
    at_z = np.array([0.,0.,0.,0.,0.,1.])@t
    np.testing.assert_allclose(at_z, [0.,0.,math.sqrt(5/(4*math.pi)),0.,0.], atol=1e-15)


def test_transform_preserves_shell_contraction_order_and_low_l_normalization():
    top = BasisTopology((0,1,2), (3,2,1), (2,1,2), (6,))
    t = transform(top)
    assert t.shape == (17,15)
    np.testing.assert_array_equal(t[:5,:5], np.eye(5))
    d = transform(BasisTopology((2,), (1,), (1,), (6,)))
    np.testing.assert_array_equal(t[5:11,5:10], d)
    np.testing.assert_array_equal(t[11:17,10:15], d)


def test_high_angular_momentum_requires_an_explicit_supported_transform():
    with pytest.raises(NotImplementedError, match='angular|s.*p.*d|l='):
        transform(BasisTopology((5,), (1,), (1,), (6,)))


@pytest.mark.parametrize('angular_momentum', [3, 4], ids=['f', 'g'])
def test_high_angular_shell_transform_is_orthonormal_in_exact_sphere_metric(angular_momentum):
    powers = cartesian_angular_tuples(angular_momentum)
    def moment(a):
        if any(k%2 for k in a): return 0.
        return 2*math.prod(math.gamma((k+1)/2) for k in a)/math.gamma((sum(a)+3)/2)
    metric = np.array([[moment(tuple(x+y for x,y in zip(a,b))) for b in powers] for a in powers])
    t = transform(BasisTopology((angular_momentum,), (1,), (1,), (6,)))
    n_spherical = 2 * angular_momentum + 1
    assert t.shape == (len(powers), n_spherical)
    np.testing.assert_allclose(t.T @ metric @ t, np.eye(n_spherical), rtol=0., atol=4e-15)
    at_z = np.zeros(n_spherical)
    at_z[angular_momentum] = math.sqrt(n_spherical / (4 * math.pi))
    np.testing.assert_allclose(t[-1], at_z, rtol=0., atol=1e-15)


@pytest.mark.parametrize('angular_momentum', [0, 1, 2, 3, 4])
def test_transform_matches_libcint_coefficients(angular_momentum):
    cart2sph = pytest.importorskip('pyscf.gto').cart2sph
    t = transform(BasisTopology((angular_momentum,), (1,), (1,), (6,)))
    np.testing.assert_allclose(t, cart2sph(angular_momentum, normalized='sp'), rtol=0., atol=4e-15)


def test_mixed_f_shells_preserve_contraction_blocks():
    t = transform(BasisTopology((3,0,2,3), (1,1,1,1), (2,1,1,1), (6,)))
    assert t.shape == (37,27)
    f = transform(BasisTopology((3,), (1,), (1,), (6,)))
    d = transform(BasisTopology((2,), (1,), (1,), (6,)))
    expected = np.zeros_like(t)
    expected[:10,:7] = f
    expected[10:20,7:14] = f
    expected[20,14] = 1.
    expected[21:27,15:20] = d
    expected[27:37,20:27] = f
    np.testing.assert_array_equal(t, expected)


@pytest.mark.parametrize('angular_momentum', [3, 4], ids=['f', 'g'])
def test_high_angular_density_embedding_preserves_potential_contraction_and_ad(angular_momentum):
    from gradscf.integrals.basis import cartesian_to_spherical
    t = cartesian_to_spherical(BasisTopology((angular_momentum,), (1,), (1,), (6,)))
    n_cartesian, n_spherical = t.shape
    rng = np.random.default_rng(0)
    factor = jnp.asarray(rng.normal(size=(n_spherical, 3)))
    density = factor@factor.T
    raw = jnp.asarray(rng.normal(size=(n_cartesian, n_cartesian)))
    potential = (raw+raw.T)/2
    direction = jnp.asarray(rng.normal(size=(n_spherical, n_spherical)))
    direction = (direction+direction.T)/2
    embed = jax.jit(lambda d: t@d@t.T)
    energy = jax.jit(lambda d: jnp.einsum('ij,ji->', potential, embed(d)))
    expected = t.T@potential@t
    np.testing.assert_allclose(energy(density), jnp.einsum('ij,ji->', expected, density), atol=1e-13)
    _, tangent = jax.jvp(embed, (density,), (direction,))
    np.testing.assert_allclose(tangent, t@direction@t.T, atol=1e-14)
    np.testing.assert_allclose(jax.grad(energy)(density), expected, atol=1e-14)
