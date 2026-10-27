"""Orbital hints may accelerate DF exchange; density defines its value and AD."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf.scf.uks import _spin_jk_matrices
from gradscf.integrals.molecular.jk import build_jk_from_df


def inputs():
    rng = np.random.default_rng(93)
    raw = rng.normal(size=(4, 3, 3))*.2
    factors = jnp.asarray((raw+raw.transpose(0, 2, 1))/2)
    coeff = jnp.eye(3)
    occ = (jnp.array([1., 1., 0.]), jnp.array([1., 0., 0.]))
    density = jnp.stack([jnp.diag(o) for o in occ])
    return factors, coeff, occ, density


def hinted(factors, density, coeff, occ):
    return _spin_jk_matrices(jnp.zeros(0), density[0], density[1], df_factors=factors,
        mo_coeff_a=coeff, mo_coeff_b=coeff, mo_occ_a=occ[0], mo_occ_b=occ[1])


def physical(factors, density):
    total = density.sum(axis=0)
    return total, build_jk_from_df(factors, total)[0], *[build_jk_from_df(factors, d)[1] for d in density]


def test_density_only_spin_guess_is_not_replaced_by_core_orbital_exchange():
    factors, coeff, occ, density = inputs()
    seed = density.at[0, 0, 0].add(-.2).at[0, 2, 2].add(.2)
    seed = seed.at[1].set(jnp.eye(3)/3)
    for actual, expected in zip(hinted(factors, seed, coeff, occ), physical(factors, seed)):
        np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_consistent_primal_keeps_independent_density_jvp():
    factors, coeff, occ, density = inputs()
    tangent = jnp.arange(18., dtype=jnp.float64).reshape(2, 3, 3)*.01
    tangent = .5*(tangent+tangent.transpose(0, 2, 1))
    actual = jax.jvp(lambda d: hinted(factors, d, coeff, occ), (density,), (tangent,))
    expected = jax.jvp(lambda d: physical(factors, d), (density,), (tangent,))
    for got, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(got, reference, atol=1e-13)


def test_spin_df_mixed_factor_density_response_uses_physical_operands():
    factors, coeff, occ, density = inputs()
    weight = jnp.arange(9.).reshape(3, 3)*.03
    def loss(b, d, hint):
        result = hinted(b, d, coeff, occ) if hint else physical(b, d)
        return jnp.sum((result[1]+result[2]-.7*result[3])*weight)
    def gradient(b, d, hint):
        return jax.grad(lambda bb, dd: loss(bb, dd, hint), argnums=(0, 1))(b, d)
    directions = (jnp.ones_like(factors)*.01, jnp.ones_like(density)*.02)
    actual = jax.jvp(lambda b,d: gradient(b,d,True), (factors,density), directions)
    expected = jax.jvp(lambda b,d: gradient(b,d,False), (factors,density), directions)
    for got, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(got, reference, atol=1e-12)
