"""SCF-free scores must test Aufbau filling, not just orbital stationarity."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks


def sample(layout='rks', *, correlated=False):
    n = 4 if layout == 'gks' else 2
    occupations = jnp.array([1., 1., 0., 0.]) if layout == 'gks' else jnp.array([2., 0.])
    energies = jnp.array([-1., -1., 1., 1.]) if layout == 'gks' else jnp.array([-1., 1.])
    coefficients = jnp.eye(n, dtype=complex if layout == 'gks' else float)
    density = jnp.diag(occupations)
    if correlated:
        density = .8*density + .2*jnp.diag(occupations[::-1])
    if layout == 'uks':
        occupations = jnp.stack((occupations/2, occupations/2))
        energies = jnp.stack((energies, energies))
        coefficients = jnp.stack((coefficients, coefficients))
        density = jnp.stack((density/2, density/2))
    target = iks.Target(density, jnp.eye(2), (1, 1) if layout == 'uks' else 2, layout=layout)
    return iks.OrbitalReference(target, coefficients, occupations, energies)


def linear_energy(fock):
    return lambda p, d: jnp.einsum('...ij,...ji->', fock, d).real


GRID = (jnp.eye(2), jnp.ones(2))


@pytest.mark.parametrize('layout', ['rks', 'uks', 'gks'])
def test_exact_candidate_and_constant_gauge(layout):
    ref = sample(layout)
    a = iks.reward(ref, linear_energy(ref.fock), grid=GRID)
    b = iks.reward(ref, linear_energy(ref.fock+3.7*ref.overlap), grid=GRID)
    assert a.valid and b.valid
    np.testing.assert_allclose(a.value, 1., atol=1e-12)
    np.testing.assert_allclose(b.value, a.value, atol=1e-12)
    np.testing.assert_allclose(a.variational_gap, 0., atol=1e-12)
    np.testing.assert_allclose(a.wu_yang_gap, 0., atol=1e-12)
    np.testing.assert_allclose(a.density, ref.density, atol=1e-12)
    np.testing.assert_allclose(a.density_error, 0., atol=1e-12)


def test_wrong_aufbau_is_not_rewarded_as_a_stationary_solution():
    ref = sample()
    energy = linear_energy(-ref.fock)
    old = iks.evaluate(ref, energy, quantities=('stationarity',))
    np.testing.assert_allclose(old.rotation_norm2, 0.)
    score = iks.reward(ref, energy, grid=GRID, energy_scale=1., density_scale=1.)
    assert score.valid
    np.testing.assert_allclose(score.variational_gap, 4.)
    np.testing.assert_allclose(score.wu_yang_gap, 4.)
    np.testing.assert_allclose(score.density_error, np.sqrt(2.))
    np.testing.assert_allclose(score.loss, 6.)
    np.testing.assert_allclose(score.value, 1/7)


def test_correlated_target_and_reference_error_remain_distinct():
    ref = sample(correlated=True)
    score = iks.reward(ref, linear_energy(ref.fock), grid=GRID)
    np.testing.assert_allclose(score.variational_gap, 0., atol=1e-12)
    assert score.reference_density_error > 0.
    np.testing.assert_allclose(score.density_error, score.reference_density_error)
    assert 0. < score.value < 1.
    # A finite-space inverse need not maximize the unrestricted matrix dual.
    shifted = iks.reward(ref, linear_energy(.5*ref.fock), grid=GRID)
    assert shifted.wu_yang_gap < 0.
    np.testing.assert_allclose(shifted.variational_gap, 0., atol=1e-12)


@pytest.mark.parametrize('layout', ['rks', 'uks', 'gks'])
def test_parameter_ad_matches_finite_difference_without_scf(layout):
    ref = sample(layout)
    perturbation = jnp.zeros_like(ref.fock)
    if layout == 'gks':
        perturbation = perturbation.at[0, 2].set(1j).at[2, 0].set(-1j)
    else:
        perturbation = perturbation.at[..., 0, 1].set(1.).at[..., 1, 0].set(1.)
    calls = []
    def energy(p, d):
        calls.append(1)
        return jnp.einsum('...ij,...ji->', ref.fock+p*perturbation, d).real
    def objective(p):
        return iks.reward(ref, energy, p, grid=GRID, energy_scale=1., density_scale=1.).value
    value, derivative = jax.jit(jax.value_and_grad(objective))(.3)
    assert len(calls) == 1  # One energy/derivative trace, no candidate iteration.
    step = 1e-4
    expected = (objective(.3+step)-objective(.3-step))/(2*step)
    np.testing.assert_allclose(derivative, expected, rtol=3e-7, atol=1e-9)
    assert jnp.isfinite(value) and jnp.isfinite(derivative)


def test_bad_reference_is_rejected_and_bad_candidate_has_zero_reward():
    ref = sample()
    fractional = replace(ref, occupations=jnp.array([1.8, .2]))
    with pytest.raises(ValueError, match='integer'):
        iks.reward(fractional, linear_energy(ref.fock), grid=GRID)
    excited = replace(ref, energies=-ref.energies)
    with pytest.raises(ValueError, match='Aufbau|ground'):
        iks.reward(excited, linear_energy(ref.fock), grid=GRID)
    for fock in (jnp.zeros((2, 2)), ref.fock*jnp.nan):
        score = iks.reward(ref, linear_energy(fock), grid=GRID)
        assert not score.valid and score.value == 0.
        assert jnp.isnan(score.loss)
    bad = iks.reward(ref, linear_energy(ref.fock*jnp.nan), grid=GRID)
    assert jnp.any(jnp.isnan(bad.fock))


@pytest.mark.parametrize('keyword,value', [('energy_scale', 0.), ('density_scale', float('inf')),
                                         ('gap_tolerance', -1.)])
def test_scales_and_gap_tolerance_are_explicit_positive_numbers(keyword, value):
    ref = sample()
    with pytest.raises(ValueError, match='positive'):
        iks.reward(ref, linear_energy(ref.fock), grid=GRID, **{keyword: value})


def test_nonorthogonal_ao_frame_and_empty_uks_channel():
    s = jnp.diag(jnp.array([1.2, .8]))
    c = jnp.diag(1/jnp.sqrt(jnp.diag(s)))
    d = jnp.stack((jnp.diag(jnp.array([1., 0.])/jnp.diag(s)), jnp.zeros((2, 2))))
    target = iks.Target(d, s, (1, 0), layout='uks')
    ref = iks.OrbitalReference(target, jnp.stack((c, c)),
        jnp.array([[1., 0.], [0., 0.]]), jnp.array([[-1., 1.], [-1., 1.]]))
    score = iks.reward(ref, linear_energy(ref.fock), grid=GRID)
    assert score.valid
    np.testing.assert_allclose(score.value, 1., atol=1e-12)
    np.testing.assert_allclose(score.density, d, atol=1e-12)


def test_internal_degeneracy_has_finite_second_parameter_derivative():
    occ = jnp.array([2., 2., 0., 0.])
    eps = jnp.array([-1., -1., 1., 1.])
    ref = iks.OrbitalReference(iks.Target(jnp.diag(occ), jnp.eye(4), 4),
                               jnp.eye(4), occ, eps)
    direction = jnp.zeros((4, 4)).at[0, 2].set(1.).at[2, 0].set(1.)
    def objective(p):
        return iks.reward(ref, lambda p, d: jnp.sum((ref.fock+p*direction)*d), p,
            grid=(jnp.eye(4), jnp.ones(4)), energy_scale=1., density_scale=1.).value
    curvature = jax.jit(jax.grad(jax.grad(objective)))(0.)
    np.testing.assert_allclose(curvature, -2., atol=2e-10)


def test_external_custom_xc_uses_the_existing_energy_adapter():
    from gradscf import dft
    from test_hybrid_energy import one_ao_state
    state = one_ao_state()
    target = iks.Target(jnp.array([[2.]]), jnp.eye(1), 2)
    ref = iks.OrbitalReference(target, jnp.eye(1), jnp.array([2.]), jnp.array([-1.]))
    functional = dft.custom_xc(input_fn=lambda fields: fields.density_matrix,
        energy_fn=lambda p, d: p*jnp.sum(d*d), name='toy_parameterized_energy')
    energy = iks.energy_function(state, functional, layout='rks', nuclear_repulsion=0.)
    score = iks.reward(ref, energy, -.2, grid=(jnp.ones((1, 1)), jnp.ones(1)))
    assert score.valid and score.value == 1.


def test_reward_weights_are_declared_and_nonzero():
    ref = sample()
    for weights in ({}, {'other': 1.}, {'gap': -1.}, {'gap': 0.}):
        with pytest.raises(ValueError, match='weights'):
            iks.reward(ref, linear_energy(ref.fock), grid=GRID, weights=weights)
