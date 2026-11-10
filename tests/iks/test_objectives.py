"""Analytic fixed-reference objectives and their mixed parameter derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks
from test_orbital_reference import reference
from test_potential import grid


def energy(ref):
    s = ref.target.overlap
    if ref.target.layout == 'gks': s = jnp.kron(jnp.eye(2), s)
    # C^dagger F C has off-diagonal p and a unit virtual-occupied gap.
    h = jnp.array([[-.4, 0.], [0., .6]])
    v = jnp.array([[0., 1.], [1., 0.]])
    if ref.target.layout == 'gks':
        h, v = jnp.kron(jnp.eye(2), h), jnp.kron(jnp.eye(2), v)
    c = ref.coefficients
    h_ao = s@c@h@c.conj().swapaxes(-1, -2)@s
    v_ao = s@c@v@c.conj().swapaxes(-1, -2)@s
    return lambda p, d: jnp.einsum('...ij,...ji->', h_ao+p*v_ao, d).real


def test_energy_only_does_not_construct_a_density_derivative():
    ref = reference()
    # A valid energy implementation can deliberately have no density AD rule.
    @jax.custom_jvp
    def opaque(d): return jnp.sum(d*d)
    @opaque.defjvp
    def rule(primals, tangents):
        raise AssertionError('Energy-only evaluation requested density AD')
    result = iks.evaluate(ref, lambda p, d: p*opaque(d), .2, quantities=('energy',))
    assert result.valid and result.fock is None and result.operator_mse is None
    np.testing.assert_allclose(result.total_energy, .2*jnp.sum(ref.density**2), atol=2e-13)


@pytest.mark.parametrize('layout', ['rks', 'uks', 'gks'])
def test_stationarity_uses_occupied_virtual_rotations_and_mixed_ad(layout):
    ref = reference(layout)
    fn = energy(ref)
    result = iks.evaluate(ref, fn, .2, quantities=('energy', 'stationarity'))
    assert result.valid
    # One rotation for RKS, one in each spin sector for UKS/spinor GKS.
    expected = .04 if layout == 'rks' else .08
    np.testing.assert_allclose(result.rotation_norm2, expected, atol=2e-13)
    def objective(p):
        return iks.loss(ref, fn, p, weights={'stationarity': {'norm2': 1.}})[0]
    derivative = jax.jit(jax.grad(objective))(.2)
    np.testing.assert_allclose(derivative, 2*expected/.2, atol=2e-12)
    finite = (objective(.20001)-objective(.19999))/.00002
    np.testing.assert_allclose(derivative, finite, atol=2e-10)


def test_operator_gauge_and_normalized_energy_objective():
    ref = reference()
    fn = energy(ref)
    shifted = lambda p, d: fn(p, d)+.3*(jnp.trace(ref.target.overlap@d)-2.)
    a = iks.evaluate(ref, fn, .2, quantities=('operator',))
    b = iks.evaluate(ref, shifted, .2, quantities=('operator',))
    np.testing.assert_allclose(a.operator_mse, b.operator_mse, atol=2e-13)
    value, result = iks.loss(ref, fn, .2, weights={'energy': {'mse': 1., 'mae': 1.}},
                             scales={'energy': 2.})
    error = result.energy_error/2.
    np.testing.assert_allclose(value, error**2+jnp.abs(error), atol=2e-13)


def test_predicted_density_retains_the_reference_inversion_residual():
    ref = reference('uks')
    provider = grid()
    result = iks.evaluate(ref, energy(ref), 0., quantities=('predicted_density',), grid=provider)
    ao = provider.block(0, provider.ngrids)[0]
    difference = ref.density-ref.target.density
    rho = jnp.einsum('gi,sij,gj->sg', ao, difference, ao)
    expected = jnp.sum(provider.weights*rho**2)
    np.testing.assert_allclose(result.predicted_density_norm2, expected, atol=2e-13)
    np.testing.assert_allclose(result.reference_density_norm2, expected, atol=2e-13)
    assert expected > 0


def test_invalid_candidate_is_not_a_finite_loss_or_reward():
    ref = reference()
    value, result = iks.loss(ref, lambda p, d: p*jnp.sum(d)*jnp.nan, .2,
                             weights={'energy': {'mse': 1.}})
    assert not result.valid and not np.isfinite(value)


def test_objective_rejects_missing_labels_and_fractional_response():
    from dataclasses import replace
    ref = reference()
    missing = replace(ref, target=replace(ref.target, energy=None))
    with pytest.raises(ValueError, match='energy|label'):
        iks.loss(missing, energy(ref), .2, weights={'energy': {'mse': 1.}})
    with pytest.raises(ValueError, match='fractional|integer'):
        iks.evaluate(reference(fractional=True), energy(ref), .2, quantities=('stationarity',))
    with pytest.raises(ValueError, match='grid'):
        iks.evaluate(ref, energy(ref), .2, quantities=('predicted_density',))


@pytest.mark.parametrize('weights,scales', [({'density': {'mse': 1.}}, {}),
    ({'energy': {'mse': -1.}}, {}), ({'energy': {'mse': 1.}}, {'energy': 0.})])
def test_objective_configuration_is_explicit(weights, scales):
    ref = reference()
    with pytest.raises(ValueError):
        iks.loss(ref, energy(ref), .2, weights=weights, scales=scales)


def test_xc_teacher_is_not_compared_to_the_total_fock():
    from dataclasses import replace
    ref = reference()
    xc = lambda p, d: p*jnp.sum(d*d)
    teacher = jax.grad(lambda d: xc(.2, d))(ref.density)
    ref = replace(ref, xc_potential=teacher)
    total = lambda p, d: xc(p, d)+jnp.trace(jnp.diag(jnp.array([-.8, .5]))@d)
    value, score = iks.loss(ref, total, .2, weights={'xc_potential': {'mse': 1.}}, xc_energy_fn=xc)
    np.testing.assert_allclose(value, 0., atol=2e-13)
    assert score.xc_potential_mse == 0
    with pytest.raises(ValueError, match='XC|xc_energy'):
        iks.loss(ref, total, .2, weights={'xc_potential': {'mse': 1.}})


def test_predicted_grid_density_matches_explicit_first_order_rotation():
    ref = reference('uks')
    provider = grid()
    fn = energy(ref)
    score = iks.evaluate(ref, fn, .13, quantities=('predicted_density',), grid=provider)
    kappa = jnp.array([[0., 0.], [-.13, 0.]])
    first = kappa*ref.occupations[..., None, :]
    delta = ref.coefficients@(first+first.swapaxes(-1, -2))@ref.coefficients.swapaxes(-1, -2)
    ao = provider.block(0, provider.ngrids)[0]
    difference = ref.density+delta-ref.target.density
    rho = jnp.einsum('gi,sij,gj->sg', ao, difference, ao)
    np.testing.assert_allclose(score.predicted_density_norm2, jnp.sum(provider.weights*rho**2), atol=2e-13)
    objective = lambda p: iks.loss(ref, fn, p, grid=provider,
        weights={'predicted_density': {'norm2': 1.}})[0]
    actual = jax.jit(jax.grad(objective))(.13)
    finite = (objective(.13001)-objective(.12999))/.00002
    np.testing.assert_allclose(actual, finite, atol=2e-10)


def test_stationarity_does_not_materialize_an_ao_density_response(monkeypatch):
    import gradscf.dft.iks.response as response
    ref = reference()
    original = response.orbital_response
    def checked(reference, fock, gap_floor, **kwargs):
        assert kwargs.get('with_density') is False
        delta, norm, valid = original(reference, fock, gap_floor, **kwargs)
        assert delta is None
        return delta, norm, valid
    monkeypatch.setattr('gradscf.dft.iks.assessment.orbital_response', checked)
    assert iks.evaluate(ref, energy(ref), .1, quantities=('stationarity',)).valid


@pytest.mark.parametrize('weights', [jnp.array([-1., -1.]), jnp.array([jnp.nan, 1.]), jnp.zeros(2)])
def test_predicted_density_cannot_reward_an_invalid_quadrature(weights):
    ref = reference('uks')
    ao = jnp.array([[1., .2], [.3, .8]])
    value, score = iks.loss(ref, energy(ref), .1, grid=(ao, weights),
        weights={'predicted_density': {'norm2': 1.}})
    assert not score.valid and not np.isfinite(value)


def test_predicted_density_rejects_complex_quadrature():
    ref = reference('uks')
    with pytest.raises(ValueError, match='real|weight|grid'):
        iks.evaluate(ref, energy(ref), .1, quantities=('predicted_density',),
            grid=(jnp.ones((2, 2)), jnp.ones(2, dtype=complex)))


def test_masked_nonfinite_parameters_do_not_receive_a_valid_score():
    ref = reference()
    fn = lambda p, d: jnp.nan_to_num(p*jnp.sum(d))
    value, score = iks.loss(ref, fn, jnp.asarray(jnp.nan), weights={'energy': {'mse': 1.}})
    assert np.isfinite(score.total_energy)
    assert not score.valid and not np.isfinite(value)
