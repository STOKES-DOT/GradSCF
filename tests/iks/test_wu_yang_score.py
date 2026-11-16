"""Direct potential assessment must share physical and differentiable operators."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks


def problem(*, grid=None):
    target = iks.Target(jnp.diag(jnp.array([2., 0.])), jnp.eye(2), 2)
    kinetic = jnp.diag(jnp.array([.1, .5]))
    base = kinetic+jnp.diag(jnp.array([-.7, .8]))
    baseline = jnp.array([[-.2, .15], [.15, .2]])
    return iks.WuYang(target, kinetic, base, baseline=baseline, grid=grid)


def test_baseline_contains_exactly_one_xc_potential():
    model = problem()
    score = model.score(model.baseline_potential)
    expected = model.base_fock+model.baseline_potential
    energies, vectors = np.linalg.eigh(expected)
    density = 2*np.outer(vectors[:, 0], vectors[:, 0])
    value = 2*energies[0]-np.trace((expected-model.kinetic)@model.target.density)
    np.testing.assert_allclose(score.fock, expected, atol=1e-14)
    np.testing.assert_allclose(score.value, value, atol=1e-14)
    np.testing.assert_allclose(score.density, density, atol=1e-14)
    assert score.valid
    assert score.reward == 0. and score.delta == 0.


def test_jit_parameter_ad_matches_density_moment_and_finite_difference():
    model = problem()
    direction = jnp.array([[.2, .3], [.3, -.2]])
    objective = jax.jit(lambda a: model.score(model.baseline_potential+a*direction).value)
    point = .13
    score = model.score(model.baseline_potential+point*direction)
    expected = jnp.einsum('ij,ji->', direction, score.density-model.target.density)
    derivative = jax.jit(jax.grad(objective))(point)
    numerical = (objective(point+1e-4)-objective(point-1e-4))/2e-4
    np.testing.assert_allclose(derivative, expected, atol=1e-12)
    np.testing.assert_allclose(derivative, numerical, rtol=1e-7, atol=1e-10)
    assert np.isfinite(jax.grad(jax.grad(objective))(point))


def test_gauge_and_nonorthogonal_density_conventions():
    s = jnp.diag(jnp.array([2., .5]))
    target = iks.Target(jnp.diag(jnp.array([1., 0.])), s, 2)
    model = iks.WuYang(target, jnp.eye(2), jnp.diag(jnp.array([-2., 1.])),
                      baseline=jnp.zeros((2, 2)))
    v = jnp.array([[.1, .2], [.2, -.1]])
    first, second = model.score(v), model.score(v+3.7*s)
    np.testing.assert_allclose(first.value, second.value, atol=1e-13)
    np.testing.assert_allclose(first.density, second.density, atol=1e-13)
    np.testing.assert_allclose(jnp.trace(s@first.density), 2., atol=1e-13)


def test_whitening_roundoff_preserves_a_symmetric_physical_hamiltonian():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(24, 24)))
    values = np.geomspace(1e-4, 5., 24)
    s = (q*values)@q.T
    root = (q*np.sqrt(values))@q.T
    inverse = (q/np.sqrt(values))@q.T
    d = 2*np.outer(inverse[:, 0], inverse[:, 0])
    target = iks.Target(d, s, 2)
    fock = root@np.diag(np.linspace(-2., 2., 24))@root
    model = iks.WuYang(target, jnp.zeros((24, 24)), fock, baseline=jnp.zeros((24, 24)))
    assert model.score(jnp.zeros((24, 24))).valid


def test_energy_callback_remains_differentiable_in_parameters():
    model = problem()
    direction = jnp.array([[0., 1.], [1., 0.]])
    def energy(p, d):
        return jnp.sum((model.baseline_potential+p*direction)*d.T)
    objective = jax.jit(lambda p: model.assess(energy, p).value)
    np.testing.assert_allclose(objective(.2), model.score(model.baseline_potential+.2*direction).value)
    assert np.isfinite(jax.jit(jax.grad(objective))(.2))


def test_nonfinite_constant_energy_is_not_hidden_by_a_zero_derivative():
    result = problem().assess(lambda p, d: jnp.asarray(jnp.nan))
    assert not result.valid and np.isnan(result.reward)


def test_internal_degeneracy_is_valid_but_boundary_closing_is_not():
    target = iks.Target(jnp.diag(jnp.array([2., 2., 0., 0.])), jnp.eye(4), 4)
    model = iks.WuYang(target, jnp.zeros((4, 4)), jnp.diag(jnp.array([-1., -1., 1., 1.])),
                      baseline=jnp.zeros((4, 4)))
    direction = jnp.ones((4, 4))*.05
    value = jax.jit(jax.value_and_grad(lambda a: model.score(a*direction).value))
    assert np.isfinite(value(0.)).all()
    assert model.score(jnp.zeros((4, 4))).valid
    bad = model.score(-model.base_fock)
    assert not bad.valid and np.isnan(bad.reward)


def test_bad_candidate_cannot_produce_a_finite_reward():
    model = problem()
    assert not model.score(jnp.full((2, 2), jnp.nan)).valid
    asymmetric = model.score(jnp.array([[0., 1.], [0., 0.]]))
    assert not asymmetric.valid and np.isnan(asymmetric.reward)
    with pytest.raises(ValueError, match='shape'):
        model.score(jnp.zeros((3, 3)))


def test_optional_grid_reports_actual_induced_density_error():
    model = problem(grid=(jnp.eye(2), jnp.ones(2)))
    result = model.score(model.baseline_potential)
    expected = jnp.linalg.norm(jnp.diag(result.density-model.target.density))/2
    np.testing.assert_allclose(result.density_error, expected, atol=1e-14)


def test_uks_counts_and_independent_spin_gauges():
    d = jnp.stack((jnp.diag(jnp.array([1., 0.])), jnp.diag(jnp.array([1., 0.]))))
    target = iks.Target(d, jnp.eye(2), (1, 1), layout='uks')
    base = jnp.stack((jnp.diag(jnp.array([-1., 1.])), jnp.diag(jnp.array([-2., 2.]))))
    model = iks.WuYang(target, jnp.eye(2), base, baseline=jnp.zeros_like(base))
    result = model.score(jnp.array([3., -2.])[:, None, None]*jnp.eye(2))
    assert result.valid
    np.testing.assert_allclose(result.density, d, atol=1e-14)
    np.testing.assert_allclose(result.reward, 0., atol=1e-12)


def test_local_field_reuses_the_blocked_potential_map_and_parameter_ad():
    from gradscf.integrals.grids.blocks import AOGrid
    grid = AOGrid(jnp.array([[.1*i, .2, .07*i] for i in range(7)]),
        jnp.linspace(.1, .3, 7), jnp.array([[0., 0., 0.], [.5, 0., .1]]),
        jnp.array([[.7], [1.1]]), jnp.ones((2, 1)), (1, 1), ((0, 0, 0),)*2, 3)
    model = problem(grid=grid)
    def field(p, coords): return p*coords[:, 0]
    ao = grid.block(0, grid.ngrids)[0]
    matrix = jnp.einsum('g,gi,gj->ij', grid.weights*field(.2, grid.coords), ao, ao)
    actual = model.local(field, .2)
    np.testing.assert_allclose(actual.value, model.score(matrix).value, atol=1e-13)
    derivative = jax.jit(jax.grad(lambda p: model.local(field, p).value))(.2)
    assert np.isfinite(derivative)


def test_custom_xc_and_neural_callbacks_share_the_direct_assessment():
    from types import SimpleNamespace
    from gradscf import dft
    model = problem()
    molecule = SimpleNamespace(ao=jnp.eye(2), grid=SimpleNamespace(weights=jnp.ones(2)))
    custom = dft.custom_xc(input_fn=lambda x: x.total_density_matrix,
        energy_fn=lambda p, d: p*jnp.sum(d*d))
    neural = dft.custom_xc(input_fn=lambda x: x.total_density_matrix.reshape(-1),
        energy_fn=lambda p, x: jnp.sum(jnp.tanh(x@p)))
    for functional, params in [(custom, jnp.asarray(.1)), (neural, jnp.ones((4, 2))*.1)]:
        energy = lambda p, d: functional.energy_for_density(p, molecule, d)
        value, gradient = jax.jit(jax.value_and_grad(lambda p: model.assess(energy, p).value))(params)
        assert np.isfinite(value) and np.isfinite(gradient).all()
