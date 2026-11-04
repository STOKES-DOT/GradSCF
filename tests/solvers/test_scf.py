"""Shared SCF execution: matrix layouts, numerical gates and explicit AD."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def api():
    from gradscf.solvers.scf import SCFConfig, SCFState, orbital_gradient_norm, solve_scf
    return SCFConfig, SCFState, orbital_gradient_norm, solve_scf


def adjoint(value):
    return value.conj().swapaxes(-1, -2)


def constant_problem(theta, layout, *, config=None, calls=None, energy_offset=0., **solve_options):
    Config, State, gradient, solve = api()
    h = jnp.array([[-1., theta], [theta, 1.]])
    if layout == "unrestricted":
        beta = jnp.array([[-.8, theta*.5], [theta*.5, 1.2]])
        h = jnp.stack((h, beta))
        occupation = jnp.array([[1., 0.], [1., 0.]])
        coeff = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    elif layout == "spinor":
        mixing = .13j*jnp.eye(2)
        h = jnp.block([[h, mixing], [mixing.conj().T, h]])
        occupation, coeff = jnp.array([1., 1., 0., 0.]), jnp.eye(4, dtype=h.dtype)
    else:
        occupation, coeff = jnp.array([2., 0.]), jnp.eye(2)

    def density_from(c):
        return (c*occupation[..., None, :]) @ adjoint(c)

    def energy(dm):
        return energy_offset+jnp.sum(dm*h.swapaxes(-1, -2)).real

    def evaluate(dm, c, eps, previous):
        if calls is not None:
            jax.debug.callback(lambda: calls.append(1), ordered=True)
        return energy(dm), h, ()

    dm = density_from(coeff)
    initial = State(dm, coeff, jnp.zeros_like(occupation), energy(dm), h)
    cfg = Config(max_cycle=5, conv_tol=1e-10, conv_tol_density=1e-10,
                 conv_tol_grad=1e-10) if config is None else config
    result = solve(initial, evaluate=evaluate, diagonalize=jnp.linalg.eigh,
        density_from_coeff=density_from,
        diis_error=lambda f, state: f @ state.density-state.density @ f,
        orbital_gradient=lambda state: gradient(state.fock, state.mo_coeff, occupation),
        config=cfg, **solve_options)
    return result, h, occupation


@pytest.mark.parametrize("method,layout", [
    ("RHF", "restricted"), ("RKS", "restricted"),
    ("UHF", "unrestricted"), ("UKS", "unrestricted"),
    ("GHF", "spinor"), ("GKS", "spinor"),
])
def test_six_consumer_layouts_converge_to_independent_eigen_solution(method, layout):
    result, h, occupation = jax.jit(lambda t: constant_problem(t, layout))(.31)
    eigenvalues, coeff = np.linalg.eigh(np.asarray(h))
    expected_dm = (coeff*np.asarray(occupation)[..., None, :]) @ adjoint(coeff)
    expected_energy = np.sum(eigenvalues*np.asarray(occupation))
    assert bool(result.converged), method
    assert int(result.cycles) == 2
    np.testing.assert_allclose(result.state.density, expected_dm, atol=2e-12, rtol=0)
    np.testing.assert_allclose(result.state.energy, expected_energy, atol=2e-12, rtol=0)
    assert float(result.metrics.gradient) < 1e-10
    assert result.history.iterates == ()


@pytest.mark.parametrize("layout", ["restricted", "unrestricted", "spinor"])
def test_dynamic_inputs_value_gradient_jvp_vjp_and_hvp(layout):
    def expected(theta):
        if layout == "unrestricted":
            return -jnp.sqrt(1+theta**2)+.2-jnp.sqrt(1+.25*theta**2)
        return -2*jnp.sqrt(1+theta**2)

    def energy(theta):
        return constant_problem(theta, layout)[0].state.energy

    value_grad = jax.jit(jax.value_and_grad(energy))
    for theta in (.31, .47):
        value, derivative = value_grad(theta)
        exact, exact_derivative = jax.value_and_grad(expected)(theta)
        np.testing.assert_allclose((value, derivative), (exact, exact_derivative), atol=2e-11, rtol=0)
    tangent = .23
    _, directional = jax.jvp(energy, (.31,), (tangent,))
    _, pullback = jax.vjp(energy, .31)
    np.testing.assert_allclose(directional, pullback(1.)[0]*tangent, atol=2e-11, rtol=0)
    actual_hvp = jax.jvp(jax.grad(energy), (.31,), (tangent,))[1]
    exact_hvp = jax.jvp(jax.grad(expected), (.31,), (tangent,))[1]
    np.testing.assert_allclose(actual_hvp, exact_hvp, atol=2e-10, rtol=0)


@pytest.mark.parametrize("complex_fock", [False, True])
def test_fractional_occupation_residual_keeps_unequal_frontier_rotations(complex_fock):
    _, _, gradient, _ = api()
    off = .3+.4j if complex_fock else .3
    fock = jnp.array([[-1., off], [jnp.conj(off), 1.]])
    coeff = jnp.eye(2, dtype=fock.dtype)
    occupation = jnp.array([1.6, .4])
    residual = lambda occ: gradient(fock, coeff, occ)
    np.testing.assert_allclose(residual(occupation), 1.2*abs(off), atol=1e-13, rtol=0)
    np.testing.assert_allclose(jax.grad(residual)(occupation),
                               jnp.array([abs(off), -abs(off)]), atol=1e-13, rtol=0)
    direction = jnp.array([.2, -.2])
    forward = jax.jvp(residual, (occupation,), (direction,))[1]
    _, back = jax.vjp(residual, occupation)
    np.testing.assert_allclose(forward, jnp.vdot(back(1.)[0], direction), atol=1e-13, rtol=0)


def test_converged_state_is_frozen_without_extra_evaluations():
    Config, _, _, _ = api()
    calls = []
    cfg = Config(max_cycle=7, conv_tol=1e-10, conv_tol_density=1e-10, conv_tol_grad=1e-10)
    result, _, _ = constant_problem(.31, "restricted", config=cfg, calls=calls)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert bool(result.converged) and int(result.cycles) == 2
    assert len(calls) == 2
    assert result.history.density_rms.shape == (7,)
    np.testing.assert_array_equal(result.history.density_rms[2:], 0.)


def test_zero_energy_tolerance_preserves_fixed_cycle_explicit_execution():
    Config, _, _, _ = api()
    calls = []
    cfg = Config(max_cycle=4, conv_tol=0., conv_tol_density=1e-10, conv_tol_grad=1e-10)
    result, _, _ = constant_problem(.31, "restricted", config=cfg, calls=calls)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert not bool(result.converged) and int(result.cycles) == 4
    assert len(calls) == 4
    assert not bool(jnp.any(result.history.converged))


@pytest.mark.parametrize("energy_only", [False, True])
def test_zero_density_threshold_preserves_training_fixed_cycle_contract(energy_only):
    Config, _, _, _ = api()
    cfg = Config(max_cycle=4, conv_tol=1e-10, conv_tol_density=0.,
                 conv_tol_grad=1e-10, energy_only=energy_only)
    result, _, _ = constant_problem(.31, "restricted", config=cfg)
    assert bool(result.converged) == energy_only
    assert int(result.cycles) == (2 if energy_only else 4)


@pytest.mark.parametrize("store", [False, True])
def test_best_iterate_and_optional_history_do_not_change_final_state(store):
    Config, State, _, solve = api()
    initial = State(jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.zeros(1),
                    jnp.asarray(0.), jnp.full((1, 1), 3.), jnp.asarray(0))

    def evaluate(dm, c, eps, previous):
        cycle = previous.aux+1
        return jnp.sum(dm**2), jnp.full((1, 1), jnp.where(cycle == 1, 3.5, 1.)), cycle

    result = solve(initial, evaluate=evaluate,
        diagonalize=lambda f: (jnp.zeros(1), f), density_from_coeff=lambda c: c,
        diis_error=lambda f, state: jnp.zeros_like(f),
        orbital_gradient=lambda state: jnp.asarray(1.),
        config=Config(max_cycle=3, conv_tol=0., diis_space=1, store_iterates=store))
    np.testing.assert_allclose(result.history.density_rms, [3., .5, 2.5], atol=0, rtol=0)
    np.testing.assert_allclose(result.state.density, [[1.]], atol=0, rtol=0)
    np.testing.assert_allclose(result.best_state.density, [[3.5]], atol=0, rtol=0)
    assert int(result.best_cycle) == 2
    if store:
        density, coeff, energies = result.history.iterates
        assert density.shape == coeff.shape == (3, 1, 1)
        assert energies.shape == (3, 1)
    else:
        assert result.history.iterates == ()


def test_fresh_fock_recheck_can_reject_incremental_convergence():
    Config, State, gradient, solve = api()
    diagonal = jnp.diag(jnp.array([-1., 1.]))
    occupation = jnp.array([1., 0.])
    initial = State(jnp.diag(occupation), jnp.eye(2), jnp.diag(diagonal),
                    jnp.asarray(-1.), diagonal)
    calls, gradient_calls = [], []

    def evaluate(dm, c, eps, previous):
        fresh = previous is None
        jax.debug.callback(lambda: calls.append(fresh), ordered=True)
        raw = diagonal+jnp.array([[0., .2], [.2, 0.]]) if fresh else diagonal
        return jnp.asarray(-1.), raw, ()

    def measure_gradient(state):
        jax.debug.callback(lambda: gradient_calls.append(1), ordered=True)
        return gradient(state.fock, state.mo_coeff, occupation)

    cfg = Config(max_cycle=3, conv_tol=1e-10, conv_tol_density=1e-10,
                 conv_tol_grad=1e-10, diis_space=1, refresh_on_convergence=True)
    kwargs = dict(evaluate=evaluate, diagonalize=jnp.linalg.eigh,
        density_from_coeff=lambda c: (c*occupation) @ c.T,
        diis_error=lambda f, state: f @ state.density-state.density @ f,
        orbital_gradient=measure_gradient)
    result = solve(initial, config=cfg, **kwargs)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert not bool(result.converged)
    assert int(result.cycles) == 3
    assert float(result.history.gradient[1]) > .1
    assert calls == [False, False, True, False]
    assert len(gradient_calls) == 4  # Three candidates and one actual fresh rebuild.
    ordinary = solve(initial, config=replace(cfg, refresh_on_convergence=False), **kwargs)
    assert bool(ordinary.converged) and int(ordinary.cycles) == 2


def test_initial_last_fock_controls_only_requested_first_step_damping():
    Config, State, gradient, solve = api()
    raw = jnp.diag(jnp.array([-1., 1.]))
    previous = raw+jnp.array([[0., .8], [.8, 0.]])
    occ = jnp.array([1., 0.])
    initial = State(jnp.diag(occ), jnp.eye(2), jnp.diag(raw), jnp.asarray(-1.), raw)
    cfg = Config(max_cycle=1, conv_tol=0., damping=.25)
    kwargs = dict(evaluate=lambda dm, c, eps, old: (jnp.sum(dm*raw.T), raw, ()),
        diagonalize=jnp.linalg.eigh, density_from_coeff=lambda c: (c*occ) @ c.T,
        diis_error=lambda f, state: f @ state.density-state.density @ f,
        orbital_gradient=lambda state: gradient(state.fock, state.mo_coeff, occ),
        initial_last_fock=previous)
    result = solve(initial, config=cfg, **kwargs)
    _, c = np.linalg.eigh(np.asarray(.75*raw+.25*previous))
    np.testing.assert_allclose(result.state.density, (c*np.asarray(occ)) @ c.T, atol=1e-13, rtol=0)
    skipped = solve(initial, config=replace(cfg, skip_first_fock_damping=True), **kwargs)
    np.testing.assert_array_equal(skipped.state.density, initial.density)


def test_energy_change_hook_can_avoid_cancelling_a_constant_energy_offset():
    h = jnp.array([[-1., .31], [.31, 1.]])
    stable_delta = lambda old, new: jnp.sum((new.density-old.density)*h.T)
    result, _, _ = constant_problem(.31, "restricted", energy_offset=1e16,
                                     energy_change=stable_delta)
    expected = 2*(np.sqrt(1+.31**2)-1)
    np.testing.assert_allclose(result.history.energy_change[0], expected, atol=1e-13, rtol=0)
    assert bool(result.converged) and int(result.cycles) == 2


@pytest.mark.parametrize("bad_energy", [jnp.nan, jnp.inf])
def test_finite_energy_change_hook_cannot_accept_nonfinite_total_energy(bad_energy):
    result, _, _ = constant_problem(.31, "restricted", energy_offset=bad_energy,
        energy_change=lambda old, new: jnp.asarray(0.))
    assert not bool(result.converged)
    assert int(result.cycles) == 5
    assert not bool(jnp.isfinite(result.state.energy))


def density_sequence(values):
    """A prescribed map isolates best-iterate selection from invalid eigensolves."""
    Config, State, _, solve = api()
    sequence = jnp.asarray(values)
    initial = State(jnp.zeros((1, 1)), jnp.full((1, 1), -1.), jnp.zeros(1),
                    jnp.asarray(0.), jnp.zeros((1, 1)), jnp.asarray(0))

    def evaluate(dm, c, eps, previous):
        cycle = previous.aux+1
        return cycle.astype(jnp.float64), jnp.full((1, 1), cycle, dtype=jnp.float64), cycle

    return solve(initial, evaluate=evaluate,
        diagonalize=lambda f: (jnp.zeros(1), f),
        density_from_coeff=lambda c: jnp.full((1, 1), sequence[c[0, 0].astype(jnp.int32)]),
        diis_error=lambda f, state: jnp.zeros_like(f),
        orbital_gradient=lambda state: jnp.asarray(0.),
        config=Config(max_cycle=len(values), conv_tol=0., diis_space=1))


@pytest.mark.parametrize("bad_density", [jnp.nan, jnp.inf])
def test_all_invalid_density_steps_return_the_actual_failed_state(bad_density):
    result = density_sequence([bad_density, bad_density, bad_density])
    assert not bool(result.converged)
    assert int(result.cycles) == int(result.best_cycle) == 3
    assert int(result.best_state.aux) == 3
    np.testing.assert_array_equal(result.best_state.mo_coeff, result.state.mo_coeff)
    assert not bool(jnp.any(jnp.isfinite(result.history.density_rms)))
    assert not bool(jnp.all(jnp.isfinite(result.best_state.density)))


@pytest.mark.parametrize("bad_density", [jnp.nan, jnp.inf])
def test_invalid_density_steps_do_not_poison_later_finite_best_selection(bad_density):
    result = density_sequence([2., bad_density, 2.5, 3., 3.1])
    assert not bool(result.converged)
    assert int(result.best_cycle) == 5
    assert int(result.best_state.aux) == 5
    np.testing.assert_allclose(result.best_state.density, [[3.1]], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result.history.density_rms[jnp.array([0, 3, 4])],
                               [2., .5, .1], atol=1e-14, rtol=0)
    assert not bool(jnp.any(jnp.isfinite(result.history.density_rms[1:3])))
