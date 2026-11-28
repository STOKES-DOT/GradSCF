"""Reuse neighboring CASSCF Ritz vectors without replacing the gap audit."""

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.mc import casci, orbitals
from gradscf.mc.csf import make_csf_space

from .test_casscf import molecular_case


def test_recycled_guard_keeps_lowest_root_discoverable_after_crossing(monkeypatch):
    space = make_csf_space(2, (1, 1), 0)
    eri = jnp.zeros((2, 2, 2, 2))
    old_h = jnp.diag(jnp.array([-1.0, 1.0]))
    new_h = -old_h
    call = casci.solve_hermitian
    seeds = []

    def record(*args, **kwargs):
        seeds.append(kwargs["initial_vectors"])
        return call(*args, **kwargs)

    monkeypatch.setattr(casci, "solve_hermitian", record)
    first, guard = casci._solve_casci(old_h, eri, space, with_response=False, retain_guard=True)
    assert first.response_valid and guard.valid
    # In the new Hamiltonian the old target is the highest root, and
    # the old guard is unchanged. Their invariant span misses the ground state.
    second, _ = casci._solve_casci(new_h, eri, space, ci0=first.ci_coeff,
                                  guard0=guard, retain_guard=True, with_response=False)
    assert second.response_valid
    np.testing.assert_allclose(second.active_energy, -2., atol=2e-10)
    np.testing.assert_allclose(second.gap, 2., atol=2e-10)
    np.testing.assert_allclose(seeds[-1][:, 0], first.ci_coeff, atol=1e-14)
    assert np.linalg.norm(np.asarray(seeds[-1][:, 1]) - np.asarray(guard.vector)) < .02


def test_unconverged_guard_restarts_from_standard_guess(monkeypatch):
    space = make_csf_space(2, (1, 1), 0)
    h = jnp.diag(jnp.array([-1., 1.]))
    eri = jnp.zeros((2, 2, 2, 2))
    cold, guard = casci._solve_casci(h, eri, space, with_response=False, retain_guard=True)
    invalid = guard._replace(valid=jnp.asarray(False))
    actual = []
    solve = casci.solve_hermitian

    def record(*args, **kwargs):
        actual.append(kwargs["initial_vectors"])
        return solve(*args, **kwargs)

    monkeypatch.setattr(casci, "solve_hermitian", record)
    casci._solve_casci(h, eri, space, ci0=cold.ci_coeff, guard0=invalid, with_response=False)
    casci.solve_casci(h, eri, space, ci0=cold.ci_coeff, with_response=False)
    np.testing.assert_array_equal(actual[0], actual[1])


def test_casscf_recycles_guard_from_accepted_state_and_reports_ci_work(monkeypatch):
    _, inputs, options, _, _ = molecular_case("H2-R")
    real_solve = casci._solve_casci
    guesses = []
    returned = []
    events = []

    def record(*args, **kwargs):
        guesses.append(kwargs.get("guard0"))
        result, guard = real_solve(*args, **kwargs)
        returned.append(guard)
        return result, guard

    monkeypatch.setattr(orbitals, "_solve_casci", record)
    with jax.disable_jit():
        warm = orbitals.solve_casscf(*inputs, **options, max_cycle=1,
                                      with_response=False, workload_callback=events.append)
    assert len(guesses) == 2
    assert guesses[0] is None and guesses[1].valid
    np.testing.assert_array_equal(guesses[1].vector, returned[0].vector)
    assert sum(event["event"] == "apply_end" and event["purpose"] == "initial"
               for event in events) == 2

    guesses.clear()
    returned.clear()
    with jax.disable_jit():
        cold = orbitals.solve_casscf(*inputs, **options, max_cycle=1,
                                      with_response=False, _recycle_ci=False)
    assert all(guess is None for guess in guesses)
    np.testing.assert_allclose(warm.total_energy, cold.total_energy, atol=2e-10, rtol=0.)
    np.testing.assert_allclose(warm.gap, cold.gap, atol=2e-9, rtol=0.)


def test_standalone_forward_casci_preserves_live_residual_jvp():
    space = make_csf_space(3, (1, 1), 0)
    h = jnp.array([[-1., .2, .1], [.2, .03, -.13], [.1, -.13, 1.]])
    eri = jnp.zeros((3, 3, 3, 3))
    config = casci.EigenSolverConfig(maxiter=1)

    def residuals(delta):
        perturbed = h + delta * jnp.eye(3)
        state = casci.solve_casci(perturbed, eri, space, config=config, with_response=False)
        action = space.matvec_symmetric(perturbed, eri, state.ci_coeff)
        reference = jnp.max(jnp.abs(action - state.active_energy * state.ci_coeff))
        return jnp.stack((state.ci_residual, reference))

    values, tangents = jax.jvp(residuals, (jnp.asarray(0.),), (jnp.asarray(1.),))
    np.testing.assert_allclose(values[0], values[1], atol=2e-12, rtol=0.)
    assert abs(tangents[1]) > 1.e-4
    np.testing.assert_allclose(tangents[0], tangents[1], atol=2e-12, rtol=0.)
