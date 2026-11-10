"""Failed spin iterations retain matching state/cycle/RMS diagnostics."""
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("selection", ["best_rms", "first_converged"])
@pytest.mark.parametrize("all_invalid", [False, True])
def test_unrestricted_nonfinite_history_keeps_state_and_metadata_together(monkeypatch, selection, all_invalid):
    from gradscf.scf import iteration
    from gradscf.solvers.scf import SCFState
    from gradscf.solvers.scf.types import SCFHistory, SCFMetrics, SCFResult
    density = jnp.full((2, 1, 1), jnp.nan if all_invalid else 3.)
    state = SCFState(density, jnp.ones((2, 1, 1)), jnp.zeros((2, 1)),
                     jnp.asarray(0.), jnp.zeros((2, 1, 1)))
    rms = jnp.array([jnp.nan, jnp.nan, jnp.nan] if all_invalid else [jnp.nan, .2, .1])
    history = SCFHistory(rms, jnp.ones(3), jnp.ones(3), jnp.zeros(3, bool), ())
    result = SCFResult(state, state, jnp.asarray(3), jnp.asarray(False), jnp.asarray(3),
                       SCFMetrics(jnp.asarray(1.), rms[-1], jnp.asarray(1.)), history)
    monkeypatch.setattr(iteration, "solve_scf", lambda *args, **kwargs: result)
    actual = iteration.run_unrestricted_scf(
        fock_builder=lambda dm, *_: (jnp.zeros_like(dm), jnp.zeros_like(dm), jnp.asarray(0.)),
        density_spin=jnp.ones((2,1,1)), mo_coeff_spin=jnp.ones((2,1,1)),
        mo_occ_spin=jnp.ones((2,1)), mo_energy_spin=jnp.zeros((2,1)),
        overlap=jnp.eye(1), max_cycle=3, damping=0., conv_tol=0.,
        conv_tol_density=1e-8, conv_tol_grad=1e-7, orthogonalization_eps=1e-10,
        iterate_selection=selection)
    assert int(actual[7]) == int(actual[8]) == 3
    np.testing.assert_array_equal(np.asarray(actual[0]), np.asarray(density))
    np.testing.assert_equal(np.asarray(actual[9]), np.asarray(rms[-1]))
    np.testing.assert_equal(np.asarray(actual[10]), np.asarray(rms[-1]))
