from pathlib import Path
import runpy

import jax
import numpy as np


def test_iteration_coefficient_jacobian_matches_finite_differences():
    script = Path("tools/scf_coefficient_gradients.py")
    assert script.is_file()
    ns = runpy.run_path(str(script))
    experiment = ns["make_experiment"](steps=4)
    fn, coefficients = experiment["energy_trace"], experiment["coefficients"]
    energies = np.asarray(jax.jit(fn)(coefficients))
    jac = np.asarray(jax.jit(jax.jacrev(fn))(coefficients))
    fd = ns["finite_difference_jacobian"](jax.jit(fn), coefficients, step=1e-4)
    assert energies.shape == (5,)
    assert jac.shape == (5, 6)
    assert np.all(np.isfinite(jac))
    np.testing.assert_allclose(jac, fd, atol=2e-6, rtol=2e-5)
    # Scaling an entire contraction leaves its normalized AO unchanged.
    for indices in experiment["contraction_indices"]:
        np.testing.assert_allclose(jac[:, indices] @ np.asarray(coefficients)[indices], 0., atol=1e-8)


def test_final_trace_agrees_with_existing_rhf_solver():
    script = Path("tools/scf_coefficient_gradients.py")
    assert script.is_file()
    ns = runpy.run_path(str(script))
    experiment = ns["make_experiment"](steps=10)
    coefficients = experiment["coefficients"]
    energies = np.asarray(jax.jit(experiment["energy_trace"])(coefficients))
    result = ns["reference_rhf_result"](experiment)
    assert result.converged
    np.testing.assert_allclose(energies[-1], result.total_energy, atol=1e-9, rtol=0)
def test_small_coefficient_trace_rejects_budget_before_integral_evaluation(monkeypatch):
    import runpy
    from gradscf import integrals
    import pytest

    ns = runpy.run_path('tools/scf_coefficient_gradients.py', run_name='coefficient_trace_test')
    def forbidden(*args, **kwargs):
        pytest.fail('Primitive ERI preflight must precede numerical integral evaluation.')
    monkeypatch.setattr(integrals, 'make_plan', forbidden)
    with pytest.raises(ValueError, match='(?i)(primitive|budget|memory)'):
        ns['make_experiment'](steps=2, max_memory=.0001)
