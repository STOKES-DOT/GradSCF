from pathlib import Path
import runpy

import numpy as np
import pytest


@pytest.fixture(scope="module")
def optimization_result():
    script = Path("tools/optimize_contraction_coefficients.py")
    assert script.is_file()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(Path("tools").resolve()))
        module = runpy.run_path(str(script))
        return module["optimize_coefficients"](maxiter=30, verbose=False)


def test_gradient_optimization_lowers_h2_energy(optimization_result):
    result = optimization_result
    assert result["optimizer_success"]
    assert result["final_energy_hartree"] < result["initial_energy_hartree"] - 1e-7
    assert abs(result["final_angle_gradient"]) < 1e-8
    initial = np.asarray(result["initial_coefficients"])
    final = np.asarray(result["final_coefficients"])
    np.testing.assert_allclose(np.linalg.norm(final[:2]), np.linalg.norm(initial[:2]), atol=1e-12)
    np.testing.assert_allclose(final[:3], final[3:], atol=1e-14)
    np.testing.assert_allclose(final[[2,5]], 1., atol=0)
    energies = [row["energy_hartree"] for row in result["history"]]
    assert np.all(np.diff(energies) <= 1e-12)
    assert all(row["scf_converged"] for row in result["history"])
    assert result["max_trace_scf_energy_difference"] < 1e-9


def test_optimized_h2_endpoint_energies_match_pyscf(optimization_result):
    pytest.importorskip("pyscf")
    from comparisons.native_experiment_reference import validate_h2_contraction_endpoints

    assert validate_h2_contraction_endpoints(optimization_result)["max_energy_error_hartree"] < 1e-8
