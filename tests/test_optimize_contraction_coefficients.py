from pathlib import Path
import runpy

import numpy as np


def test_gradient_optimization_lowers_h2_energy(monkeypatch):
    script = Path("tools/optimize_contraction_coefficients.py")
    assert script.is_file()
    monkeypatch.syspath_prepend(str(Path("tools").resolve()))
    module = runpy.run_path(str(script))
    result = module["optimize_coefficients"](maxiter=30, verbose=False)
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
