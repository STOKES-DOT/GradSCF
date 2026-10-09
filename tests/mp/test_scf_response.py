"""Complete implicit-HF -> MP chain versus independently reconverged FD."""
import runpy

import numpy as np


def test_full_implicit_scf_gradient_matches_reconverged_finite_difference():
    data = runpy.run_path("examples/mp/implicit_gradient.py")
    assert len(data["records"]) == 2
    for energy, gradient, finite_difference in data["records"]:
        assert np.isfinite(energy) and np.isfinite(gradient)
        np.testing.assert_allclose(gradient, finite_difference, atol=2e-7, rtol=1e-5)
