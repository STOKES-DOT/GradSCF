from pathlib import Path
import runpy


def test_total_geometry_gradient_matches_fd_and_pyscf():
    check = runpy.run_path(str(Path("tools/check_geometry_gradients.py")))["check_geometry"]
    result = check("h2")
    assert result["max_ad_fd_error"] < 1e-7
    assert result["max_ad_pyscf_error"] < 1e-7
    assert result["max_scf_commutator"] < 1e-8
