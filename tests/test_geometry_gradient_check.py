import pytest

from pathlib import Path
import runpy


@pytest.fixture(scope="module")
def geometry_result():
    check = runpy.run_path(str(Path("tools/check_geometry_gradients.py")))["check_geometry"]
    return check("h2")


def test_total_geometry_gradient_matches_fd(geometry_result):
    assert geometry_result["max_ad_fd_error"] < 1e-7
    assert geometry_result["max_scf_commutator"] < 1e-8


def test_total_geometry_gradient_matches_pyscf(geometry_result):
    pytest.importorskip("pyscf")
    from comparisons.native_experiment_reference import validate_geometry

    assert validate_geometry(geometry_result)["max_ad_pyscf_error"] < 1e-7
