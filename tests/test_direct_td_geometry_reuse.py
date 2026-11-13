"""Shared shell-direct full TD keeps relaxed coordinate AD on CPU.

H2/STO-3G, distances in bohr and energies in Ha. The existing geometry
oracle rebuilds the grid, SCF state and full/direct response at each point.
"""
import jax
import pytest

from gradscf.data.molecule import parse_molecule_spec
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.tddft import execution
from test_direct_excitation_geometry import _check_relaxed_excitation_geometry


@pytest.mark.parametrize("xc_spec", ["pbe", "network"],
                         ids=["rks-pbe-full", "rks-globalnn-full"])
def test_shared_direct_full_td_preserves_relaxed_coordinate_ad(
        monkeypatch, record_property, xc_spec):
    original = execution._cacheable
    native_admissions = []

    def observe(driver):
        accepted = original(driver)
        if type(getattr(driver.molecule, "direct_basis", None)) is NativeDirectBasis:
            native_admissions.append(accepted)
        return accepted

    monkeypatch.setattr(execution, "_cacheable", observe)
    # The default target must also remain explicit while the outer geometry
    # transform presents traced reference arrays to the shared TD program.
    with jax.default_device(jax.devices("cpu")[0]):
        spec = parse_molecule_spec("H 0 0 0; H 0 0 1.6", unit="Bohr", charge=0, spin=0)
        metrics = _check_relaxed_excitation_geometry(
            spec, False, xc_spec, True, basis="sto-3g")

    assert native_admissions, "The geometry oracle must enter native shell-direct TD"
    assert all(native_admissions), "Native shell-direct TD must use the shared program"
    record_property("native_direct_program_admissions", len(native_admissions))
    for name, value in metrics.items():
        record_property(name, value)
