"""Bounded real public workflows; H2/STO-3G, native CPU float64, grid0.

These tests certify assembly/convergence and reuse of a completed SCF state,
not basis/frequency convergence. Physical derivative gates live in the existing
neural training and molecular GW geometry tests named in the probe README.
"""
from pathlib import Path
import runpy

import jax
import numpy as np
import pytest


_workflow = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                              "benchmarks/probes/workflow_reuse/run.py"))


@pytest.fixture(scope="module", params=_workflow["MODES"])
def mean_field(request):
    source = _workflow["mean_field"](request.param).run()
    jax.block_until_ready((source.e_tot, source.mo_energy, source.mo_coeff))
    assert source.converged
    return request.param, source


def test_six_ground_states_keep_completed_state_at_response_boundary(mean_field, monkeypatch):
    from gradscf.scf import builders, facade, gks

    name, source = mean_field
    energy = float(source.e_tot)
    ground_result = source.scf_result
    density = np.asarray(source.make_rdm1()).copy()
    cycles = source.cycles

    def forbidden(*args, **kwargs):
        raise AssertionError("A response consumer must reuse the completed ground-state solve")

    for module, names in ((builders, ("run_rks_from_integrals", "run_uks_from_integrals")),
                          (facade, ("run_rks_from_integrals",)),
                          (gks, ("run_gks_from_integrals",))):
        for attribute in names:
            monkeypatch.setattr(module, attribute, forbidden)
    for full in (False, True):
        if name in ("ghf", "gks"):
            with pytest.raises(NotImplementedError, match="spinor response"):
                _workflow["td"](source, full)
        else:
            response = _workflow["td"](source, full)
            jax.block_until_ready(response.kernel())
            assert response.converged
            assert np.isfinite(response.e).all()
    assert source.scf_result is ground_result
    assert source.cycles == cycles and float(source.e_tot) == energy
    np.testing.assert_array_equal(source.make_rdm1(), density)


@pytest.fixture(scope="module")
def dft_source():
    source = _workflow["mean_field"]("rks").run()
    assert source.converged
    return source


@pytest.fixture(scope="module", params=_workflow["METHODS"])
def quasiparticles(request, dft_source):
    calculation = _workflow["quasiparticles"](dft_source, request.param).run()
    jax.block_until_ready(calculation.result)
    assert calculation.converged
    return request.param, calculation


@pytest.mark.parametrize("full", [False, True], ids=["tda", "full"])
def test_four_gw_drivers_preserve_real_screening_and_bse_capabilities(quasiparticles, dft_source, full):
    name, calculation = quasiparticles
    result = calculation.result
    ground_result = dft_source.scf_result
    if name == "scgw":
        assert result.mo_energy is None
        assert np.isfinite(result.self_energy_iw).all()
        assert float(result.fixed_point_residual) < 1e-6
        assert abs(float(result.particle_number_error)) < 1e-6
        with pytest.raises(NotImplementedError, match="QP|continuation"):
            _workflow["excitons"](calculation, full).run()
    else:
        assert np.asarray(result.qp_computed_mask).all()
        assert np.asarray(result.converged_mask).all()
        if name == "g0w0":
            np.testing.assert_array_equal(result.screening_energy, dft_source.mo_energy)
        elif name == "evgw":
            np.testing.assert_allclose(result.screening_energy, result.mo_energy, atol=2e-7, rtol=0)
        response = _workflow["excitons"](calculation, full).run()
        jax.block_until_ready(response.result)
        assert np.asarray(response.converged).all()
        assert np.asarray(response.result.stable).all()
        assert np.asarray(response.result.screening_valid).all()
        assert np.isfinite(response.e).all()
        np.testing.assert_allclose(response.reference.screening_energy, result.screening_energy, atol=0, rtol=0)
        np.testing.assert_allclose(response.result.residual_norms, 0., atol=2e-8, rtol=0)
        assert np.isfinite(response.oscillator_strength()).all()
    assert dft_source.scf_result is ground_result
