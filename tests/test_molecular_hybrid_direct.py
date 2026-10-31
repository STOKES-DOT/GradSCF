"""Complete PBE0/original omegaB97X-D geometry response without AO ERIs.

H2/STO-3G uses Cartesian AOs, a level-0 moving grid and float64. Coordinates
are in bohr, gradients in Ha/bohr and Hessian-vector products in Ha/bohr^2.
The full calculation is a small-system oracle, not a production allocation.
"""
from copy import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, gto
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.integrals.plan import IntegralPlan
from gradscf_tools import molecular_response


@pytest.fixture(scope="module", params=["pbe0", "wb97xd"])
def hybrid_reference(request):
    # Keep the actual functional and convergence settings of the Raman gate.
    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True)
    reference = dft.RKS(mol, xc=request.param, grids_level=0, max_cycle=200,
                        conv_tol=1e-12, conv_tol_density=1e-11,
                        conv_tol_grad=1e-10).run()
    assert reference.converged
    return reference


def _forbid_eri(monkeypatch):
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == "eri":
            raise AssertionError("Direct hybrid geometry must not assemble full or LR AO ERIs")
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, "evaluate", checked)


def test_hybrid_direct_inputs_preserve_geometry_and_controls(hybrid_reference, monkeypatch):
    reference = copy(hybrid_reference)
    reference.max_memory = 512.
    reference.grid_block_size = 64
    reference.grid_max_memory = 16.
    reference.direct_scf_workers = 2
    reference.direct_scf_tol = 1e-8
    build = molecular_response.build_rks_integral_inputs
    configurations = []

    def record_build(**kwargs):
        configurations.append(kwargs.get("config"))
        return build(**kwargs)

    monkeypatch.setattr(molecular_response, "build_rks_integral_inputs", record_build)
    _forbid_eri(monkeypatch)
    for maximum, expected in ((None, reference.max_memory), (384., 384.)):
        energy = molecular_response.MolecularEnergy(
            reference, jk_backend="direct", max_memory=maximum)
        coordinates = energy.coordinates
        inputs = energy.inputs(coordinates)
        config = configurations[-1]
        assert config is not None
        assert config.xc_spec == reference.xc
        assert config.jk_backend == "direct"
        assert config.direct_scf_tol == 0.
        assert config.direct_scf_workers == reference.direct_scf_workers
        assert config.max_memory == expected
        assert config.grid_block_size == reference.grid_block_size
        assert config.grid_max_memory == reference.grid_max_memory
        assert inputs["eri"] is None or inputs["eri"].size == 0
        assert inputs.get("eri_lr") is None or inputs["eri_lr"].size == 0
        provider = inputs["direct_basis"]
        assert type(provider) is NativeDirectBasis
        np.testing.assert_array_equal(provider.parameters.nuclear_coords, coordinates)
        np.testing.assert_array_equal(provider.parameters.centers, coordinates[energy.owners])
        grid = inputs["grid_ao"]
        assert grid.block_size == reference.grid_block_size
        direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
        shifted_coordinates = coordinates + .01*direction
        shifted = energy.inputs(shifted_coordinates)
        np.testing.assert_array_equal(shifted["direct_basis"].parameters.nuclear_coords,
                                      shifted_coordinates)
        assert not np.array_equal(shifted["overlap"], inputs["overlap"])
        assert not np.array_equal(shifted["grid_ao"].coords, grid.coords)
        assert not np.array_equal(shifted["grid_weights"], inputs["grid_weights"])
        if reference.xc == "wb97xd":
            assert np.isfinite(float(inputs["dispersion_energy"]))
            assert float(shifted["dispersion_energy"]) != float(inputs["dispersion_energy"])
        else:
            assert inputs.get("dispersion_energy") is None
    assert reference._config().jk_backend == "full"
    assert reference.direct_scf_tol == 1e-8
    assert reference.max_memory == 512.


def test_hybrid_direct_geometry_gradient_and_hvp_match_full(hybrid_reference, monkeypatch):
    full = molecular_response.MolecularEnergy(hybrid_reference, gradient_tolerance=1e-10)
    direct = molecular_response.MolecularEnergy(
        hybrid_reference, jk_backend="direct", gradient_tolerance=1e-10)
    coordinates = full.coordinates
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    full_calculate = jax.jit(jax.value_and_grad(full))
    full_gradient = jax.jit(jax.grad(full))
    full_state = jax.jit(full.solve)(coordinates)
    expected_value, expected_gradient = full_calculate(coordinates)
    expected_hvp = jax.jit(lambda r: jax.jvp(
        full_gradient, (r,), (direction,))[1])(coordinates)
    assert bool(full_state.stationary)
    np.testing.assert_allclose(expected_value, hybrid_reference.e_tot, atol=2e-9, rtol=0)
    full_differences = []
    for step in (3e-4, 1e-4):
        plus = full_calculate(coordinates + step*direction)
        minus = full_calculate(coordinates - step*direction)
        full_differences.append(((plus[0]-minus[0])/(2*step),
                                 (plus[1]-minus[1])/(2*step)))

    _forbid_eri(monkeypatch)
    state = jax.jit(direct.solve)(coordinates)
    calculate = jax.jit(jax.value_and_grad(direct))
    gradient = jax.jit(jax.grad(direct))
    value, observed_gradient = calculate(coordinates)
    observed_hvp = jax.jit(lambda r: jax.jvp(
        gradient, (r,), (direction,))[1])(coordinates)
    assert bool(state.stationary)
    for response in (value, expected_gradient, expected_hvp, observed_gradient, observed_hvp):
        assert np.isfinite(np.asarray(response)).all()
    np.testing.assert_allclose(value, expected_value, atol=2e-10, rtol=0)
    np.testing.assert_allclose(state.density_matrix, full_state.density_matrix, atol=2e-8, rtol=0)
    np.testing.assert_allclose(state.fock_matrix, full_state.fock_matrix, atol=2e-9, rtol=0)
    np.testing.assert_allclose(observed_gradient, expected_gradient, atol=3e-8, rtol=2e-7)
    np.testing.assert_allclose(observed_hvp, expected_hvp, atol=3e-7, rtol=2e-6)
    np.testing.assert_allclose(observed_gradient.sum(axis=0), 0., atol=2e-8, rtol=0)
    for step, (full_energy_fd, full_gradient_fd) in zip((3e-4, 1e-4), full_differences):
        plus = calculate(coordinates + step*direction)
        minus = calculate(coordinates - step*direction)
        energy_fd = (plus[0]-minus[0])/(2*step)
        gradient_fd = (plus[1]-minus[1])/(2*step)
        for observed, reference in ((jnp.sum(expected_gradient*direction), full_energy_fd),
                                    (jnp.sum(observed_gradient*direction), energy_fd)):
            np.testing.assert_allclose(observed, reference, atol=3e-6, rtol=2e-5)
        for observed, reference in ((expected_hvp, full_gradient_fd),
                                    (observed_hvp, gradient_fd)):
            np.testing.assert_allclose(observed, reference, atol=4e-6, rtol=2e-5)
