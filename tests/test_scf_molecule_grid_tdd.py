"""Fixed regression cases for molecular grid preparation and reuse."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gradscf import gto, scf
from gradscf.scf import builders
from gradscf.scf.inputs import assembly, grid


@pytest.fixture
def tracked_grid(monkeypatch):
    monkeypatch.setattr(grid, "_GRID_AO_INPUT_CACHE", {})
    builds = []
    orders = []

    def build_quadrature(*args, **kwargs):
        builds.append(1)
        return assembly.build_molecular_grid_from_spec(*args, **kwargs)

    def evaluate_ao(*args, **kwargs):
        orders.append(kwargs["deriv"])
        return assembly.evaluate_cartesian_ao_with_derivatives(*args, **kwargs)

    mol = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g")
    options = dict(
        basis_from_molecule_spec=assembly.basis_from_molecule_spec,
        build_molecular_grid_from_spec=build_quadrature,
        evaluate_cartesian_ao_with_derivatives=evaluate_ao,
        spec=mol.to_spec(), basis=mol.basis, max_l=3, grids_level=0,
        needs_ao_laplacian=False,
    )
    return options, builds, orders


def test_quadrature_and_ao_reused_after_scf_budget_changes(tracked_grid):
    options, builds, orders = tracked_grid
    first = grid._cached_grid_ao_input_bundle(**options, grid_block_size=-1,
                                             scf_max_cycle=21, max_memory=128)
    second = grid._cached_grid_ao_input_bundle(**options, grid_block_size=-1,
                                              scf_max_cycle=22, max_memory=256)

    assert len(builds) == 1
    assert orders == [1]
    assert second.coords is first.coords
    assert second.ao is first.ao

    coords = np.array(options["spec"].coords_bohr, copy=True)
    coords[1, 2] += 0.15
    moved = replace(options["spec"], coords_bohr=coords)
    third = grid._cached_grid_ao_input_bundle(
        **{**options, "spec": moved}, grid_block_size=-1, scf_max_cycle=22, max_memory=256,
    )
    assert len(builds) == 2
    assert orders == [1, 1]
    assert not np.array_equal(third.coords, first.coords)


def test_grid_replans_streamed_and_dense_from_the_same_quadrature(tracked_grid):
    options, builds, orders = tracked_grid
    streamed = grid._cached_grid_ao_input_bundle(**options, grid_block_size=64)
    assert streamed.grid_ao is not None and not streamed.ao.size
    dense = grid._cached_grid_ao_input_bundle(**options, grid_block_size=-1)

    assert len(builds) == 1
    assert orders == [1]
    assert dense.coords is streamed.coords
    assert dense.grid_ao is None
    np.testing.assert_allclose(streamed.grid_ao.map(lambda ao, *_: ao, deriv=0),
                               dense.ao, rtol=1e-11, atol=1e-12)


def test_streamed_grid_payload_reuses_cached_empty_arrays(tracked_grid):
    options, _, orders = tracked_grid
    context = grid._prepare_basis_grid_context(**options, grid_block_size=64)
    ao, derivative, laplacian = grid._grid_ao_payload(
        context,
        needs_ao_laplacian=False,
        evaluate_cartesian_ao_with_derivatives=options["evaluate_cartesian_ao_with_derivatives"],
    )
    assert ao is context.grid_ao_bundle.ao
    assert derivative is context.grid_ao_bundle.ao_deriv1
    assert laplacian is None
    assert orders == []


def test_named_meta_gga_energy_requests_only_needed_ao_derivatives(monkeypatch):
    monkeypatch.setattr(grid, "_GRID_AO_INPUT_CACHE", {})
    orders = []
    evaluate = assembly.evaluate_cartesian_ao_with_derivatives

    def evaluate_ao(*args, **kwargs):
        orders.append(kwargs["deriv"])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(assembly, "evaluate_cartesian_ao_with_derivatives", evaluate_ao)
    mol = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g")
    cfg = scf.RKSConfig(xc_spec="tpss", grid_block_size=-1)
    _, energy_inputs = builders.build_restricted_scf_result_from_facade(
        mol.to_spec(), mol=mol, xc="tpss", grids_level=0, max_l=3,
        integral_backend="libcint", geometry_grad_policy="analytic", grid_ao_backend="jax",
        rks_config=cfg, init_guess="hcore", chkfile=None, sap_basis=None,
        init_guess_chkfile_project=None, geometry_is_traced=False,
        build_inputs_fn=assembly.build_rks_integral_inputs,
        run_rks_fn=lambda **_kwargs: SimpleNamespace(), return_inputs=True,
    )
    assert orders == [1]
    assert energy_inputs.ao_laplacian is None

    response_inputs = assembly.build_rks_integral_inputs(atom=mol.to_spec(), basis=mol.basis,
                                                        config=cfg, include_dipole_integrals=False)
    assert orders == [1, 2]
    np.testing.assert_allclose(energy_inputs.ao, response_inputs.ao, rtol=1e-11, atol=1e-12)
    assert response_inputs.ao_laplacian.shape == energy_inputs.ao.shape
