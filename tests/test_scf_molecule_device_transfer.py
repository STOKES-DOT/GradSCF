"""Shared grid leaves stay shared across an explicit device transfer."""

import jax
import numpy as np
import pytest

from gradscf.integrals.grids.blocks import AOGrid
from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule, UnrestrictedMolecule
from gradscf.tools.device import put_molecule_on_device


@pytest.mark.parametrize("molecule_type", [RestrictedMolecule, UnrestrictedMolecule])
def test_reference_and_streamed_provider_share_moved_grid_leaves(molecule_type):
    cpus = jax.devices("cpu")
    target = cpus[-1]
    coords = np.array([[0., 0., 0.], [.23, .11, .31], [.42, .44, .37]])
    weights = np.array([.27, .44, .15])
    # Two CPU devices exercise a real device-to-device transfer. A NumPy source
    # still exercises the two shared leaves in an ordinary single-device run.
    if len(cpus) > 1:
        coords, weights = jax.device_put((coords, weights), cpus[0])
    provider = AOGrid(
        coords=coords,
        weights=weights,
        centers=np.zeros((1, 3)),
        exponents=np.ones((1, 1)),
        coefficients=np.ones((1, 1)),
        nprims=(1,),
        angulars=((0, 0, 0),),
        block_size=2,
        adaptive=True,
    )
    molecule = molecule_type(
        ao=np.zeros((0, 1)),
        grid=QuadratureGrid(weights=weights, coords=coords),
        grid_ao=provider,
        dipole_integrals=np.zeros((3, 1, 1)),
        rep_tensor=np.zeros((0, 0, 0, 0)),
        mo_coeff=np.ones((2, 1, 1)),
        mo_occ=np.ones((2, 1)),
        mo_energy=np.zeros((2, 1)),
        rdm1=np.ones((2, 1, 1)),
        h1e=np.zeros((1, 1)),
        nuclear_repulsion=0.,
    )
    moved = put_molecule_on_device(molecule, device=target)
    assert moved.grid.coords is moved.grid_ao.coords
    assert moved.grid.weights is moved.grid_ao.weights
    np.testing.assert_array_equal(moved.grid.coords, coords)
    np.testing.assert_array_equal(moved.grid.weights, weights)
    assert all(leaf.devices() == {target} for leaf in jax.tree.leaves(moved.grid_ao))
    assert moved.grid_ao.block_size == provider.block_size
    assert moved.grid_ao.adaptive is provider.adaptive
    assert molecule.grid_ao is provider and molecule.grid.coords is coords
