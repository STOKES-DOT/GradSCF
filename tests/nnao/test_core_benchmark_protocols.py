"""Physical invariants for the independent NNAO core benchmark protocols."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import simpson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'comparisons'))
from nnao_core.protocols import (
    atomic_radial_density, counterpoise_energies, polarizability_from_response,
    subset_packed_factors,
)


def test_hydrogen_radial_density_integrates_to_electron_number():
    radius = np.r_[0., np.geomspace(1e-7, 20., 6000)]
    shell = [[0, [0.7, 1.]]]
    density = atomic_radial_density(shell, np.array([[2.]]), radius)
    np.testing.assert_allclose(simpson(density, x=radius), 2., atol=1e-9)
    expected = 2*radius**2*4*(2*.7)**1.5/np.sqrt(np.pi)*np.exp(-1.4*radius**2)
    np.testing.assert_allclose(density, expected, atol=1e-12)


def test_radial_density_is_independent_of_p_orbital_orientation():
    radius = np.linspace(0, 8, 1000)
    shell = [[1, [0.8, 1.]]]
    first = np.diag([1., 0., 0.])
    vector = np.array([1., 2., -3.])/np.sqrt(14.)
    rotated = np.outer(vector, vector)
    np.testing.assert_allclose(atomic_radial_density(shell, first, radius),
                               atomic_radial_density(shell, rotated, radius), atol=1e-12)


def test_counterpoise_correction_uses_nested_fragment_basis_energies():
    result = counterpoise_energies(-10.03, -5., -5., -5.002, -5.003)
    np.testing.assert_allclose(result['uncorrected_interaction_hartree'], -.03, atol=1e-12)
    np.testing.assert_allclose(result['cp_interaction_hartree'], -.025, atol=1e-12)
    np.testing.assert_allclose(result['bsse_hartree'], .005, atol=1e-12)


def test_negative_cp_correction_is_rejected_instead_of_relabelled():
    with pytest.raises(ValueError, match='nested'):
        counterpoise_energies(-10.03, -5., -5., -4.99, -5.)


@pytest.mark.parametrize('indices,expected', [([0, 2], [0, 3, 5]), ([2, 0], [5, 3, 0])])
def test_fragment_packed_df_slicing_preserves_pair_order(indices, expected):
    actual = subset_packed_factors(np.arange(6.)[None], indices, nao=3)
    np.testing.assert_array_equal(actual, np.array(expected)[None])


def test_static_polarizability_sign_and_cartesian_components():
    position = np.zeros((3, 2, 2))
    response = np.zeros_like(position)
    position[0, 0, 1] = position[0, 1, 0] = .4
    response[0, 0, 1] = response[0, 1, 0] = -.8
    tensor = polarizability_from_response(position, response)
    np.testing.assert_allclose(tensor, np.diag([.64, 0., 0.]), atol=1e-12)


def test_asymmetric_polarizability_is_not_silently_symmetrized():
    position = np.zeros((3, 2, 2))
    response = np.zeros_like(position)
    position[0, 0, 1] = position[0, 1, 0] = .4
    response[1, 0, 1] = response[1, 1, 0] = -.8
    with pytest.raises(ValueError, match='symmetric'):
        polarizability_from_response(position, response)


def test_cp_fragments_keep_bound_geometry_when_reference_monomers_are_relaxed():
    from nnao_core.prepare import fragments
    dimer = dict(structure_id='water-dimer', symbols=['O', 'H', 'H']*2,
        coords_angstrom=[[0., 0., 0.], [.96, 0., 0.], [-.24, .93, 0.],
                         [3., 0., 0.], [3.96, 0., 0.], [2.76, .93, 0.]])
    monomer = dict(natoms=3, symbols=['O', 'H', 'H'],
                   coords_angstrom=[[0., 0., 0.], [.97, 0., 0.], [-.24, .94, 0.]])
    groups, deformation = fragments(dimer, [monomer, monomer])
    assert groups == [[0, 1, 2], [3, 4, 5]]
    assert deformation > 1e-5
