"""Stoichiometry, electronic states and energy conventions for benchmarks."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'comparisons'))
from nnao_reactivity.protocols import parse_din, parse_xyz, reaction_profile, validate_reaction


def test_din_keeps_signed_stoichiometry_and_labels():
    rows = parse_din('# source\n1\nTS\n-2\nH\n0\n10.2 forward\n')
    assert rows == [dict(terms={'TS': 1, 'H': -2}, reference_kcal_mol=10.2, label='forward')]


def test_din_does_not_accept_an_unterminated_reaction():
    with pytest.raises(ValueError):
        parse_din('1\nTS\n-1\nH\n')


def test_xyz_uses_multiplicity_instead_of_assuming_closed_shell():
    row = parse_xyz('2\n0 2\nO 0 0 0\nH 0 0 1\n', 'OH')
    assert row['spin'] == 1
    assert row['nelectron'] == 9
    assert row['charge'] == 0


def test_xyz_requires_explicit_electronic_state():
    with pytest.raises(ValueError, match='charge'):
        parse_xyz('1\nunknown state\nH 0 0 0\n', 'H')


def test_reaction_profiles_sum_all_fragments_and_keep_reverse_barrier():
    result = reaction_profile({'R1': 1, 'R2': 2}, 'TS', {'P': 1},
                              {'R1': -10., 'R2': -1., 'TS': -11.98, 'P': -12.03})
    np.testing.assert_allclose(result['relative_hartree'], [0., .02, -.03], atol=1e-13)
    np.testing.assert_allclose(result['forward_kcal_mol']-result['reverse_kcal_mol'],
                               result['reaction_kcal_mol'], atol=1e-10)


def test_submerged_barrier_is_not_clamped_or_drawn_above_reactants():
    result = reaction_profile({'R': 1}, 'TS', {'P': 1}, {'R': -1., 'TS': -1.01, 'P': -1.02})
    assert result['forward_kcal_mol'] < 0


def test_reaction_charge_and_element_balance_are_checked():
    h = dict(symbols=['H'], charge=0)
    h2 = dict(symbols=['H', 'H'], charge=0)
    validate_reaction({'H': 2}, 'TS', {'H2': 1}, {'H': h, 'H2': h2, 'TS': h2})
    with pytest.raises(ValueError, match='balanced'):
        validate_reaction({'H': 1}, 'TS', {'H2': 1}, {'H': h, 'H2': h2, 'TS': h2})
    with pytest.raises(ValueError, match='balanced'):
        validate_reaction({'H': 2}, 'TS', {'H2': 1},
                          {'H': h, 'H2': dict(h2, charge=1), 'TS': h2})


def test_published_forward_reverse_and_reaction_references_close_with_rounding():
    from nnao_reactivity.prepare import assemble_bh9
    rows = [dict(label='BH9_1_forward', terms={'BH9_01_1TS': 1, 'R1': -1, 'R2': -2}, reference_kcal_mol=11.74),
            dict(label='BH9_1_reverse', terms={'BH9_01_1TS': 1, 'P': -1}, reference_kcal_mol=9.01),
            dict(label='BH9-RE_1', terms={'P': 1, 'R1': -1, 'R2': -2}, reference_kcal_mol=2.74)]
    result, = assemble_bh9(rows)
    assert result['reactants'] == {'R1': 1, 'R2': 2}
    assert abs(result['reference_closure_kcal_mol']) < .0100001
    with pytest.raises(ValueError, match='inconsistent'):
        assemble_bh9(rows[:2]+[dict(rows[2], reference_kcal_mol=4.)])


def test_cp_partition_checks_rigid_monomer_geometry_instead_of_only_formula():
    from nnao_reactivity.prepare import fragment_indices
    monomer = dict(natoms=2, symbols=['H', 'H'], coords_angstrom=[[0., 0., 0.], [0., 0., .74]])
    dimer = dict(natoms=4, symbols=['H']*4,
                 coords_angstrom=[[0., 0., 0.], [.74, 0., 0.], [0., 3., 0.], [0., 3., .74]])
    assert fragment_indices(dimer, [monomer, monomer]) == [[0, 1], [2, 3]]
    stretched = dict(monomer, coords_angstrom=[[0., 0., 0.], [0., 0., .8]])
    with pytest.raises(AssertionError):
        fragment_indices(dimer, [stretched, monomer])


def test_bse_general_contractions_and_sp_channels_preserve_original_columns():
    from nnao_reactivity.protocols import bse_element_shells
    element = dict(electron_shells=[
        dict(angular_momentum=[0], exponents=['2.', '1.'], coefficients=[['.4', '.6'], ['.7', '.8']]),
        dict(angular_momentum=[0, 1], exponents=['.5'], coefficients=[['.9'], ['1.1']])])
    assert bse_element_shells(element) == [
        [0, [2., .4, .7], [1., .6, .8]], [0, [.5, .9]], [1, [.5, 1.1]]]


def test_bse_ecp_is_rejected_for_an_all_electron_benchmark():
    from nnao_reactivity.protocols import bse_element_shells
    with pytest.raises(ValueError, match='ECP'):
        bse_element_shells(dict(ecp_electrons=28, electron_shells=[]))
