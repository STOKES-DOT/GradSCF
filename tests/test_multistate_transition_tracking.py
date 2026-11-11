"""Host-side isolated-state tracking; real full-TDDFT X+Y transition charts.

Eight anchors and twelve candidates share no=3 != nv=5. Energies/gap guards
are Hartree. These are algebra/guard tests, not new electronic calculations.
"""
import numpy as np
import pytest

from gradscf_tools import excited


NO, NV = 3, 5


def fixture():
    anchors = np.eye(NO*NV)[:8].reshape(8, NO, NV)
    order = np.array([8, 4, 0, 10, 2, 7, 1, 11, 3, 9, 6, 5])
    phases = np.array([-1., 1., -1., 1., 1., -1., -1., 1., -1., 1., 1., -1.])
    candidates = np.eye(NO*NV)[order].reshape(12, NO, NV)*phases[:, None, None]
    roots = np.array([int(np.flatnonzero(order == root)[0]) for root in range(8)])
    return anchors, candidates, .10+.01*order, roots, phases[roots]


def match(anchors, candidates, energies, **kwargs):
    eye = np.eye(NO+NV)
    return excited.match_transition_states(eye, anchors, eye, candidates, eye,
        energies, nocc=NO, **kwargs)


def test_rectangular_assignment_returns_all_eight_distinct_signed_roots():
    anchors, candidates, energies, roots, phases = fixture()
    result = match(anchors, candidates, energies)
    np.testing.assert_array_equal(result['roots'], roots)
    np.testing.assert_array_equal(result['phases'], phases)
    assert len(set(result['roots'])) == 8
    assert np.shape(result['signed_overlaps']) == (8, 12)
    np.testing.assert_allclose(result['matched_overlap'], 1., atol=2e-14)
    np.testing.assert_allclose(result['margins'], 1., atol=2e-14)
    np.testing.assert_allclose(result['gaps_hartree'], .01, atol=2e-14)


def test_independent_occupied_virtual_rotations_and_reference_signs_are_gauge_changes():
    anchors, candidates, energies, roots, phases = fixture()
    rng = np.random.default_rng(741)
    charts, amplitudes = [], []
    for states in (anchors, candidates):
        occupied, _ = np.linalg.qr(rng.normal(size=(NO, NO)))
        virtual, _ = np.linalg.qr(rng.normal(size=(NV, NV)))
        chart = np.zeros((NO+NV, NO+NV))
        chart[:NO, :NO], chart[NO:, NO:] = occupied, virtual
        charts.append(chart)
        amplitudes.append(np.einsum('ji,sjb,ba->sia', occupied, states, virtual))
    signs = np.array([1., -1., 1., -1., -1., 1., -1., 1.])
    amplitudes[0] *= signs[:, None, None]
    result = excited.match_transition_states(charts[0], amplitudes[0], charts[1],
        amplitudes[1], np.eye(NO+NV), energies, nocc=NO)
    np.testing.assert_array_equal(result['roots'], roots)
    np.testing.assert_array_equal(result['phases'], phases*signs)
    np.testing.assert_allclose(result['matched_overlap'], 1., atol=2e-14)


def test_zero_transition_dipoles_do_not_destroy_a_transition_density_phase():
    anchors, candidates, energies, roots, phases = fixture()
    zero_dipole_operator = np.zeros((3, NO, NV))
    assert not np.any(np.einsum('xia,sia->sx', zero_dipole_operator, candidates))
    result = match(anchors, candidates, energies)
    np.testing.assert_array_equal(result['roots'], roots)
    np.testing.assert_array_equal(result['phases'], phases)


def test_overlap_magnitudes_preserve_existing_single_anchor_contract():
    anchors, candidates, energies, _, _ = fixture()
    eye = np.eye(NO+NV)
    legacy = np.stack([excited.transition_density_overlaps(eye, anchor, eye,
        candidates, eye, nocc=NO) for anchor in anchors])
    result = match(anchors, candidates, energies)
    np.testing.assert_allclose(result['overlaps'], legacy, atol=2e-14)
    np.testing.assert_allclose(abs(result['signed_overlaps']), legacy, atol=2e-14)


def test_physical_metric_norm_supports_nonorthogonal_mo_charts():
    anchors, candidates, energies, roots, phases = fixture()
    rng = np.random.default_rng(94)
    transforms, coefficients, states = [], [], []
    for amplitudes, offset in ((anchors, .7), (candidates, 1.2)):
        basis = np.eye(NO+NV)+.08*rng.normal(size=(NO+NV, NO+NV))
        scales = np.linspace(offset, offset+1., NO+NV)
        transforms.append(basis)
        coefficients.append(np.linalg.solve(basis, np.diag(scales)))
        states.append(amplitudes/(scales[:NO, None]*scales[None, NO:]))
    left, right = transforms
    result = excited.match_transition_states(coefficients[0], states[0],
        coefficients[1], states[1], left.T@right, energies, nocc=NO,
        reference_metric=left.T@left, metric=right.T@right)
    np.testing.assert_array_equal(result['roots'], roots)
    np.testing.assert_array_equal(result['phases'], phases)
    np.testing.assert_allclose(result['matched_overlap'], 1., atol=3e-14)
    assert np.max(result['overlaps']) <= 1.+3e-14


def test_near_degeneracy_with_an_unassigned_candidate_is_rejected():
    anchors, candidates, energies, roots, _ = fixture()
    unassigned = next(index for index in range(12) if index not in roots)
    energies[unassigned] = energies[roots[0]]+1e-6
    with pytest.raises(RuntimeError, match='(?i)(gap|degener)'):
        match(anchors, candidates, energies)


def test_ambiguous_overlap_is_rejected_after_assignment():
    anchors, _, _, _, _ = fixture()
    candidates = np.stack((anchors[0]+anchors[1], anchors[0]-anchors[1]))/np.sqrt(2.)
    with pytest.raises(RuntimeError, match='(?i)(ambiguous|margin)'):
        match(anchors[:2], candidates, np.array([.1, .2]), minimum_overlap=.6)


def test_hungarian_assignment_does_not_accept_a_forced_bad_second_choice():
    anchors, _, _, _, _ = fixture()
    previous = np.stack((anchors[0], .98*anchors[0]+.2*anchors[1]))
    with pytest.raises(RuntimeError, match='(?i)(overlap|ambiguous|margin)'):
        match(previous, anchors[:2], np.array([.1, .2]), minimum_overlap=.1)


def test_more_anchors_than_available_candidates_is_rejected():
    anchors, candidates, energies, _, _ = fixture()
    with pytest.raises(ValueError, match='(?i)(candidate|match|state)'):
        match(anchors, candidates[:7], energies[:7])


def test_self_metrics_must_be_supplied_together():
    anchors, candidates, energies, _, _ = fixture()
    with pytest.raises(ValueError, match='(?i)(metric|both|together)'):
        match(anchors, candidates, energies, reference_metric=np.eye(NO+NV))


def test_zero_transition_amplitude_is_not_a_dark_but_trackable_state():
    anchors, candidates, energies, _, _ = fixture()
    candidates[0] = 0.
    with pytest.raises(ValueError, match='(?i)(norm|nonzero|amplitude)'):
        match(anchors, candidates, energies)


@pytest.mark.parametrize('bad', ['amplitude', 'energy', 'cross_overlap'])
def test_nonfinite_inputs_fail_explicitly(bad):
    anchors, candidates, energies, _, _ = fixture()
    cross = np.eye(NO+NV)
    if bad == 'amplitude':
        candidates[0, 0, 0] = np.nan
    elif bad == 'energy':
        energies[0] = np.nan
    else:
        cross[0, 0] = np.nan
    with pytest.raises(ValueError, match='(?i)(finite|input|overlap|amplitude|ener)'):
        excited.match_transition_states(np.eye(NO+NV), anchors, np.eye(NO+NV),
            candidates, cross, energies, nocc=NO)
