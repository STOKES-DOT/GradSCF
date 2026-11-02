"""The A/B pilot records reference restrictions, weights and complete results."""
import json

import numpy as np
import pytest

from gradscf.model.nnao.workflows.pool_calibration import load_references, run_pilot


def test_default_references_are_explicit_neutral_rhf_problems():
    data = load_references()
    assert [x['structure_id'] for x in data['structures']] == ['H2', 'CH4', 'NH3', 'H2O', 'HF', 'CO']
    assert all(x['spin'] == 0 and x['charge'] == 0 and x['weight'] > 0 for x in data['structures'])
    assert data['method'] == 'RHF' and data['units'] == 'Angstrom'


@pytest.mark.parametrize('field,value', [('basis_family', 'szp442_direct'), ('core_primitives', 3)])
def test_reference_metadata_cannot_silently_change_the_basis(tmp_path, field, value):
    data = load_references()
    data[field] = value
    path = tmp_path/'references.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='szp663|core6'):
        load_references(path)


def test_small_pilot_records_pool_free_contraction_and_recoverable_parameters(tmp_path):
    data = load_references(ids=['H2'])
    result = run_pilot(data, tmp_path, calibration_steps=0, contraction_steps=1, starts=1)
    assert result['status'] == 'complete'
    record = result['comparisons'][0]
    assert record['pool_nao'] == 24 and record['contracted_nao'] == 8
    assert record['delta_contraction_hartree'] >= -1e-7
    assert record['network_energy_hartree'] is None
    assert result['selected_initial_pool'] is None  # A short pilot is not final calibration.
    assert (tmp_path/'REPORT.md').is_file()
    with np.load(tmp_path/record['checkpoint']) as saved:
        metadata = json.loads(str(saved['metadata'].item()))
        assert metadata['kind'] == 'free_contraction'
        assert np.isfinite(saved['parameters']).all() and saved['optimizer'].size > 0
    with pytest.raises(ValueError, match='empty|exists'):
        run_pilot(data, tmp_path, calibration_steps=0, contraction_steps=1, starts=1)


def test_reference_checkpoint_restores_the_next_adam_update(tmp_path):
    from gradscf.model.nnao.training.pool import minimize_reference
    from gradscf.model.nnao.training.trainer import Adam
    from gradscf.model.nnao.workflows.pool_calibration import _save_result
    evaluate = lambda x: (float(np.sum(x*x)), 2*x, {'converged': True})
    first = minimize_reference(np.array([1., -.4]), evaluate, max_steps=1)
    path = tmp_path/'reference.npz'
    _save_result(path, first, kind='free_contraction', identity={'fixture': 'quadratic'})
    optimizer = Adam(.01)
    with np.load(path) as saved:
        x, g = saved['parameters'], saved['gradient']
        state = optimizer.decode(saved['optimizer'].tobytes(), optimizer.init(x))
        continued, continued_state = optimizer.update(x, g, state)
    uninterrupted = minimize_reference(np.array([1., -.4]), evaluate, max_steps=2)
    np.testing.assert_array_equal(continued, uninterrupted['parameters'])
    assert optimizer.encode(continued_state) == uninterrupted['optimizer']
