"""Check continuity across an allocation change."""
from copy import deepcopy

import pytest

from gradscf.model.autoxc.workflows.xc100.continuation import check_evaluator, frozen_potential


def test_evaluator_may_change_allocation_but_not_scoring():
    old = {'gradient_tolerance': 1e-6, 'sample_ids': ['001'],
           'fingerprint': 'old',
           'worker': {'cases': {'001': 'source'}, 'execution': {'cpu_masks': [[0]]},
                      'outer_normalization': [0.003, 0.01, 0.002]}}
    new = deepcopy(old)
    new['fingerprint'] = 'new'
    new['worker']['execution'] = {'cpu_masks': [[2]]}
    check_evaluator(old, new)
    new['gradient_tolerance'] = 2e-6
    with pytest.raises(ValueError, match='scoring'):
        check_evaluator(old, new)
    new['gradient_tolerance'] = 1e-6
    new['worker']['cases']['001'] = 'changed'
    with pytest.raises(ValueError, match='scoring'):
        check_evaluator(old, new)


def test_potential_freezes_parameters_in_formula_and_definitions():
    candidate = {'formula': 'q*g', 'definitions': [{'name': 'g', 'expression': 'p*n'}],
                 'kind': 'potential', 'hypothesis': 'probe',
                 'parameters': [{'name': 'p', 'value': 0, 'lower': -1, 'upper': 1},
                                {'name': 'q', 'value': 0, 'lower': -1, 'upper': 1}]}
    frozen = frozen_potential(candidate, {'p': 0.5, 'q': -0.2})
    assert not frozen.parameters
    assert frozen.features == {'n'}
    assert float(frozen.evaluate({}, {'n': 4.})) == pytest.approx(-0.4)
