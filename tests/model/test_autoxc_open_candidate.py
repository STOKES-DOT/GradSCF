"""Open analytic candidates retain a safe, differentiable scalar contract."""
import ast
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.model.autoxc import Candidate, Parameter
from gradscf.model.autoxc.candidate import PROPOSAL_SCHEMA


def test_candidate_accepts_more_than_eight_parameters():
    parameters = tuple(Parameter(f'a{i}', .1, -1., 1.) for i in range(10))
    candidate = Candidate('+'.join(f'a{i}*n' for i in range(10)), parameters)
    assert len(candidate.parameters) == 10
    assert candidate.affine
    np.testing.assert_allclose(candidate.evaluate(candidate.initial, {'n': 2.}), 2.)
    assert 'maxItems' not in PROPOSAL_SCHEMA['properties']['parameters']


def test_definitions_round_trip_are_immutable_and_hashable():
    payload = {
        'formula': 'amplitude*shape',
        'parameters': [{'name': 'amplitude', 'value': .2, 'lower': -1., 'upper': 1.}],
        'kind': 'potential',
        'hypothesis': 'Reuse named scalar intermediates.',
        'definitions': [
            {'name': 'density', 'expression': 'n+1'},
            {'name': 'shape', 'expression': 'sqrt(density)+sigma'},
        ],
    }
    candidate = Candidate.from_json(json.dumps(payload))
    assert candidate.definitions == (('density', 'n+1'), ('shape', 'sqrt(density)+sigma'))
    assert candidate.to_dict() == payload
    assert Candidate.from_json(candidate.to_dict()) == candidate
    assert hash(Candidate.from_json(payload)) == hash(candidate)
    assert Candidate.from_json(payload).digest == candidate.digest
    payload['definitions'][0]['expression'] = 'n+2'
    assert candidate.definitions[0] == ('density', 'n+1')
    assert Candidate.from_json(payload).digest != candidate.digest
    assert 'definitions' in PROPOSAL_SCHEMA['properties']
    assert 'definitions' not in PROPOSAL_SCHEMA['required']


def test_definitions_preserve_jit_parameter_and_feature_gradients():
    candidate = Candidate(
        'scaled*scaled', (Parameter('a', .2, -1., 1.),),
        definitions=(('density', 'n+1'), ('scaled', 'a*density')),
    )
    value = jax.jit(lambda a, n: jnp.sum(candidate.evaluate({'a': a}, {'n': n})))
    n = jnp.array([1., 2.])
    np.testing.assert_allclose(value(.2, n), .52)
    np.testing.assert_allclose(jax.grad(value, 0)(.2, n), 5.2)
    np.testing.assert_allclose(jax.grad(value, 1)(.2, n), [.16, .24])
    assert not candidate.affine


def test_features_are_reachable_external_leaves():
    candidate = Candidate(
        'a*shape+extra', (Parameter('a', .2, -1., 1.),),
        definitions=(('unused', 'unused_feature'), ('density', 'n+1'),
                     ('shape', 'sqrt(density)+sigma')),
    )
    assert candidate.features == frozenset({'n', 'sigma', 'extra'})
    assert candidate.affine
    np.testing.assert_allclose(
        candidate.evaluate(candidate.initial, {'n': 3., 'sigma': 1., 'extra': .4}), 1.,
    )


@pytest.mark.parametrize('definitions', [
    (('shape', 'n'), ('shape', 'sigma')),
    (('shape', 'shape+n'),),
    (('first', 'second+n'), ('second', 'n')),
    (('first', 'second+n'), ('second', 'first+sigma')),
    (('a', 'n'),),
    (('n', 'sigma'),),
    (('weights', 'n'),),
    (('exp', 'n'),),
    (('_hidden', 'n'),),
])
def test_definitions_reject_duplicates_cycles_forward_references_and_shadowing(definitions):
    with pytest.raises(ValueError):
        Candidate('n', (Parameter('a', .2, -1., 1.),), definitions=definitions)


@pytest.mark.parametrize('definition', [
    {'name': 'shape'},
    {'name': 'shape', 'expression': 'n', 'extra': 'n'},
    {'name': 'shape', 'expression': ['n']},
])
def test_definition_json_requires_exact_name_expression_fields(definition):
    payload = Candidate('n').to_dict()
    payload['definitions'] = [definition]
    with pytest.raises(ValueError):
        Candidate.from_json(payload)


@pytest.mark.parametrize('definitions', [{}, '', None])
def test_definition_json_requires_an_outer_list(definitions):
    payload = Candidate('n').to_dict()
    payload['definitions'] = definitions
    with pytest.raises(ValueError):
        Candidate.from_json(payload)


def test_definition_json_accepts_an_empty_list():
    candidate = Candidate('n')
    payload = candidate.to_dict()
    payload['definitions'] = []
    restored = Candidate.from_json(payload)
    assert restored == candidate
    assert restored.to_dict() == candidate.to_dict()
    assert restored.digest == candidate.digest


@pytest.mark.parametrize('formula,expected', [
    ('n**a', False),
    ('n**exp(a)', False),
    ('a**b', False),
    ('(a*n)**0', True),
    ('(a*n)**1', True),
    ('a*n**sigma', True),
    ('a*n**(1/3)', True),
    ('a*b/(1+a*a)', False),
])
def test_nonlinear_coefficients_and_dynamic_powers(formula, expected):
    candidate = Candidate(formula, (Parameter('a', .2, -1., 1.),
                                    Parameter('b', .3, -1., 1.)))
    assert candidate.affine is expected
    assert np.isfinite(candidate.evaluate(candidate.initial, {'n': 2., 'sigma': .4}))


def test_parameter_dependent_exponent_through_definition_is_nonaffine():
    candidate = Candidate(
        'n**power', (Parameter('a', .2, -1., 1.),),
        definitions=(('power', '1+a'),),
    )
    assert not candidate.affine
    value = lambda a: candidate.evaluate({'a': a}, {'n': 2.})
    np.testing.assert_allclose(jax.grad(value)(.2), 2.**1.2*np.log(2.))


def test_old_payload_and_digest_are_unchanged_without_definitions():
    candidate = Candidate('a*n**2', (Parameter('a', .2, -.5, .5),), 'energy', 'old')
    payload = {'formula': 'a*n**2',
               'parameters': [{'name': 'a', 'value': .2, 'lower': -.5, 'upper': .5}],
               'kind': 'energy', 'hypothesis': 'old'}
    assert candidate.to_dict() == payload
    digest_payload = {key: value for key, value in payload.items() if key != 'hypothesis'}
    digest_payload['formula'] = ast.dump(ast.parse('a*n**2', mode='eval').body)
    expected = hashlib.sha256(json.dumps(digest_payload, sort_keys=True,
                                         allow_nan=False).encode()).hexdigest()
    assert candidate.digest == expected


def test_large_balanced_formula_has_no_term_count_limit():
    terms = ['a*n']*64
    while len(terms) > 1:
        terms = [f'({terms[i]}+{terms[i+1]})' for i in range(0, len(terms), 2)]
    candidate = Candidate(terms[0], (Parameter('a', .2, -1., 1.),))
    assert candidate.affine
    np.testing.assert_allclose(candidate.evaluate(candidate.initial, {'n': 2.}), 25.6)


@pytest.mark.parametrize('formula', [
    '2**100000', '__import__("os")', 'n.__class__', 'n[0]',
    '+'.join(['n']*3000), 'sqrt('*100+'n'+')'*100,
])
def test_open_candidates_keep_scalar_syntax_and_resource_limits(formula):
    with pytest.raises(ValueError):
        Candidate(formula)
