"""Analytic candidates are data, not executable agent code."""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def candidate():
    from gradscf.model.autoxc import Candidate, Parameter
    return Candidate('a*n**2', parameters=(Parameter('a', .2, -.5, .5),), kind='energy')


def test_candidate_round_trip_and_parameter_ad():
    from gradscf.model.autoxc import Candidate
    c = candidate()
    restored = Candidate.from_json(c.to_dict())
    assert restored.digest == c.digest
    assert c.formula == 'a*n**2'
    fn = jax.jit(lambda a: jnp.sum(c.evaluate({'a': a}, {'n': jnp.array([1., 2.])})))
    np.testing.assert_allclose(jax.grad(fn)(.2), 5.)


@pytest.mark.parametrize('expression', ['__import__("os")', 'n.__class__', '[n for n in n]',
                                         'open("file")', 'lambda: n', 'n[0]', '2**100000'])
def test_executable_or_unbounded_syntax_is_rejected(expression):
    from gradscf.model.autoxc import Candidate
    with pytest.raises(ValueError):
        Candidate(expression)


def test_candidate_parameters_are_validated_and_frozen():
    from gradscf.model.autoxc import Parameter
    with pytest.raises(ValueError):
        Parameter('n', 0., -1., 1.)
    with pytest.raises(ValueError):
        Parameter('a', 2., -.5, .5)
    with pytest.raises(ValueError):
        Parameter('a', 0., jnp.nan, 1.)


def test_large_constants_and_powers_cannot_execute_python_big_integer_arithmetic():
    from gradscf.model.autoxc import Candidate
    with pytest.raises(ValueError):
        Candidate('9'*400)
    result = Candidate('(((2**8)**8)**8)**8').evaluate({}, {})
    assert np.isinf(result)


def test_scalar_energy_uses_existing_custom_xc_contract():
    c = candidate()
    model = c.functional(lambda inputs: {'n': jnp.array([1., 2.]), 'weights': jnp.ones(2)})
    np.testing.assert_allclose(model.apply({'a': .2}, {'n': jnp.array([1., 2.]), 'weights': jnp.ones(2)}), 1.)
    from gradscf.model.autoxc import Candidate
    with pytest.raises(ValueError, match='potential'):
        Candidate('n', kind='potential').functional(lambda x: x)


def test_import_does_not_start_or_require_codex():
    from gradscf.model import autoxc
    assert callable(autoxc.run)
    assert 'openai_codex' not in sys.modules
