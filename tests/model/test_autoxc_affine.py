"""Only provably affine formulas may use a cached AO linear expansion."""
import pytest

from gradscf.model.autoxc import Candidate, Parameter


@pytest.mark.parametrize('formula,expected', [('a*n+b*sigma', True),
    ('a*exp(n)/(1+sigma)', True), ('n+a', True), ('a*b*n', False),
    ('exp(a*n)', False), ('n/(1+a)', False), ('a**2*n', False)])
def test_affine_contract(formula, expected):
    candidate = Candidate(formula, parameters=(Parameter('a', 0., -.5, .5), Parameter('b', 0., -.5, .5)))
    assert candidate.affine is expected
    assert candidate.features <= {'n', 'sigma'}
