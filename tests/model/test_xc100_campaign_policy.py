"""The 50-round policy is fixed independently of model-provided coefficients."""
import pytest

from gradscf.model.autoxc import Candidate, Parameter


def test_valid_affine_uniform_limit_correction_passes():
    from gradscf.model.autoxc.workflows.xc100.campaign import validate_candidate
    validate_candidate(Candidate('n**0.3333333333333333*(a*u+b*u**2)', kind='potential',
        parameters=(Parameter('a', 0., -.2, .2), Parameter('b', 0., -.2, .2))))


@pytest.mark.parametrize('formula,bounds', [('a*n',(-.2,.2)), ('exp(a)*u',(-.2,.2)),
    ('a*n*u',(-1.,1.)), ('a*tau*u',(-.2,.2)), ('a*u/(n-n)',(-.2,.2))])
def test_invalid_shape_constraints_or_bounds_are_rejected(formula,bounds):
    from gradscf.model.autoxc.workflows.xc100.campaign import validate_candidate
    with pytest.raises(ValueError):
        validate_candidate(Candidate(formula,parameters=(Parameter('a',0.,*bounds),)))
