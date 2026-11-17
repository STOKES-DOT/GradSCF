"""Expensive grid diagnostics are computed after fitting, without changing reward."""
import jax.numpy as jnp

from gradscf.dft.iks import WuYangScore
from gradscf.model.autoxc import Candidate, Evaluator, Parameter


def test_optional_diagnostics_are_called_once_after_parameter_fitting():
    calls = []
    class Score:
        def __call__(self, p):
            reward = -(p['a']-.2)**2
            return WuYangScore(0., 0., reward, 0., jnp.eye(2), jnp.eye(2), 1., 0., jnp.nan, True)
        def diagnose(self, p):
            calls.append(1)
            return self(p)._replace(density_error=jnp.asarray(.01))
    candidate = Candidate('a*n', parameters=(Parameter('a', 0., -.5, .5),))
    result = Evaluator(lambda c: {'sample': Score()}, sample_ids=('sample',), fingerprint='diagnostics').evaluate(candidate)
    assert calls == [1]
    assert result.samples['sample']['density_error'] == .01
