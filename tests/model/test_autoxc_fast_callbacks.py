"""Optional compiled callbacks must preserve the full-batch fitting objective."""
import jax.numpy as jnp
import numpy as np

from gradscf.dft.iks import WuYangScore
from gradscf.model.autoxc import Candidate, Evaluator, Parameter


def test_streaming_uses_cached_loss_and_solved_density_diagnostics():
    calls = {'fast':0,'forward':0,'diagnostic':0}
    class Score:
        def __call__(self,p):
            calls['forward'] += 1
            r = -(p['a']-.2)**2
            return WuYangScore(0.,0.,r,0.,jnp.eye(2),jnp.eye(2),1.,0.,jnp.nan,True)
        def loss_and_grad(self,vector):
            calls['fast'] += 1
            return (vector[0]-.2)**2,2*(vector-jnp.asarray([.2]))
        def diagnose_result(self,score):
            calls['diagnostic'] += 1
            return score._replace(density_error=jnp.asarray(.01))
    candidate = Candidate('a*n',parameters=(Parameter('a',0.,-.5,.5),))
    result = Evaluator(lambda c:{'one':Score()},sample_ids=('one',),fingerprint='fast-v1',
                       streaming=True).evaluate(candidate)
    assert result.valid and calls['fast']>0
    assert calls['forward']==1 and calls['diagnostic']==1
    np.testing.assert_allclose(result.parameters['a'],.2,atol=1e-9)
    np.testing.assert_allclose(result.reward,0.,atol=1e-12)
