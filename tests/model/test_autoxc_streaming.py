"""Whole-dataset fitting may stream sample gradients without changing the mean."""
import jax.numpy as jnp
import numpy as np

from gradscf.dft.iks import WuYangScore
from gradscf.model.autoxc import Candidate, Evaluator, Parameter


def test_streamed_and_joint_objectives_fit_the_identical_full_batch():
    def score(p, optimum):
        reward = -(p['a']-optimum)**2
        return WuYangScore(0., 0., reward, 0., jnp.eye(2), jnp.eye(2), 1., 0., jnp.nan, True)
    bind = lambda c: {'first': lambda p: score(p, .2), 'second': lambda p: score(p, .4)}
    candidate = Candidate('a*n', parameters=(Parameter('a', 0., -.5, .5),))
    results = [Evaluator(bind, sample_ids=('first', 'second'), fingerprint='batch-v1',
                         streaming=mode).evaluate(candidate) for mode in (False, True)]
    for result in results:
        assert result.valid
        np.testing.assert_allclose(result.parameters['a'], .3, atol=1e-8)
        np.testing.assert_allclose(result.reward, -.01, atol=1e-12)
    assert results[0].parameters == results[1].parameters


def test_streaming_preserves_rejection_of_any_failed_training_sample():
    def score(p, valid):
        return WuYangScore(0., 0., -(p['a']-.2)**2, 0., jnp.eye(2), jnp.eye(2), 1., 0., jnp.nan, valid)
    evaluator = Evaluator(lambda c: {'ok': lambda p: score(p, True), 'bad': lambda p: score(p, False)},
        sample_ids=('ok', 'bad'), fingerprint='batch-v1', streaming=True)
    result = evaluator.evaluate(Candidate('a*n', parameters=(Parameter('a', 0., -.5, .5),)))
    assert not result.valid and result.reward is None
