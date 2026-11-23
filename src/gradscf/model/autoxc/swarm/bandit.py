"""Contextual Thompson sampling on fitted mutation gains."""
from math import erfc, exp, pi, sqrt

import numpy as np


class ContextualBandit:
    """Fit a discounted Bayesian linear model for each legal action.

    The fitness model learns signed, valid parent-to-child gains. A separate
    beta posterior estimates numerical validity. Each decision samples these
    models and ranks actions by expected improvement over the incumbent.
    """

    def __init__(self, actions, context_size, *, decay=.98):
        self.decay = decay
        self._prior_precision = np.eye(context_size)
        self._precision = np.zeros((actions, context_size, context_size))
        self._cross = np.zeros((actions, context_size))
        self._squares = np.zeros(actions)
        self._valid = np.zeros(actions)
        self._invalid = np.zeros(actions)
        self.valid_evaluations = 0
        self.invalid_evaluations = 0

    def observe(self, action, context, gain):
        for statistic in (self._precision, self._cross, self._squares,
                          self._valid, self._invalid):
            statistic *= self.decay
        if gain is None:
            self._invalid[action] += 1
            self.invalid_evaluations += 1
        else:
            x = np.asarray(context)
            self._precision[action] += np.outer(x, x)
            self._cross[action] += gain * x
            self._squares[action] += gain**2
            self._valid[action] += 1
            self.valid_evaluations += 1

    def select(self, context, legal, margin, rng):
        x = np.asarray(context)
        actions = np.flatnonzero(legal)
        priorities = []
        for action in actions:
            factor = np.linalg.cholesky(self._prior_precision + self._precision[action])
            whitened = np.linalg.solve(factor, self._cross[action])
            projected = np.linalg.solve(factor, x)
            # Normal-inverse-gamma posterior with a 0.05 reward-scale prior.
            residual = max(0., self._squares[action] - whitened @ whitened)
            shape = 3. + self._valid[action]/2
            scale = 2*.05**2 + residual/2
            sigma = sqrt(scale / rng.gamma(shape))
            mean = rng.normal(projected @ whitened, sigma*np.linalg.norm(projected))
            z = (mean - margin) / sigma
            improvement = sigma * (exp(-z*z/2)/sqrt(2*pi) + z*erfc(-z/sqrt(2))/2)
            valid = rng.beta(1+self._valid[action], 1+self._invalid[action])
            priorities.append(valid*max(improvement, 0.))
        winners = actions[np.asarray(priorities) == max(priorities)]
        return int(rng.choice(winners))
