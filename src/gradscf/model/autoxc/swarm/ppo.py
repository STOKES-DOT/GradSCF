"""Masked categorical PPO for choosing a gene and an edit operation."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Transition:
    state: list
    mask: list
    action: int
    old_log_prob: float
    reward: float


def _log_probs(actor, states, masks):
    return jax.nn.log_softmax(jnp.where(masks, states @ actor, -1e20), axis=-1)


@jax.jit
def _update_epoch(actor, critic, states, masks, actions, old_logs, advantages, targets):
    def objective(weights):
        logs = _log_probs(weights, states, masks)
        ratio = jnp.exp(logs[jnp.arange(actions.size), actions] - old_logs)
        clipped = jnp.clip(ratio, .8, 1.2)
        return -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))

    value_gradient = jax.grad(lambda w: jnp.mean((states @ w - targets)**2))
    return (actor - .02 * jax.grad(objective)(actor),
            critic - .02 * value_gradient(critic))


class MutationPolicy:
    """A small actor and critic updated from one completed on-policy segment."""

    def __init__(self, state_size, action_count, checkpoint=None):
        if checkpoint is None:
            self.actor = jnp.zeros((state_size, action_count))
            self.critic = jnp.zeros(state_size)
        else:
            self.actor = jnp.asarray(checkpoint['actor'], dtype=float)
            self.critic = jnp.asarray(checkpoint['critic'], dtype=float)
            if (self.actor.shape != (state_size, action_count) or self.critic.shape != (state_size,)
                    or not np.isfinite(self.actor).all() or not np.isfinite(self.critic).all()):
                raise ValueError('PPO checkpoint has incompatible dimensions or nonfinite weights')

    def checkpoint(self):
        return {'actor': self.actor.tolist(), 'critic': self.critic.tolist()}

    def probabilities(self, state, mask):
        probabilities = np.exp(np.asarray(_log_probs(self.actor, jnp.asarray(state), jnp.asarray(mask))))
        return probabilities / probabilities.sum()

    def update(self, trajectory):
        if not trajectory:
            return
        states = jnp.asarray([row.state for row in trajectory])
        masks = jnp.asarray([row.mask for row in trajectory])
        actions = jnp.asarray([row.action for row in trajectory])
        old_logs = jnp.asarray([row.old_log_prob for row in trajectory])
        values = np.asarray(states @ self.critic)
        advantages = np.empty_like(values)
        next_value = advantage = 0.
        # The segment ends at a selection or a fixed mutation-budget boundary.
        for i in reversed(range(len(trajectory))):
            delta = trajectory[i].reward + .99 * next_value - values[i]
            advantage = delta + .99 * .95 * advantage
            advantages[i] = advantage
            next_value = values[i]
        targets = jnp.asarray(values + advantages)
        for _ in range(4):
            self.actor, self.critic = _update_epoch(
                self.actor, self.critic, states, masks, actions, old_logs,
                jnp.asarray(advantages), targets,
            )
