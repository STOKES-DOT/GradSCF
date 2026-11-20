"""Replayable epsilon-greedy choices over search actions."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import random
from statistics import mean


ACTION_GUIDANCE = {
    'refine': 'Refine one calibrated structural channel in a valid candidate.',
    'physical_exploration': 'Use a different available physical dependency, such as occupied/virtual coupling.',
    'ablation': 'Remove or replace one specific descriptor in a valid comparator and request a matched refit.',
    'refit': 'Test an explicit fit change such as an affine reparameterization or justified active bound.',
}


@dataclass(frozen=True)
class ActionPolicy:
    seed: int = 20261118
    epsilon: float = .25
    tolerance: float = 1e-8

    def __post_init__(self):
        if not 0 <= self.epsilon <= 1 or self.tolerance <= 0:
            raise ValueError('Invalid exploration probability or plateau tolerance')

    @property
    def actions(self):
        return tuple(ACTION_GUIDANCE)

    @property
    def configuration(self):
        return {'seed': self.seed, 'epsilon': self.epsilon, 'tolerance': self.tolerance,
                'actions': ACTION_GUIDANCE, 'source': sha256(Path(__file__).read_bytes()).hexdigest()}

    def choose(self, round_number, records):
        recent = records[-3:]
        if len(recent) >= 2 and all(
            row.get('status') == 'evaluated' and not (row.get('evaluation') or {}).get('valid')
            for row in recent[-2:]
        ):
            state = 'fitting_failures'
        elif len(recent) == 3 and all(row['policy_reward'] <= self.tolerance for row in recent):
            state = 'plateau'
        else:
            state = 'improving'
        histories = {
            action: [row for row in records if row['decision']['action'] == action]
            for action in self.actions
        }
        estimates = {}
        for action, history in histories.items():
            matched = [row['policy_reward'] for row in history if row['decision']['state'] == state]
            estimates[action] = {'count': len(matched), 'mean': mean(matched) if matched else None}
        rng = random.Random(f'{self.seed}:{round_number}')
        untried = [action for action, history in histories.items() if not history]
        if not any(row['policy_reward'] > 0 or (row.get('evaluation') or {}).get('valid')
                   for row in records):
            action, mode, probability = 'physical_exploration', 'initial_design', 1.
        elif untried:
            action, mode, probability = rng.choice(untried), 'warmup', 1 / len(untried)
        else:
            # Use the pooled action mean only when the current state is unobserved.
            scores = {a: (estimates[a]['mean'] if estimates[a]['count'] else
                          mean(r['policy_reward'] for r in histories[a])) for a in self.actions}
            greedy = max(self.actions, key=scores.__getitem__)
            explore = rng.random() < self.epsilon
            action, mode = (rng.choice(self.actions), 'explore') if explore else (greedy, 'exploit')
            probability = self.epsilon / len(self.actions) + (1 - self.epsilon if action == greedy else 0)
        return {'state': state, 'action': action, 'mode': mode,
                'probability': probability, 'estimates': estimates,
                'instruction': ACTION_GUIDANCE[action],
                'related_rounds': [row['round'] for row in histories[action][-2:] if 'round' in row]}
