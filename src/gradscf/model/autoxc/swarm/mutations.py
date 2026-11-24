"""Shared Thompson decisions and durable per-attempt feedback."""
import json
from threading import Lock

from ..history import write_json
from .bandit import ContextualBandit


class MutationHistory:
    def __init__(self, directory, actions, context_size, best_reward):
        self._directory = directory
        self._lock = Lock()
        self._bandit = ContextualBandit(actions, context_size)
        self._best_reward = best_reward
        self._sequence = 0
        events = sorted((json.loads(path.read_text()) for path in directory.glob('*.json')),
                        key=lambda entry: entry['sequence'])
        for event in events:
            if event['sequence'] != self._sequence:
                raise ValueError('Mutation feedback sequence is incomplete or duplicated')
            self._apply(event['trial'])

    def _path(self, trial):
        return self._directory / (
            f"{trial['agent']:03d}-{trial['generation']:06d}-{trial['step']:04d}.json"
        )

    def replay(self, trial):
        path = self._path(trial)
        if not path.is_file():
            return None
        event = json.loads(path.read_text())
        if event['trial']['parent_evaluation_id'] != trial['parent_evaluation_id']:
            raise ValueError('Recorded mutation has a different parent')
        return event

    def select(self, context, legal, parent_reward, rng):
        with self._lock:
            incumbent = self._best_reward
            index = self._bandit.select(context, legal, incumbent-parent_reward, rng)
            return {
                'action_index': index,
                'context': context,
                'parent_reward': parent_reward,
                'incumbent_reward': incumbent,
                'completed_evaluations': (self._bandit.valid_evaluations +
                                          self._bandit.invalid_evaluations),
            }

    def _apply(self, trial):
        selection = trial.get('mutation_selection')
        evaluation = trial.get('evaluation')
        if evaluation is not None:
            if selection is not None:
                self._bandit.observe(selection['action_index'], selection['context'],
                                     trial['reward'] if evaluation['valid'] else None)
            if evaluation['valid']:
                self._best_reward = max(self._best_reward, evaluation['reward'])
        self._sequence += 1

    def commit(self, trial, session_id, rng):
        with self._lock:
            path = self._path(trial)
            if path.exists():
                raise ValueError('Mutation attempt was already committed')
            write_json(path, {'sequence': self._sequence, 'trial': trial,
                              'session_id': session_id, 'rng_state': rng.bit_generator.state})
            self._apply(trial)

    def summary(self):
        return {
            'valid_evaluations': self._bandit.valid_evaluations,
            'invalid_evaluations': self._bandit.invalid_evaluations,
        }
