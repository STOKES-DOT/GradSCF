"""Deterministic split, declared reward and immutable functional-loop records."""
from __future__ import annotations

import csv
import hashlib
import json
from math import isfinite
from pathlib import Path

from .cafes_qm9 import deterministic_split


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def freeze_molecule_split(molecule_ids):
    ids = tuple(str(value) for value in molecule_ids)
    if len(ids) != 30 or len(set(ids)) != 30:
        raise ValueError('The pilot split requires30 distinct molecule IDs')
    split = deterministic_split(ids)
    return split, canonical_hash(split)


def freeze_reward_config(baseline_wrmsd, *, epsilon=1e-12, constraint_names=None, provenance=None):
    """Freeze the declared adaptation before candidate ranking, in Hartree.

    The literature uses reciprocal validation WRMSD with a0.9 factor for each
    failed constraint. This pilot uses uniform molecule weights and calibrates
    its target to0.85 times the actually measured PBE validation baseline.
    Epsilon is an explicit denominator floor, not a missing-metric substitute.
    """
    baseline_wrmsd, epsilon = float(baseline_wrmsd), float(epsilon)
    if not isfinite(baseline_wrmsd):
        raise ValueError('The measured baseline must be finite')
    if baseline_wrmsd <= 0:
        raise ValueError('J_target requires a positive measured PBE baseline')
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Reward epsilon must be positive and finite')
    config = dict(version=1, fitness='validation_energy_WRMSD', units='Hartree',
        molecule_weights='uniform', baseline_wrmsd=baseline_wrmsd,
        J_target=.85*baseline_wrmsd, epsilon=epsilon, constraint_penalty=.9,
        constraint_names=None if constraint_names is None else list(constraint_names),
        provenance={} if provenance is None else dict(provenance),
        primary_source='https://arxiv.org/abs/2605.05460')
    config['sha256'] = canonical_hash(config)
    return config


def compute_reward(validation_wrmsd, config, constraints):
    value = float(validation_wrmsd)
    if not isfinite(value) or value < 0:
        raise ValueError('Validation WRMSD must be finite and nonnegative')
    body = {key: item for key,item in config.items() if key != 'sha256'}
    if config.get('sha256') != canonical_hash(body):
        raise ValueError('The frozen reward configuration changed')
    if not isinstance(constraints, dict) or not constraints or any(type(v) is not bool for v in constraints.values()):
        raise ValueError('Constraints require explicit measured boolean results')
    expected = config.get('constraint_names')
    if expected is not None and set(constraints) != set(expected):
        raise ValueError('Constraint results must match the frozen declared set')
    failed = [name for name, passed in constraints.items() if not passed]
    raw = config['J_target']/max(value, config['epsilon'])
    return dict(raw=raw, penalized=raw*config['constraint_penalty']**len(failed),
                fitness=value, units=config['units'], failed_constraints=failed,
                constraints=dict(constraints), reward_sha256=config['sha256'])


_PREDICTION_COLUMNS = ('molecule_id', 'split', 'reference', 'prediction',
                       'signed_error', 'absolute_error', 'units')


def write_candidate_bundle(root, record, *, formula, feedback, predictions):
    """Write once after evaluation; proposal bundles use a separate root.

    Formula/proposal history is preserved under proposals/, while measured
    candidates live under candidates/. An unevaluated metrics=None staysnull.
    This helper never invents metrics or overwrites an earlier attempt.
    """
    candidate_id = str(record.get('candidate_id', ''))
    if not candidate_id or Path(candidate_id).name != candidate_id or candidate_id in ('.', '..'):
        raise ValueError('A candidate needs a single local directory name')
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError('Record the complete nonempty functional formula')
    folder = Path(root)/candidate_id
    folder.mkdir(parents=True, exist_ok=False)
    payload = dict(record, formula_sha256=hashlib.sha256(formula.encode()).hexdigest())
    (folder/'candidate.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    (folder/'formula.md').write_text(formula.rstrip()+'\n')
    (folder/'scores.json').write_text(json.dumps(record.get('scores', record.get('metrics')),
                                              indent=2, allow_nan=False)+'\n')
    (folder/'feedback.md').write_text(feedback.rstrip()+'\n')
    with (folder/'predictions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(predictions)
    return folder


__all__ = ['canonical_hash', 'freeze_molecule_split', 'freeze_reward_config',
           'compute_reward', 'write_candidate_bundle']


def freeze_validation_winner(records):
    """Freeze one model after two to five adaptive rounds, before any test score.

    The unchanged PBE baseline participates. Ranking uses the declared penalized
    validation reward; exact ties prefer fewer parameters and then the ID.
    """
    records=tuple(records)
    ids=[r['candidate_id'] for r in records]
    if len(set(ids)) != len(ids) or 'c000' not in ids or not 3 <= len(ids) <= 6:
        raise ValueError('Require the baseline and two to five distinct scored rounds')
    protocols={r['reference_protocol_sha256'] for r in records}
    rewards={r['reward_sha256'] for r in records}
    if len(protocols)!=1 or len(rewards)!=1:
        raise ValueError('All candidates must use the same frozen reference protocol and reward')
    for row in records:
        score=row['scores'];metrics=score['metrics']
        if row['status']!='evaluated' or set(metrics)!={'train','validation'}:
            raise ValueError('Only evaluated train/validation results may select a winner; test is sealed')
        if not all(value['valid'] for value in metrics.values()):
            raise ValueError('A winner requires finite valid fixed-reference response')
        reward=score['reward']
        if (reward['reward_sha256']!=row['reward_sha256'] or not isfinite(reward['penalized'])
                or not all(isfinite(value) for value in row['parameters'])):
            raise ValueError('Reward identity, score and parameters must be valid')
    best=min(records,key=lambda r:(-r['scores']['reward']['penalized'],len(r['parameters']),r['candidate_id']))
    frozen=dict(status='frozen',candidate_id=best['candidate_id'],parameters=best['parameters'],
        protocol_sha256=best['reference_protocol_sha256'],reward_sha256=best['reward_sha256'],
        module_sha256=best['module_sha256'],validation_reward=best['scores']['reward']['penalized'],
        candidate_record_sha256=canonical_hash(best),ranking='penalized validation reward; ties: fewer parameters, then ID',
        evaluated_candidates=ids)
    frozen['sha256']=canonical_hash(frozen)
    return frozen


__all__ += ['freeze_validation_winner']
