"""Resume an XC100 history after equivalence checks on a new allocation."""
import ast
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from gradscf.model.autoxc import Candidate, CodexDesigner, WYSelection, run
from gradscf.model.autoxc.history import History, source_digest, versions, write_json
from .distributed import RemoteEvaluator


def check_evaluator(previous, current):
    """Accept new placement and tiny reduction drift, preserving the scoring contract."""
    old, new = deepcopy(previous), deepcopy(current)
    for config in (old, new):
        del config['fingerprint']
        del config['worker']['execution']
    before = old['worker'].pop('outer_normalization')
    after = new['worker'].pop('outer_normalization')
    if old != new or len(before) != len(after) or not all(
        math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12) for a, b in zip(before, after)
    ):
        raise ValueError('The new allocation changed the XC100 scoring contract')


def frozen_potential(payload, coefficients):
    """Turn a fitted expression into an exact fixed-parameter scoring request."""
    original = Candidate.from_json(payload)
    if set(coefficients) != set(original.initial):
        raise ValueError('Checkpoint coefficients do not match the potential')

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in coefficients:
                return ast.copy_location(ast.Constant(value=float(coefficients[node.id])), node)
            return node

    def freeze(expression):
        return ast.unparse(Substitute().visit(ast.parse(expression, mode='eval')))

    return Candidate(freeze(original.formula), kind=original.kind,
                     definitions=tuple((name, freeze(expression))
                                       for name, expression in original.definitions),
                     hypothesis='Recheck the saved incumbent with fixed coefficients.')


def stage_history(previous, output, config, evaluator):
    """Recheck the incumbent and preserve all past records byte for byte."""
    history_file = previous / 'run' / 'history.jsonl'
    raw = history_file.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    final = json.loads((previous / 'final.json').read_text())
    if records != final['records'] or [r['round'] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError('The completed checkpoint and the round history disagree')
    winner = next((r for r in reversed(records) if r['accepted']), None)
    if winner is None or winner['digest'] != final['incumbent']['digest']:
        raise ValueError('The checkpoint does not identify its retained candidate')
    old = winner['evaluation']
    replayed = evaluator.evaluate(frozen_potential(winner['candidate'], old['parameters']))
    if not replayed.valid or set(replayed.samples) != set(old['samples']):
        raise ValueError('The incumbent could not be scored on the complete new allocation')
    reward_error = abs(replayed.reward - old['reward'])
    sample_errors = {
        key: max(abs(replayed.samples[i][key] - old['samples'][i][key]) for i in old['samples'])
        for key in ('reward', 'density_l1', 'density_error')
    }
    if reward_error > 5e-9 or any(error > 2e-8 for error in sample_errors.values()):
        raise ValueError(f'Incumbent recheck disagrees with the checkpoint: {reward_error=}, {sample_errors=}')
    receipt = {
        'parent': str(previous), 'inherited_rounds': len(records),
        'history_sha256': sha256(raw).hexdigest(), 'incumbent_round': winner['round'],
        'incumbent_reward_before': old['reward'], 'incumbent_reward_after': replayed.reward,
        'max_sample_errors': sample_errors, 'job_id': evaluator.ready['job_id'],
        'source': final['source'],
        'original_evaluator_fingerprint': json.loads((previous/'run/config.json').read_text())['evaluator']['fingerprint'],
        'new_evaluator_fingerprint': evaluator.configuration['fingerprint'],
    }
    with TemporaryDirectory(prefix='checkpoint-', dir=output) as directory:
        staged = Path(directory) / 'run'
        history = History(staged, config)
        (staged / 'history.jsonl').write_bytes(raw)
        history.write('continuation.json', receipt)
        staged.rename(output / 'run')
    print(f'Incumbent recheck passed: {old["reward"]:.12f} -> {replayed.reward:.12f}. '
          f'Resuming after round {len(records)}.', flush=True)


def continue_campaign(host, root, *, previous, output, job_id, system_prompt, action_policy, rounds):
    """Attach an independent completed history to a compatible numerical worker."""
    previous, output = Path(previous).resolve(), Path(output).resolve()
    previous_config = json.loads((previous / 'run' / 'config.json').read_text())
    guardian = RemoteEvaluator(host, root, {}, receipt_directory=output/'numerical')
    try:
        if (source_digest() != previous_config['source'] or versions() != previous_config['versions']
                or action_policy.configuration != previous_config['action_policy']):
            raise ValueError('Local source, versions or design policy differ from the completed run')
        evaluator = RemoteEvaluator.wait_ready(host, root, receipt_directory=output/'numerical',
                                               timeout=7 * 24 * 3600, job_id=job_id)
        check_evaluator(previous_config['evaluator'], evaluator.configuration)
        selection = WYSelection(density_scales=previous_config['selection']['density_scales'],
                                tolerance=previous_config['selection']['tolerance'])
        saved_session = next(row['session_id'] for row in reversed(
            [json.loads(line) for line in (previous/'run/history.jsonl').read_text().splitlines()])
            if row['session_id'])
        original = previous_config['designer']
        designer = CodexDesigner(cwd=previous/'designer', model=original['model'],
                                 effort=original['effort'], kind=original['kind'],
                                 timeout=original['timeout'], session_id=saved_session,
                                 system_prompt=system_prompt)
        if designer.configuration != original:
            raise ValueError('Designer configuration differs from the completed run')
        config = {**previous_config, 'evaluator': evaluator.configuration}
        if not (output/'run').exists():
            stage_history(previous, output, config, evaluator)
        else:
            History(output/'run', config)
        with designer:
            result = run(designer, evaluator, directory=output/'run', rounds=rounds,
                         context=previous_config['context'], fail_fast=previous_config['fail_fast'],
                         initial_reward=previous_config['initial_reward'],
                         improvement_tolerance=previous_config['improvement_tolerance'],
                         selection=selection, action_policy=action_policy)
        write_json(output/'final.json', result)
        return result
    finally:
        try:
            guardian.release()
        except Exception as exc:
            write_json(output/'release_error.json', {'error': str(exc)})
