"""Continue incomplete free-contraction references from their saved Adam state."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from gradscf.model.nnao.training.pool import PoolRHF, minimize_reference
from gradscf.model.nnao.workflows.pool_calibration import _measure, _report, _save_result
from gradscf.model.nnao.runtime import atomic_json, file_hash, repository_root, scientific_source_hash

parent = Path('artifacts/pool-calibration-refined-20261012')
output = Path('artifacts/pool-free-convergence-20261012')
if output.exists():
    raise RuntimeError('Use a fresh continuation directory.')
output.mkdir()
shutil.copytree(parent/'checkpoints', output/'checkpoints')
metadata = json.loads((parent/'metadata.json').read_text())
result = copy.deepcopy(json.loads((parent/'summary.json').read_text()))
result.update(status='running', parent=str(parent), continuation_measurements=[])
keys = tuple(map(tuple, metadata['channel_keys']))
rows = {x['structure_id']: x for x in metadata['references']['structures']}
continuation_identity = dict(parent=str(parent), total_step_budget=1000,
    source_sha256=scientific_source_hash(repository_root()),
    checkpoint_energy_gradient_revalidated=True, optimizer_state_preserved=True)
atomic_json(output/'continuation.json', continuation_identity)
for record in result['comparisons']:
    if record['optimizer_converged']:
        continue
    identifier, candidate = record['structure_id'], record['candidate']
    model = PoolRHF(rows[identifier], keys)
    parameters = np.asarray(json.loads((parent/f'{candidate}-parameters.json').read_text())['values'])
    free = model.prepare_contraction(parameters)
    directory = Path(record['checkpoint']).parent
    digest = Path(record['checkpoint']).name.split('-start-')[0]
    runs = []
    for path in sorted((parent/directory).glob(digest+'-start-*.npz')):
        with np.load(path) as saved:
            previous = json.loads(str(saved['metadata'].item()))
            previous.update(parameters=np.array(saved['parameters']),
                gradient=np.array(saved['gradient']), optimizer=saved['optimizer'].tobytes())
        if previous['status'] == 'max_steps':
            current = _measure(f'{identifier}/{path.stem}/continue',
                lambda: minimize_reference(previous['parameters'], free.evaluate,
                    max_steps=1000, learning_rate=.01, resume=previous),
                result['continuation_measurements'])
            identity = dict(previous['identity'], continuation=continuation_identity,
                            parent_checkpoint_sha256=file_hash(path))
            _save_result(output/directory/path.name, current, kind='free_contraction', identity=identity)
        else:
            current = previous
        runs.append((current, directory/path.name))
    best, checkpoint = min(runs, key=lambda x: x[0]['energy_hartree'])
    record.update(free_energy_hartree=best['energy_hartree'],
        delta_contraction_hartree=best['energy_hartree']-record['pool_energy_hartree'],
        optimizer_status=best['status'], optimizer_converged=best['optimizer_converged'],
        gradient_norm=best['gradient_norm'], checkpoint=str(checkpoint), free_scf=best['info'])
    assert record['delta_contraction_hartree'] >= -1e-7
    atomic_json(output/'summary.json', result)
    _report(output, result)
    print(identifier, best['status'], best['accepted_steps'], best['gradient_norm'],
          best['energy_hartree'], flush=True)
result['status'] = 'complete'
atomic_json(output/'summary.json', result)
_report(output, result)
