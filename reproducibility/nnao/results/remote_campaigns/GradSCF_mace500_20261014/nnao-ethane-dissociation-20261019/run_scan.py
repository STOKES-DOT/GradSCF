"""Run both MACE and independently optimized contractions at each distance."""
from pathlib import Path
import gc
import json
import os
import shutil
import time
import traceback

import jax
from gradscf.model.nnao.runtime import atomic_json, file_hash
import run_point

OUT = Path(__file__).resolve().parent
ROWS = json.loads((OUT/'structures.json').read_text())
started = time.perf_counter()
summary = dict(status='running', job_id=os.environ.get('SLURM_JOB_ID'),
    method='DF-RHF', requested_points=len(ROWS), points=[],
    electronic_branch='restricted closed-shell singlet; not physical homolytic dissociation',
    script_sha256=file_hash(__file__))
for index, row in enumerate(ROWS):
    folder = OUT/f'distance-{index:03d}'
    folder.mkdir(exist_ok=True)
    atomic_json(folder/'structure.json', row)
    summary.update(phase=folder.name, distance_angstrom=row['distance_angstrom'],
                   elapsed_seconds=time.perf_counter()-started)
    atomic_json(OUT/'scan-progress.json', summary)
    print('BEGIN', index, row['distance_angstrom'], flush=True)
    if (folder/'result.json').is_file():
        record = json.loads((folder/'result.json').read_text())
        assert record['distance_angstrom'] == row['distance_angstrom']
    else:
        record = dict(distance_angstrom=row['distance_angstrom'], folder=folder.name)
        if index == 0:
            old = OUT.parent/'nnao-ethane-torsion-20261018'/'angle-000'
            point = json.loads((old/'summary.json').read_text())
            assert point['status'] == 'complete' and point['best_gradient_converged']
            for filename in ['summary.json', 'frozen-bases.jsonl', 'best-coefficients.npz']:
                shutil.copy2(old/filename, folder/filename)
            record.update(reused_baseline=True, reused_source=str(old))
        else:
            try:
                run_point.OUT, run_point.IDENTIFIER = folder, row['structure_id']
                run_point.main()
                point = json.loads((folder/'summary.json').read_text())
            except Exception as error:
                record.update(status='failed', error=str(error), traceback=traceback.format_exc())
                if (folder/'progress.json').is_file():
                    record['failure_progress'] = json.loads((folder/'progress.json').read_text())
                point = None
        if point is not None:
            record.update(status=('passed' if point['status'] == 'complete' and
                point['best_gradient_converged'] else 'not_converged'),
                mace_energy_hartree=point['mace_energy_hartree'],
                free_energy_hartree=point['best_energy_hartree'],
                gap_kcal_mol=point['mace_minus_free_kcal_mol'],
                gradient_norm=point['best_gradient_norm'],
                free_gradient_converged=point['best_gradient_converged'],
                endpoint_spread_hartree=point['endpoint_spread_hartree'],
                nao=point['nao'], device=point['device'],
                starts=point['starts'], seconds=point['elapsed_seconds'])
        atomic_json(folder/'result.json', record)
    summary['points'].append(record)
    summary.update(passed=sum(p['status'] == 'passed' for p in summary['points']),
        failed=sum(p['status'] == 'failed' for p in summary['points']),
        not_converged=sum(p['status'] == 'not_converged' for p in summary['points']),
        elapsed_seconds=time.perf_counter()-started)
    atomic_json(OUT/'scan-progress.json', summary)
    print('END', index, json.dumps(record, allow_nan=False), flush=True)
    jax.clear_caches()
    gc.collect()
summary.update(status='complete' if summary['passed'] == len(ROWS) else 'complete_with_failures',
               phase='complete', elapsed_seconds=time.perf_counter()-started)
atomic_json(OUT/'scan-summary.json', summary)
print('SCAN_FINISHED', summary['status'], summary['passed'], '/', len(ROWS), flush=True)
