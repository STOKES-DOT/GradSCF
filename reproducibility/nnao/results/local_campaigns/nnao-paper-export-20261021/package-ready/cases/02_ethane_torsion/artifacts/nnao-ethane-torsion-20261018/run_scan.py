"""Compare frozen MACE and free contractions on a rigid ethane torsion grid."""
from pathlib import Path
import csv
import gc
import json
import os
import time

import jax
from gradscf.model.nnao.runtime import atomic_json, file_hash
import run_point

OUT = Path(__file__).resolve().parent
rows = json.loads((OUT/'structures.json').read_text())
started = time.perf_counter()
summary = dict(status='running', job_id=os.environ.get('SLURM_JOB_ID'),
    script_sha256=file_hash(__file__), requested_points=len(rows), points=[])
for row in rows:
    tag = f"angle-{int(row['angle_deg']):03d}"
    folder = OUT/tag
    folder.mkdir(exist_ok=True)
    atomic_json(folder/'structure.json', row)
    summary.update(phase=tag, elapsed_seconds=time.perf_counter()-started)
    atomic_json(OUT/'scan-progress.json', summary)
    print('BEGIN ANGLE', row['angle_deg'], flush=True)
    run_point.OUT = folder
    run_point.IDENTIFIER = row['structure_id']
    run_point.main()
    record = json.loads((folder/'summary.json').read_text())
    assert record['status'] == 'complete', record['starts']
    summary['points'].append(dict(angle_deg=row['angle_deg'], structure_id=row['structure_id'],
        folder=tag, nao=record['nao'], pool_nao=record['pool_nao'],
        mace_energy_hartree=record['mace_energy_hartree'],
        free_energy_hartree=record['best_energy_hartree'],
        gap_millihartree=record['mace_minus_free_millihartree'],
        gap_kcal_mol=record['mace_minus_free_kcal_mol'],
        coefficient_gradient_norm=record['best_gradient_norm'],
        coefficient_gradient_converged=record['best_gradient_converged'],
        endpoint_spread_hartree=record['endpoint_spread_hartree'],
        converged_starts=sum(s['gradient_converged'] for s in record['starts']),
        starts=len(record['starts']), seconds=record['elapsed_seconds']))
    summary['elapsed_seconds'] = time.perf_counter()-started
    atomic_json(OUT/'scan-progress.json', summary)
    print('END ANGLE', json.dumps(summary['points'][-1]), flush=True)
    jax.clear_caches()
    gc.collect()
summary.update(status='complete', phase='complete', elapsed_seconds=time.perf_counter()-started)
atomic_json(OUT/'scan-summary.json', summary)
with (OUT/'energies.csv').open('w', newline='') as stream:
    writer=csv.DictWriter(stream, fieldnames=list(summary['points'][0]))
    writer.writeheader()
    writer.writerows(summary['points'])
