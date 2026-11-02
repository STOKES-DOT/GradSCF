"""Exercise A/B updates and gradient gates on all 26 reference structures."""
from pathlib import Path
import sys
import jax
from gradscf.model.nnao.runtime import atomic_json,repository_root,scientific_source_hash
from gradscf.model.nnao.workflows.pool_calibration import load_references,run_pilot

family=sys.argv[1]
root=Path('artifacts/pool-calibration-repair-20261014')
data=dict(load_references('artifacts/main-group1000-core-sp6-d3-20261013/inputs/pilot.json'),basis_family=family)
assert jax.default_backend()=='gpu',jax.devices()
result=run_pilot(data,root/'calibration-validation'/family,calibration_steps=1,
                 contraction_steps=1,starts=1,scf_platform='cuda')
assert result['status']=='complete'
records=result['comparisons']
assert {r['structure_id'] for r in records}=={r['structure_id'] for r in data['structures']}
assert all(r['pool_scf']['scf_backend']=='gpu' and r['pool_scf']['stable'] and r['free_scf']['stable'] for r in records)
atomic_json(root/f'{family}-passed.json',dict(passed=True,family=family,
    references=len(data['structures']),comparisons=len(records),
    source_sha256=scientific_source_hash(repository_root())))
