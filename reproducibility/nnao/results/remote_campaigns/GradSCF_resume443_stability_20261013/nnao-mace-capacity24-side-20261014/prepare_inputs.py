"""Copy the historical pilot24 definition; keep geometry and pool fixed."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
source=ROOT/'artifacts/nnao-joint-native-24-20261010/job-95224/structures.jsonl'
snapshot_path=ROOT/'artifacts/nnao-relative200-20261014/snapshot.json'
snapshot=json.loads(snapshot_path.read_text())
rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
assert len(rows)==24 and len({r['structure_id'] for r in rows})==24
for row in rows:
    assert row['charge']==row['spin']==0
    assert set(row['symbols'])<=set('HCNOF')
    row.update(weight=1.,split='train',formal_split='capacity-diagnostic-train')
definition=dict(method='RHF',units='Angstrom',basis_family='szp663_direct',
    core_primitives=6,structures=rows,
    description='Historical 24 structures; matched architecture-capacity diagnostic, not held-out testing.')
candidate=snapshot['candidate']
common=dict(elements=[1,6,7,8,9],num_interactions=2,correlation=2,cutoff=5.,
            zero_init=True,basis_family='szp663_direct')
manifest=dict(epochs=200,seed=0,learning_rate=1e-4,storage='mmap',
    variants={'small':dict(common,channels=8,max_ell=1),
              'default':dict(common,channels=32,max_ell=2)},
    execution_order=['default','small'],
    source_sha256=snapshot['source_sha256'],
    pool_provenance=dict(checkpoint_epoch=snapshot['checkpoint_epoch'],
        checkpoint_sha256=snapshot['checkpoint_sha256'],role='Exponent pool only; no model weights reused'),
    original_24_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    changed_variables=['channels: 8 -> 32','max_ell: 1 -> 2'],
    interpretation='Combined default-architecture comparison, not an isolated width ablation; one seed, train-set diagnostic only.')
for name,data in [('references.json',definition),('candidate.json',candidate),('experiment.json',manifest)]:
    path=OUT/name
    text=json.dumps(data,indent=2,sort_keys=True)+'\n'
    if path.exists(): assert path.read_text()==text,'Existing experiment differs'
    else:path.write_text(text)
print(json.dumps(dict(structures=len(rows),max_atoms=max(r['natoms'] for r in rows),
    variants=manifest['variants'],epochs=manifest['epochs']),indent=2))
