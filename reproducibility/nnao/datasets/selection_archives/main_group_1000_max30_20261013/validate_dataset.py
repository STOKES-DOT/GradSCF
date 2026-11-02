"""Independent input/export checks; no SCF, JAX, cluster or training calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from prepare_dataset import GMT, DB, OLD, geometry_signature

root = Path(__file__).resolve().parent
p = root/'dataset'
rows = [json.loads(line) for line in (p/'train.jsonl').read_text().splitlines()]
manifest = json.loads((p/'manifest.json').read_text())
assert len(rows) == len({r['structure_id'] for r in rows}) == 1000
assert Counter(r['source_dataset'] for r in rows) == {'GMTKN55-v2': 800, 'QH9Stable': 200}
assert all(1 <= r['natoms'] <= 30 for r in rows)
assert all(r['charge'] == r['spin'] == 0 and r['nelectron'] % 2 == 0 for r in rows)
assert all(r['nelectron'] == sum(r['atomic_numbers']) for r in rows)
assert all(r['natoms'] == len(r['symbols']) == len(r['coords_angstrom']) for r in rows)
assert all(np.isfinite(r['coords_angstrom']).all() for r in rows)
assert len({r['geometry_signature_sha256'] for r in rows}) == 1000
assert {r['formula'] for r in rows if r['natoms'] == 1} == {'He'}
assert sum(r['natoms'] == 1 for r in rows) == 1
assert len({s for r in rows for s in r['symbols']}) == 31
assert all(geometry_signature(r['symbols'], r['coords_angstrom']) == r['geometry_signature_sha256'] for r in rows)
data = json.loads((p/'references.json').read_text())
assert data['structures'] == rows and data['training_ready'] is False
old = json.loads(OLD.read_text())['structures']
eligible_old = {r['structure_id'] for r in old if len(r['symbols']) <= 30}
assert {r['source_id'] for r in rows if r['in_current_200']} == eligible_old
assert len(eligible_old) == 181
source = {r['structure_id']: r for r in map(json.loads, GMT.read_text().splitlines())}
conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
index = json.loads((p/'xyz-index.json').read_text())
for row in rows:
    if row['source_dataset'] == 'GMTKN55-v2':
        original = source[row['source_id']]
        assert original['symbols'] == row['symbols']
        assert original['charge'] == row['charge'] and original['spin'] == row['spin']
        np.testing.assert_array_equal(original['coords_angstrom'], row['coords_angstrom'])
    else:
        i, n, zb, rb = conn.execute('select id,N,Z,pos from data where id=?', (int(row['source_id']),)).fetchone()
        np.testing.assert_array_equal(np.frombuffer(zb, '<i4'), row['atomic_numbers'])
        np.testing.assert_array_equal(np.frombuffer(rb, '<f8').reshape(n, 3), row['coords_angstrom'])
        assert hashlib.sha256(str(i).encode()+str(n).encode()+zb+rb).hexdigest() == row['source_row_sha256']
    xyz = (p/index[row['structure_id']]).read_text().splitlines()
    assert int(xyz[0]) == row['natoms']
    assert [line.split()[0] for line in xyz[2:]] == row['symbols']
    np.testing.assert_allclose([[float(v) for v in line.split()[1:]] for line in xyz[2:]],
                               row['coords_angstrom'], atol=1e-13, rtol=0)
conn.close()
assert hashlib.sha256((p/'train.jsonl').read_bytes()).hexdigest() == manifest['train_jsonl_sha256']
assert hashlib.sha256((root/'prepare_dataset.py').read_bytes()).hexdigest() == manifest['selector_sha256']
report = dict(passed=True, records_checked=1000, maximum_atoms=30, elements=31,
    exact_source_geometry_matches=1000, xyz_roundtrips=1000, unique_distance_fingerprints=1000,
    original_200_eligible_retained=181, scf_or_model_validation=False,
    train_jsonl_sha256=manifest['train_jsonl_sha256'])
(p/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
