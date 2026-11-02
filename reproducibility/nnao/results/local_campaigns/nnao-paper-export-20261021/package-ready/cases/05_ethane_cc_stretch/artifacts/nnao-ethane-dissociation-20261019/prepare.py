"""Prepare the dense grid and a provenance-recorded adaptation of the old runner."""
from pathlib import Path
import hashlib
import json
import numpy as np
from geometry import stretch_rows

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'nnao-ethane-torsion-20261018'
base = json.loads((OLD/'angle-000'/'structure.json').read_text())
r0 = float(np.linalg.norm(np.diff(np.asarray(base['coords_angstrom'])[:2], axis=0)))
distances = [r0, *np.round(np.arange(1.55, 3.001, .05), 8),
             *np.round(np.arange(3.1, 5.001, .1), 8), 5.25, 5.5, 6., 7., 8.]
assert len(distances) == 56 and np.all(np.diff(distances) > 0)
rows = stretch_rows(base, distances)
np.testing.assert_allclose(rows[0]['coords_angstrom'], base['coords_angstrom'], atol=1e-15, rtol=0)
(OUT/'structures.json').write_text(json.dumps(rows, indent=2)+'\n')
source = (OLD/'run_point.py').read_text()
source_hash = hashlib.sha256(source.encode()).hexdigest()
replacements = [
    ('from geometry import torsion_rows', 'from geometry import stretch_rows'),
    ("expected = torsion_rows(base, [row['angle_deg']])[0]",
     "expected = stretch_rows(base, [row['distance_angstrom']])[0]"),
    ("source_geometry_in_training_set=True, angle_deg=row['angle_deg']",
     "source_geometry_in_training_set=False, distance_angstrom=row['distance_angstrom']"),
    ('exports.append(dict(structure_id=IDENTIFIER, basis=label, symbols=row[\'symbols\'],',
     "exports[:] = [record for record in exports if record['basis'] != label]\n"
     "        exports.append(dict(structure_id=IDENTIFIER, basis=label, symbols=row['symbols'],"),
    ("progress('mace_implicit_forward_backward')",
     "export(mace, 'MACE-update3351', None)\n    progress('mace_implicit_forward_backward')"),
]
for old, new in replacements:
    assert source.count(old) == 1, old
    source = source.replace(old, new)
source = source.replace('"""One-geometry, fixed-pool coefficient optimization against frozen MACE."""',
    '"""RHF C-C stretch; adapted from the frozen torsion runner, no solver changes."""')
assert 'torsion_rows' not in source and 'angle_deg' not in source
compile(source, 'run_point.py', 'exec')
(OUT/'run_point.py').write_text(source)
(OUT/'preparation.json').write_text(json.dumps(dict(points=len(rows),
    baseline_distance_angstrom=r0, source_runner=str((OLD/'run_point.py').relative_to(OUT.parent.parent)),
    source_runner_sha256=source_hash, adapted_runner_sha256=hashlib.sha256(source.encode()).hexdigest(),
    modifications=['torsion geometry replaced by rigid C-C stretch',
                   'metadata identifies stretched geometries as outside the original training geometry',
                   'MACE basis exported before SCF so failed points remain inspectable'],
    method='DF-RHF', electronic_branch='restricted closed-shell singlet',
    physical_dissociation_curve=False, fixed_exponents=True, fixed_core=True,
    free_contraction_starts=['mace', 'reference'], finite_differences=False), indent=2)+'\n')
print(f'Prepared {len(rows)} distances from {distances[0]:.9f} to {distances[-1]:g} Angstrom.')
