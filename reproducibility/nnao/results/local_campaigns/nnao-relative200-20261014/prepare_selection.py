"""Freeze complete GMTKN55 v2 reactions before evaluating any model energies."""
import json
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
DATA = Path('/Volumes/TF/GradSCF_GMTKN55/data/v2')
GROUPS = {
    'reaction': ['G2RC/2', 'G2RC/5', 'G2RC/12', 'G2RC/17', 'ISO34/1', 'ISO34/14'],
    'barrier': ['BH76/37', 'BH76/38', 'BHPERI/1', 'BHPERI/7', 'BHROT27/1', 'BHROT27/7'],
    'noncovalent': ['S22/1', 'S22/2', 'S22/3', 'S22/9', 'S22/10', 'S22/16'],
}

def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def formula(row):
    return tuple(sorted(Counter(row['symbols']).items()))

def distance_signature(row):
    xyz = np.asarray(row['coords_angstrom'])
    return sorted((tuple(sorted((row['symbols'][i], row['symbols'][j]))),
                   round(float(np.linalg.norm(xyz[i]-xyz[j])), 5))
                  for i in range(len(xyz)) for j in range(i))

def main():
    structures = {r['structure_id']: r for r in read_rows(DATA/'structures.jsonl')}
    reactions = {r['reaction_id']: r for r in read_rows(DATA/'reactions.jsonl')}
    training = json.loads((OUT/'training-references.json').read_text())['structures']
    training_ids = {r['structure_id'].removeprefix('GMTKN55-v2/') for r in training}
    selected = []
    for category, ids in GROUPS.items():
        for key in ids:
            row = dict(reactions[key], category=category)
            balance = Counter()
            for term in row['terms']:
                structure = structures[term['structure_id']]
                assert structure['charge'] == structure['spin'] == 0
                assert set(structure['symbols']) <= set('HCNOF')
                assert structure['natoms'] <= 30
                for symbol, n in Counter(structure['symbols']).items():
                    balance[symbol] += term['coefficient'] * n
            assert all(v == 0 for v in balance.values()), (key, balance)
            assert row['energy_add_kcal_mol'] == 0 and row['error_multiplier'] == 1
            selected.append(row)
    ids = sorted({t['structure_id'] for r in selected for t in r['terms']})
    rows = []
    for key in ids:
        row = dict(structures[key])
        same_formula = [r for r in training if formula(r) == formula(row)]
        matches = [r['structure_id'] for r in same_formula
                   if distance_signature(r) == distance_signature(row)]
        row.update(training_id_overlap=key in training_ids,
                   training_distance_signature_matches=matches,
                   training_formula_matches=[r['structure_id'] for r in same_formula])
        assert not row['training_id_overlap'] and not matches, key
        rows.append(row)
    for name, items in [('test-selection.jsonl', rows), ('reactions.jsonl', selected)]:
        (OUT/name).write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in items))
    manifest = dict(dataset='GMTKN55 v2', commit='ccabc16c8c4677fc60c19870ae96ac25405844a2',
        categories=GROUPS, structure_count=len(rows), reaction_count=len(selected),
        max_atoms=max(r['natoms'] for r in rows), training_id_overlap=0,
        training_distance_signature_overlap=0,
        training_formula_overlap=sum(bool(r['training_formula_matches']) for r in rows),
        overlap_scope='Current 200 training structures; element-pair distances rounded to 1e-5 Angstrom. Not chemical-family, fragment, pilot-calibration or model-selection independence.',
        noncovalent_convention='Original GMTKN55 sign: monomer sum minus complex, positive binding. Uncorrected for BSSE. MACE evaluated independently on each real structure.',
        selected_before_energies=True,
        sha256={name: hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                for name in ['test-selection.jsonl', 'reactions.jsonl', 'snapshot.json']},
        source_sha256={name: hashlib.sha256((DATA/name).read_bytes()).hexdigest()
                       for name in ['structures.jsonl', 'reactions.jsonl']})
    (OUT/'selection-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
