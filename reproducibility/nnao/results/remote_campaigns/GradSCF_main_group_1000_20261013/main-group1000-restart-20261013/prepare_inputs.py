"""Freeze the <=30-atom, period-1--4, main-group campaign inputs."""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PARENT = REPO/'artifacts/expanded-elements-20261013/dataset_supported29'
spec = importlib.util.spec_from_file_location('selector', REPO/'artifacts/main-group-1000-20261013/prepare_dataset.py')
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)
MAIN4 = set('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Ga Ge As Se Br Kr'.split())


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main():
    out = ROOT/'inputs'
    out.mkdir(exist_ok=False)
    old = [json.loads(x) for x in (PARENT/'train.jsonl').read_text().splitlines()]
    rows = [r for r in old if set(r['symbols']) <= MAIN4]
    excluded = [r['structure_id'] for r in old if not set(r['symbols']) <= MAIN4]
    signatures = {r['geometry_signature_sha256'] for r in rows}
    ids = {r['structure_id'] for r in rows}
    formulas = Counter(r['formula'] for r in rows)
    counts = Counter(s for r in rows for s in set(r['symbols']))
    candidates = []
    for raw in map(json.loads, selector.GMT.read_text().splitlines()):
        if not set(raw['symbols']) <= MAIN4 or not raw.get('referenced_by_retained_reaction'):
            continue
        try:
            r = selector.normalize(raw, 'GMTKN55-v2')
        except ValueError:
            continue
        if r['structure_id'] not in ids and r['geometry_signature_sha256'] not in signatures:
            candidates.append(r)
    added = []
    while len(rows) < 1000:
        possible = [r for r in candidates if r['structure_id'] not in ids
                    and r['geometry_signature_sha256'] not in signatures and formulas[r['formula']] < 4]
        if not possible:
            raise ValueError('Not enough valid replacement structures.')
        r = min(possible, key=lambda r: (formulas[r['formula']],
            -sum(1/(1+counts[s]) for s in set(r['symbols'])-selector.CURRENT), selector.order(r['structure_id'])))
        r['in_current_200'] = False
        rows.append(r)
        ids.add(r['structure_id'])
        signatures.add(r['geometry_signature_sha256'])
        formulas[r['formula']] += 1
        counts.update(set(r['symbols']))
        added.append(r['structure_id'])
    rows.sort(key=lambda r: r['structure_id'])
    for r in rows:
        r['required_support'] = ['SCF_stability_and_implicit_gradient_validation']
        r['training_ready'] = False
    elements = sorted(counts, key=selector.Z.get)
    references = dict(method='RHF', units='Angstrom', basis_family='nnao_szp_l',
        core_primitives='reference', elements=elements, structures=rows,
        description='1000 structures, <=30 atoms including H, neutral spin=0, period-1--4 main group. Fresh MACE/Adam; runtime gates required.')
    write(out/'references.json', references)
    (out/'train.jsonl').write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False)+'\n' for r in rows))
    # Retain the established molecular pilot and add the smallest explicit
    # reference for each newly observed element. Do not synthesize geometries.
    pilot = json.loads((REPO/'src/gradscf/model/nnao/data/pool_pilot.json').read_text())
    pilot.update(basis_family='nnao_szp_l', core_primitives='reference', elements=elements)
    covered = {s for r in pilot['structures'] for s in r['symbols']}
    for symbol in elements:
        if symbol in covered:
            continue
        r = min((r for r in rows if symbol in r['symbols']),
                key=lambda r: (r['natoms'], r['nelectron'], r['structure_id']))
        pilot['structures'].append(r)
        covered.update(r['symbols'])
    pilot['description'] = 'Explicit equal-weight calibration references covering every training element; finite-budget A/B assessment.'
    write(out/'pilot.json', pilot)
    assert set(elements) == covered and len(rows) == len(signatures) == 1000
    assert all(max(r['atomic_numbers']) <= 36 and r['natoms'] <= 30 for r in rows)
    assert sum(r['in_current_200'] for r in rows) == 181
    assert Counter(r['source_dataset'] for r in rows) == {'GMTKN55-v2':800,'QH9Stable':200}
    manifest = dict(count=1000, target_epochs=1000, seed=0, learning_rate=1e-4,
        elements=elements, element_count=len(elements), supported_scope_not_observed=sorted(MAIN4-set(elements)),
        maximum_atoms_including_hydrogen=30, maximum_atomic_number=36,
        main_group_only=True, transition_metals_included=False,
        excluded_ids=excluded, replacement_ids=added, excluded_elements=['Sn','Sb','Te','I'],
        sources=dict(Counter(r['source_dataset'] for r in rows)),
        element_structure_counts=dict(counts), pilot_count=len(pilot['structures']),
        pilot_ids=[r['structure_id'] for r in pilot['structures']],
        original_200_overlap=181, no_heldout_split=True, fresh_MACE=True, reset_Adam=True,
        retained_old_job=96510, max_simultaneous_gpus=3, cpus_per_job=32,
        parent_sha256=hashlib.sha256((PARENT/'train.jsonl').read_bytes()).hexdigest(),
        references_sha256=hashlib.sha256((out/'references.json').read_bytes()).hexdigest(),
        train_jsonl_sha256=hashlib.sha256((out/'train.jsonl').read_bytes()).hexdigest(),
        pilot_sha256=hashlib.sha256((out/'pilot.json').read_bytes()).hexdigest())
    write(out/'manifest.json', manifest)
    with (out/'systems.csv').open('w', newline='') as f:
        fields = ['structure_id','source_dataset','formula','natoms','nelectron','charge','spin','in_current_200']
        w = csv.DictWriter(f,fieldnames=fields); w.writeheader()
        w.writerows({k:r[k] for k in fields} for r in rows)
    print(json.dumps({k:v for k,v in manifest.items() if not k.endswith('_ids')}, indent=2))


if __name__ == '__main__':
    main()
