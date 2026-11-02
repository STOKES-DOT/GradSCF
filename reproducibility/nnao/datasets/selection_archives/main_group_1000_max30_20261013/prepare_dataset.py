"""Prepare a local structure-only candidate set; never submit or modify training."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
TF = Path('/Volumes/TF')
GMT = TF/'GradSCF_GMTKN55/data/v2/structures.jsonl'
DB = TF/'QH9_db/QH9Stable.db'
OLD = REPO/'artifacts/fixed-pool200-20261012/references.json'
SYMBOLS = 'X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn'.split()
Z = {s: i for i, s in enumerate(SYMBOLS)}
MAIN = set('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Ga Ge As Se Br Kr Rb Sr In Sn Sb Te I Xe Cs Ba Tl Pb Bi Po At Rn'.split())
CURRENT = set('H C N O F'.split())
SEED = 0
MAX_ATOMS = 30


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def order(identifier):
    return digest(f'{SEED}:{identifier}'.encode())


def geometry_signature(symbols, coords, charge=0, spin=0):
    """Element-labelled distance fingerprint, invariant to rigid motion/order.

    Distances rounded to 1e-5 Angstrom. This removes geometric duplicates;
    it is not a connectivity identity or a guarantee against homometric cases.
    Mirror geometries have the same signature and are treated as duplicates.
    """
    xyz = np.asarray(coords, dtype=float)
    distances = np.linalg.norm(xyz[:, None]-xyz[None, :], axis=-1)
    signatures = sorted((s, sorted((symbols[j], int(np.rint(distances[i, j]*1e5)))
        for j in range(len(symbols)) if i != j)) for i, s in enumerate(symbols))
    return digest(json.dumps([charge, spin, signatures], sort_keys=True).encode())


def normalize(row, source):
    symbols = row['symbols']
    n = len(symbols)
    if not set(symbols) <= MAIN:
        raise ValueError('not_main_group')
    if not (2 <= n <= MAX_ATOMS or symbols == ['He']):
        raise ValueError('atom_count')
    if row.get('charge') != 0 or row.get('spin') != 0:
        raise ValueError('charge_or_spin')
    xyz = np.asarray(row['coords_angstrom'], dtype=float)
    if xyz.shape != (n, 3) or not np.isfinite(xyz).all():
        raise ValueError('coordinates')
    distances = np.linalg.norm(xyz[:, None]-xyz[None, :], axis=-1)
    if n > 1 and distances[np.triu_indices(n, 1)].min() < .2:
        raise ValueError('coincident_or_extremely_close_nuclei')
    electrons = sum(Z[s] for s in symbols)
    if electrons % 2 or row.get('nelectron', electrons) != electrons:
        raise ValueError('electron_count')
    counts = Counter(symbols)
    formula = ''.join(s+(str(counts[s]) if counts[s] != 1 else '')
                      for s in sorted(counts, key=lambda s: Z[s]))
    support = []
    if not set(symbols) <= CURRENT:
        support.append('extend_grid_templates_MACE_elements_and_exponent_channels')
    templates = json.loads((REPO/'src/gradscf/model/nnao/data/szp3.json').read_text())['elements']
    if any(s not in templates for s in symbols):
        support.append('missing_legacy_NNAO_template')
    if any(Z[s] > 20 for s in symbols):
        support.append('unvalidated_core6_for_Z_above_20')
    if any(Z[s] >= 31 for s in symbols):
        support.append('heavy_element_Hamiltonian_and_auxiliary_basis_review')
    if n == 1:
        support.append('isolated_atom_MACE_and_SCF_gate')
    return dict(structure_id=f"{source}/{row['structure_id']}", source_id=row['structure_id'],
        source_dataset=source, subset=row.get('subset', source), symbols=symbols,
        atomic_numbers=[Z[s] for s in symbols], coords_angstrom=xyz.tolist(), units='Angstrom',
        natoms=n, nelectron=electrons, charge=0, spin=0, multiplicity=1, weight=1,
        formula=formula, geometry_signature_sha256=geometry_signature(symbols, xyz),
        role='isolated_atom_reference' if n == 1 else 'molecule_or_complex',
        electronic_state_provenance=row.get('electronic_state_provenance', dict(
            charge_source=row.get('charge_source'), spin_source=row.get('spin_source'))),
        source_path=row.get('source_path'), source_row_sha256=row.get('source_row_sha256'),
        scf_validated=False, training_ready=False, required_support=support,
        existing_grid_element_scope=not support,
        reference_energy_used=False)


def main():
    output = ROOT/'dataset'
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing dataset.')
    old = json.loads(OLD.read_text())['structures']
    old_ids = {r['structure_id'] for r in old if len(r['symbols']) <= MAX_ATOMS}
    all_gmt = [json.loads(x) for x in GMT.read_text().splitlines()]
    excluded = Counter()
    eligible, seen, aliases = [], {}, []
    for raw in sorted(all_gmt, key=lambda x: (x['structure_id'] not in old_ids, order(x['structure_id']))):
        try:
            if not raw.get('referenced_by_retained_reaction'):
                raise ValueError('not_referenced_by_retained_reaction')
            row = normalize(raw, 'GMTKN55-v2')
        except ValueError as error:
            excluded[str(error)] += 1
            continue
        signature = row['geometry_signature_sha256']
        if signature in seen:
            aliases.append(dict(duplicate=row['structure_id'], retained=seen[signature]))
            continue
        seen[signature] = row['structure_id']
        row['in_current_200'] = raw['structure_id'] in old_ids
        eligible.append(row)
    print('GMT eligible unique:', len(eligible), flush=True)
    frequencies = Counter(s for row in eligible for s in set(row['symbols']))
    selected, selected_ids = [], set()
    formulas, subsets, element_counts, bins = Counter(), Counter(), Counter(), Counter()
    def add(row):
        if row['structure_id'] in selected_ids:
            return
        selected.append(row)
        selected_ids.add(row['structure_id'])
        formulas[row['formula']] += 1
        subsets[row['subset']] += 1
        element_counts.update(set(row['symbols']))
        bins[(row['natoms']-1)//10] += 1
    for row in eligible:
        if row['in_current_200']:
            add(row)
    # Retain every eligible geometry carrying an element with <=20 examples.
    rare = {s for s, count in frequencies.items() if count <= 20}
    for row in eligible:
        if set(row['symbols']) & rare:
            add(row)
    # Aim for 40 structures per new element, where the source contains them.
    for element in sorted(set(frequencies)-CURRENT, key=lambda s: (frequencies[s], Z[s])):
        pool = [r for r in eligible if element in r['symbols']]
        while element_counts[element] < min(40, frequencies[element]):
            options = [r for r in pool if r['structure_id'] not in selected_ids]
            chosen = min(options, key=lambda r: (formulas[r['formula']], subsets[r['subset']], order(r['structure_id'])))
            add(chosen)
    coverage_count = len(selected)
    while len(selected) < 800:
        options = [r for r in eligible if r['structure_id'] not in selected_ids and formulas[r['formula']] < 4]
        if not options:
            raise ValueError('Not enough diverse GMTKN55 candidates under the explicit formula cap.')
        def score(r):
            novel = sum(1/(1+element_counts[s]) for s in set(r['symbols'])-CURRENT)
            return (2/(1+formulas[r['formula']]) + 1/(1+subsets[r['subset']])
                    + 5*novel + .2/(1+bins[(r['natoms']-1)//10]))
        add(min(options, key=lambda r: (-score(r), order(r['structure_id']))))
    assert len(selected) == 800
    selected_signatures = {r['geometry_signature_sha256'] for r in selected}
    print('GMT selected:', len(selected), 'coverage-first:', coverage_count, flush=True)

    # Read only a reproducible geometry sample; do not read Hamiltonian blobs.
    connection = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    highest = connection.execute('select max(id) from data').fetchone()[0]
    ids = np.sort(np.random.default_rng(SEED).choice(highest+1, size=3000, replace=False)).tolist()
    qh_rows = []
    qh_excluded = Counter()
    for begin in range(0, len(ids), 300):
        chunk = ids[begin:begin+300]
        query = 'select id,N,Z,pos from data where id in ('+','.join('?' for _ in chunk)+') order by id'
        for index, n, numbers_blob, xyz_blob in connection.execute(query, chunk):
            numbers = np.frombuffer(numbers_blob, dtype='<i4')
            xyz = np.frombuffer(xyz_blob, dtype='<f8').reshape(n, 3)
            if numbers.shape != (n,) or not set(numbers.tolist()) <= {1, 6, 7, 8, 9}:
                raise ValueError('Unexpected QH9 geometry blob layout/elements.')
            raw = dict(structure_id=str(index), symbols=[SYMBOLS[z] for z in numbers],
                coords_angstrom=xyz.tolist(), charge=0, spin=0,
                electronic_state_provenance=dict(kind='dataset_generation_convention',
                    note='QH9Stable official generator uses neutral singlet Mole defaults and RKS; charge/spin are not SQLite columns.',
                    source='https://github.com/divelab/AIRS/blob/main/OpenDFT/QHBench/QH9/dataset_generation_examples/QH9_stable.py'),
                source_path=f'{DB}::data[id={index}]',
                source_row_sha256=digest(str(index).encode()+str(n).encode()+numbers_blob+xyz_blob))
            try:
                row = normalize(raw, 'QH9Stable')
            except ValueError as error:
                qh_excluded[str(error)] += 1
                continue
            if row['geometry_signature_sha256'] in selected_signatures:
                qh_excluded['geometry_duplicate'] += 1
                continue
            selected_signatures.add(row['geometry_signature_sha256'])
            row['in_current_200'] = False
            qh_rows.append(row)
        print('QH9 sampled:', min(begin+300, len(ids)), flush=True)
    connection.close()
    qh_selected = []
    qh_ids = set()
    while len(qh_selected) < 200:
        options = [r for r in qh_rows if r['structure_id'] not in qh_ids and formulas[r['formula']] < 4]
        if not options:
            raise ValueError('Insufficient QH9 formula diversity.')
        chosen = min(options, key=lambda r: (formulas[r['formula']], bins[(r['natoms']-1)//10], order(r['structure_id'])))
        qh_selected.append(chosen)
        qh_ids.add(chosen['structure_id'])
        add(chosen)
    assert len(selected) == 1000
    selected.sort(key=lambda r: (r['source_dataset'], r['structure_id']))
    assert len({r['geometry_signature_sha256'] for r in selected}) == 1000
    assert set(element_counts) == set(frequencies)
    assert all(r['natoms'] <= 30 and r['charge'] == r['spin'] == 0 for r in selected)
    output.mkdir()
    (output/'train.jsonl').write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False)+'\n' for r in selected))
    dump(output/'references.json', dict(format='nnao-expanded-candidate-v1', method='RHF',
        units='Angstrom', basis_family=None, core_primitives=None, training_ready=False,
        description='Structure-only candidate set; expand and validate the basis/model before training.',
        structures=selected))
    with (output/'systems.csv').open('w', newline='') as f:
        fields = ['structure_id', 'source_dataset', 'subset', 'formula', 'natoms', 'nelectron',
                  'charge', 'spin', 'role', 'in_current_200', 'existing_grid_element_scope', 'required_support']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in selected:
            writer.writerow({k: ';'.join(r[k]) if isinstance(r[k], list) else r[k] for k in fields})
    xyzdir = output/'xyz'
    xyzdir.mkdir()
    for i, r in enumerate(selected):
        text = f"{r['natoms']}\n{r['structure_id']} charge=0 spin=0 units=Angstrom\n"
        text += ''.join(f'{s} {a[0]:.16g} {a[1]:.16g} {a[2]:.16g}\n' for s, a in zip(r['symbols'], r['coords_angstrom']))
        (xyzdir/f'{i:04d}.xyz').write_text(text)
        r['xyz_file'] = f'xyz/{i:04d}.xyz'
    dump(output/'xyz-index.json', {r['structure_id']: r['xyz_file'] for r in selected})
    inventory = {s: dict(atomic_number=Z[s], structures=element_counts[s],
        atom_instances=sum(r['symbols'].count(s) for r in selected),
        eligible_GMTKN55_geometries=frequencies[s], sparse=element_counts[s] < 10)
        for s in sorted(element_counts, key=lambda s: Z[s])}
    dump(output/'element-coverage.json', inventory)
    dump(output/'deduplication.json', dict(method='element-labelled per-atom distance fingerprint, rounding 1e-5 Angstrom',
        limitations='Not molecular connectivity grouping; conformers remain distinct, mirrors merge; rare homometric cases not excluded.',
        GMTKN55_aliases=aliases))
    dump(output/'source-audit.json', dict(
        GMTKN55=dict(path=str(GMT), sha256=digest(GMT.read_bytes()), total_rows=len(all_gmt),
            excluded=dict(excluded), eligible_unique=len(eligible)),
        QH9Stable=dict(path=str(DB), sqlite_open_mode='read_only', sampled_ids=ids,
            database_size_bytes=DB.stat().st_size, database_mtime_ns=DB.stat().st_mtime_ns,
            full_database_sha256=None, selected_geometry_row_hashes_recorded=True,
            sampled_eligible=len(qh_rows), excluded=dict(qh_excluded), labels_read=False,
            coordinate_unit_evidence='https://github.com/divelab/AIRS/blob/main/OpenDFT/QHBench/QH9/datasets.py'),
        other_TF_sources=dict(QM9_GW_BSE='133886 xyz archive entries; CHNOF overlap, not added.',
            OpenMolcas='Example geometries have no additional main-group coverage and inconsistent electronic-state metadata; not selected.',
            agent_workspace='Generated molecular conformers; not added without consistent charge/spin/source audit.')))
    metadata = dict(count=1000, sources=dict(Counter(r['source_dataset'] for r in selected)),
        element_count=len(inventory), elements=list(inventory), seed=SEED, maximum_atoms=MAX_ATOMS,
        size_bins={f'{lo}-{hi}':sum(lo <= r['natoms'] <= hi for r in selected)
                   for lo, hi in [(1,1),(2,10),(11,20),(21,30)]},
        existing_200_retained=sum(r['in_current_200'] for r in selected),
        existing_200_over_atom_limit=[r['structure_id'] for r in old if len(r['symbols']) > MAX_ATOMS],
        CHNOF_only=sum(set(r['symbols']) <= CURRENT for r in selected),
        expanded_element_systems=sum(not set(r['symbols']) <= CURRENT for r in selected),
        formula_count=len(formulas), subset_counts=dict(subsets),
        formula_counts_above_four={k:v for k,v in formulas.items() if v>4},
        formula_cap_exception='Current 200 eligible structures and rare-element coverage have priority over the fill-stage cap of four.',
        weight_policy='All weights 1; diversity selection changes element frequencies.',
        intended_loss='Mean self-consistent variational energy; no supervised energy labels.',
        new_validation_split=False, scf_evaluations_run=0, training_launched=False,
        basis_parameters_changed=False, existing_jobs_modified=False,
        selector_sha256=digest(Path(__file__).read_bytes()),
        current_200_source_sha256=digest(OLD.read_bytes()),
        train_jsonl_sha256=digest((output/'train.jsonl').read_bytes()))
    dump(output/'manifest.json', metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
