"""Prepare explicit benchmark structures; no geometry or electronic optimization."""
from collections import Counter
from itertools import permutations
from pathlib import Path
import hashlib
import json
import numpy as np

NUMBERS = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
RADII = {'H': .31, 'C': .76, 'N': .71, 'O': .66}
MOLECULES = [
    ('benzene', 'benzene', {'C': 6, 'H': 6}),
    ('naphthalene', 'naphthalene', {'C': 10, 'H': 8}),
    ('anthracene', 'anthracene', {'C': 14, 'H': 10}),
    ('p-nitroaniline', '4-nitroaniline', {'C': 6, 'H': 6, 'N': 2, 'O': 2}),
    ('cyclohexane', 'cyclohexane', {'C': 6, 'H': 12}),
    ('hexane', 'hexane', {'C': 6, 'H': 14}),
    ('ethylene', 'ethylene', {'C': 2, 'H': 4}),
    ('butadiene', '1,3-butadiene', {'C': 4, 'H': 6}),
    ('hexatriene', 'trans-1,3,5-hexatriene', {'C': 6, 'H': 8}),
]


def fingerprint(row, indices):
    xyz, symbols = np.asarray(row['coords_angstrom']), row['symbols']
    result = {}
    for pos, i in enumerate(indices):
        for j in indices[:pos]:
            key = tuple(sorted([symbols[i], symbols[j]]))
            result.setdefault(key, []).append(float(np.linalg.norm(xyz[i]-xyz[j])))
    return {key: np.sort(values) for key, values in result.items()}


def bond_graph(row):
    xyz = np.asarray(row['coords_angstrom'])
    radii = np.array([RADII[s] for s in row['symbols']])
    distance = np.linalg.norm(xyz[:, None]-xyz[None], axis=2)
    return (distance < 1.25*(radii[:, None]+radii[None])) & (distance > 1e-5)


def same_connectivity(first_symbols, first, second_symbols, second):
    candidates = [[j for j, symbol in enumerate(second_symbols)
                   if symbol == first_symbols[i] and first[i].sum() == second[j].sum()]
                  for i in range(len(first_symbols))]
    order = sorted(range(len(first_symbols)), key=lambda i: (len(candidates[i]), -first[i].sum()))
    mapping = {}

    def match(step):
        if step == len(order):
            return True
        i = order[step]
        for j in candidates[i]:
            if j in mapping.values() or any(first[i, a] != second[j, b] for a, b in mapping.items()):
                continue
            mapping[i] = j
            if match(step+1):
                return True
            del mapping[i]
        return False
    return match(0)


def fragments(dimer, monomers):
    symbols = dimer['symbols']
    bonded = bond_graph(dimer)
    unused, groups = set(range(len(symbols))), []
    while unused:
        group, pending = [], [min(unused)]
        while pending:
            i = pending.pop()
            if i not in unused:
                continue
            unused.remove(i)
            group.append(i)
            pending.extend(np.flatnonzero(bonded[i]).tolist())
        groups.append(sorted(group))
    if len(groups) != 2:
        raise ValueError(f"{dimer['structure_id']}: expected two covalent fragments, got {groups}.")
    # The upstream monomers can be relaxed. Match their molecular graphs, then
    # retain the bound dimer coordinates for all CP fragment calculations.
    for ordering in permutations(groups):
        errors = []
        for group, monomer in zip(ordering, monomers):
            if Counter(symbols[i] for i in group) != Counter(monomer['symbols']):
                break
            if not same_connectivity([symbols[i] for i in group], bonded[np.ix_(group, group)],
                                     monomer['symbols'], bond_graph(monomer)):
                break
            first = fingerprint(dimer, group)
            second = fingerprint(monomer, list(range(monomer['natoms'])))
            if first.keys() != second.keys():
                break
            errors.append(max((np.max(abs(first[k]-second[k]), initial=0.) for k in first), default=0.))
        if len(errors) == 2:
            return list(ordering), max(errors)
    raise ValueError(f"{dimer['structure_id']}: fragment connectivity does not match upstream monomers.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gmtkn55', type=Path, default=Path('/Volumes/TF/GradSCF_GMTKN55/data/v2'))
    args = parser.parse_args()
    data_path = args.output/'data'
    structures_path, reactions_path = args.gmtkn55/'structures.jsonl', args.gmtkn55/'reactions.jsonl'
    structures = {row['structure_id']: row for row in map(json.loads, structures_path.read_text().splitlines())}
    s22 = []
    for reaction in map(json.loads, reactions_path.read_text().splitlines()):
        if reaction['subset'] != 'S22':
            continue
        dimer = [term for term in reaction['terms'] if term['coefficient'] == -1]
        assert len(dimer) == 1
        row = dict(structures[dimer[0]['structure_id']])
        monomers = [structures[term['structure_id']] for term in reaction['terms'] if term['coefficient'] > 0
                    for _ in range(int(term['coefficient']))]
        assert len(monomers) == 2 and row['charge'] == row['spin'] == 0
        groups, error = fragments(row, monomers)
        row.update(fragment_indices=groups, reference_monomer_deformation_angstrom=error,
            fragment_validation='element-labelled molecular graph match; retain dimer coordinates',
            fragment_sources=[m['structure_id'] for m in monomers],
            reference_dissociation_kcal_mol=abs(reaction['reference_energy_kcal_mol']),
            label=reaction.get('source_annotation', row['structure_id']))
        s22.append(row)
    s22.sort(key=lambda x: int(x['structure_id'].split('/')[-1]))
    assert len(s22) == 22
    polar = []
    for label, query, formula in MOLECULES:
        path = data_path/f'{label}-pubchem.sdf'
        text = path.read_text()
        records = [record for record in text.split('$$$$') if record.strip()]
        assert len(records) == 1 and '3D' in text.splitlines()[1]
        lines = text.splitlines()
        count = int(lines[3][:3])
        symbols = [line[31:34].strip() for line in lines[4:4+count]]
        coordinates = [[float(line[a:a+10]) for a in (0, 10, 20)] for line in lines[4:4+count]]
        # Nitro groups encode balanced formal charges although the molecule
        # is neutral. Validate their sum instead of rejecting the records.
        formal_charge = sum(sum(map(int, line.split()[4::2])) for line in lines if line.startswith('M  CHG'))
        assert Counter(symbols) == formula and formal_charge == 0
        polar.append(dict(structure_id=f'PubChem/{lines[0].strip()}', label=label,
            symbols=symbols, atomic_numbers=[NUMBERS[s] for s in symbols], natoms=count,
            charge=0, spin=0, nelectron=sum(NUMBERS[s] for s in symbols),
            coords_angstrom=coordinates, source='PubChem 3D conformer; not the paper coordinates',
            source_query=query, source_cid=int(lines[0]),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    atoms = [dict(structure_id=label, label=label, symbols=[symbol], atomic_numbers=[NUMBERS[symbol]],
                  natoms=1, coords_angstrom=[[0., 0., 0.]], charge=charge, spin=spin,
                  nelectron=NUMBERS[symbol]-charge)
             for label, symbol, charge, spin in [('H', 'H', 0, 1), ('H-', 'H', -1, 0),
                                                ('C', 'C', 0, 2), ('C-', 'C', -1, 3)]]
    result = dict(format='nnao-core-benchmarks-v1', method='HF', auxbasis='def2-universal-jkfit',
        atoms=atoms, s22=s22, polarizability=polar,
        atomic_reference='aug-cc-pv5z; not a reproduction of the paper 14s reference',
        polarizability_reference='ma-def2-qzvpp',
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [structures_path, reactions_path]},
        paper='10.1063/5.0172373', geometry_note='Same nine molecular identities as Fig. 2; independently sourced PubChem geometries.')
    (args.output/'dataset.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(atoms=len(atoms), s22=len(s22), polarizability=len(polar),
        max_reference_monomer_deformation_angstrom=max(row['reference_monomer_deformation_angstrom'] for row in s22))))


if __name__ == '__main__':
    main()
