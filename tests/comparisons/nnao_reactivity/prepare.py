"""Select complete distance curves and one small reaction per BH9 category."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnao_reactivity.protocols import bse_element_shells, parse_din, parse_xyz, validate_reaction

S66 = ['Water-Water', 'AcOH-AcOH', 'Benzene-Benzene_pi-pi',
       'Cyclopentane-Cyclopentane', 'Benzene-Benzene_TS', 'Benzene-Water_OH-pi']
HB = ['1.01', '1.09', '1.15', '1.22', '1.28', '2.10']
CATEGORY_NAMES = dict(zip(['01', '02', '03', '04', '05', '06', '07', '08', '09'],
    ['Radical rearrangement and addition', 'Pericyclic', 'Halogen atom transfer',
     'Hydrogen atom transfer', 'Hydride transfer', 'B- and Si-containing reactions',
     'Proton transfer', 'Nucleophilic substitution', 'Nucleophilic addition']))


def assemble_bh9(records):
    grouped = defaultdict(dict)
    for row in records:
        match = re.fullmatch(r'BH9_(\d+)_(forward|reverse)', row['label'])
        if match:
            grouped[int(match[1])][match[2]] = row
    result = []
    for number, sides in sorted(grouped.items()):
        f, r = sides['forward'], sides['reverse']
        ts, = [k for k, v in f['terms'].items() if v == 1 and k.endswith('TS')]
        if r['terms'].get(ts) != 1:
            raise ValueError('Forward and reverse barriers use different transition states.')
        reactants = {k: -v for k, v in f['terms'].items() if v < 0}
        products = {k: -v for k, v in r['terms'].items() if v < 0}
        stoichiometry = Counter(products)
        stoichiometry.subtract(reactants)
        stoichiometry = {k: v for k, v in stoichiometry.items() if v}
        reaction, = [row for row in records if row['label'] == 'BH9-RE_'+str(number)]
        if reaction['terms'] != stoichiometry:
            raise ValueError('Reference reaction stoichiometry differs from its barriers.')
        closure = f['reference_kcal_mol']-r['reference_kcal_mol']-reaction['reference_kcal_mol']
        if abs(closure) > .0200001:
            raise ValueError('Reference barrier/reaction identity is inconsistent.')
        category = ts.split('_')[1]
        result.append(dict(identifier=ts[:-2], category=category, category_name=CATEGORY_NAMES[category], number=number,
            reactants=reactants, transition=ts, products=products,
            reference_forward_kcal_mol=f['reference_kcal_mol'],
            reference_reverse_kcal_mol=r['reference_kcal_mol'],
            reference_reaction_kcal_mol=reaction['reference_kcal_mol'],
            reference_closure_kcal_mol=closure))
    return result


def fragment_indices(dimer, monomers):
    groups, start = [], 0
    for monomer in monomers:
        end = start+monomer['natoms']
        if dimer['symbols'][start:end] != monomer['symbols']:
            raise ValueError('Upstream dimer ordering differs from its monomer files.')
        xyz = np.asarray(dimer['coords_angstrom'][start:end])
        reference = np.asarray(monomer['coords_angstrom'])
        distance = lambda r: np.linalg.norm(r[:, None]-r[None, :], axis=-1)
        np.testing.assert_allclose(distance(xyz), distance(reference), atol=1e-5, rtol=0)
        groups.append(list(range(start, end)))
        start = end
    if start != dimer['natoms']:
        raise ValueError('Monomers do not partition the entire dimer.')
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--paths-only', action='store_true')
    args = parser.parse_args()
    root = args.directory / 'raw'
    source = json.loads((root/'source-index.json').read_text())
    snapshot = json.loads(Path('artifacts/nnao-relative500-20261018/snapshot.json').read_text())
    elements = set(snapshot['model_config']['elements'])

    def read(directory, name):
        path = directory+'/'+name+'.xyz'
        raw = (root/path).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if actual != source['files'][path]['sha']:
            raise ValueError('Source XYZ checksum differs: '+path)
        row = parse_xyz(raw.decode(), directory+'/'+name)
        row['source_path'] = path
        row['source_sha256'] = hashlib.sha256(raw).hexdigest()
        return row

    curves, paths = [], []
    for dataset, directory, chosen in [('S66x8', '20_s66x8', S66), ('HB300SPXx10', '20_hb300spx', HB)]:
        entries = parse_din((root/'10_din'/('s66x8.din' if dataset == 'S66x8' else 'hb300spx.din')).read_text())
        for entry in entries:
            name, = [k for k, v in entry['terms'].items() if v == 1]
            if dataset == 'S66x8':
                group, scale = name.rsplit('_', 1)
                label = group
            else:
                group, scale, label = name.split('_', 2)
            if group not in chosen:
                continue
            monomers = [k for k, v in entry['terms'].items() if v == -1]
            if len(monomers) != 2:
                raise ValueError('Expected exactly two monomers.')
            paths += [directory+'/'+k+'.xyz' for k in [name]+monomers]
            curves.append(dict(dataset=dataset, group=group, label=label, scale=float(scale),
                identifier=directory+'/'+name, name=name, directory=directory,
                monomer_names=monomers, reference_kcal_mol=entry['reference_kcal_mol']))
    paths = sorted(set(paths))
    (args.directory/'selected-paths.json').write_text(json.dumps(paths, indent=2)+'\n')
    if args.paths_only:
        print(json.dumps(dict(curve_points=len(curves), files=len(paths))))
        return
    asset_path = root/'bse-631gdp-Br.json'
    asset_raw = asset_path.read_bytes()
    asset = json.loads(asset_raw)
    if asset['version'] != '1' or set(asset['elements']) != {'35'}:
        raise ValueError('Unexpected BSE basis asset version or element coverage.')
    basis_assets = {'6-31g(d,p)/Br': dict(shells=bse_element_shells(asset['elements']['35']),
        source_sha256=hashlib.sha256(asset_raw).hexdigest(), version=asset['version'],
        revision_date=asset['revision_date'], source_function_types=asset['function_types'],
        source_url='https://www.basissetexchange.org/api/basis/6-31g%28d%2Cp%29/format/json/?elements=35&version=1')}
    for row in curves:
        dimer = read(row['directory'], row['name'])
        monomers = [read(row['directory'], name) for name in row['monomer_names']]
        if any(r['charge'] or r['spin'] for r in [dimer]+monomers):
            raise ValueError('The selected CP protocol requires neutral closed-shell fragments.')
        if not set(dimer['atomic_numbers']) <= elements:
            raise ValueError('Selected curve contains an unsupported model element.')
        row.update(structure=dimer, monomers=monomers,
                   fragment_indices=fragment_indices(dimer, monomers))
    for dataset, groups, expected in [('S66x8', S66, 8), ('HB300SPXx10', HB, 10)]:
        for group in groups:
            scales = [r['scale'] for r in curves if r['dataset'] == dataset and r['group'] == group]
            if len(scales) != expected or len(set(scales)) != expected or 1. not in scales:
                raise ValueError('Incomplete original distance curve: '+group)
    reactions = assemble_bh9(parse_din((root/'10_din/bh9.din').read_text()))
    accepted, excluded, structures = [], [], {}
    for reaction in reactions:
        names = sorted(set(reaction['reactants']) | set(reaction['products']) | {reaction['transition']})
        rows = {name: read('20_bh9', name) for name in names}
        validate_reaction(reaction['reactants'], reaction['transition'], reaction['products'], rows)
        reasons = []
        unsupported = sorted({z for r in rows.values() for z in r['atomic_numbers']} - elements)
        if unsupported:
            reasons.append('unsupported model atomic numbers: '+str(unsupported))
        if max(r['natoms'] for r in rows.values()) > 30:
            reasons.append('initial panel maximum 30 atoms per species')
        reaction.update(species=names, max_atoms=max(r['natoms'] for r in rows.values()),
                        total_nuclear_charge=sum(sum(r['atomic_numbers']) for r in rows.values()))
        if reasons:
            excluded.append(dict(identifier=reaction['identifier'], reasons=reasons))
        else:
            accepted.append(reaction)
            structures.update(rows)
    selected = []
    for category in sorted({r['category'] for r in reactions}):
        candidates = [r for r in accepted if r['category'] == category]
        if not candidates:
            raise ValueError('No supported small reaction in category '+category)
        selected.append(min(candidates, key=lambda r: (r['max_atoms'], r['total_nuclear_charge'], r['number'])))
    names = sorted({name for r in selected for name in r['species']})
    hb_equilibrium = [p for p in source['files'] if p.startswith('20_hb300spx/') and '_1.00_' in p and p.endswith('.xyz')]
    unsupported_hb = [p for p in hb_equilibrium if not set(read('20_hb300spx', Path(p).stem)['atomic_numbers']) <= elements]
    output = dict(profile='representative', method='DF-HF', auxbasis='def2-universal-jkfit',
        source_commit=source['commit'], source_repository=source['repository'],
        checkpoint_sha256=snapshot['checkpoint_sha256'], checkpoint_update=snapshot['update_step'],
        full_counts=dict(S66x8=528, HB300SPXx10=3000, BH9_reactions=449),
        curves=curves, reactions=selected, structures={k: structures[k] for k in names}, basis_assets=basis_assets,
        selection=dict(S66_groups=S66, HB_groups=HB, BH9='smallest supported reaction per original numeric category',
            maximum_species_atoms=30, bh9_eligible_reactions=len(accepted), bh9_excluded=excluded,
            hb_unsupported_equilibrium_files=unsupported_hb),
        reference_sources=dict(S66x8='https://doi.org/10.1039/C6CP00688D',
            HB300SPXx10='https://doi.org/10.1021/acs.jctc.0c00715', BH9='https://doi.org/10.1021/acs.jctc.1c00694'))
    (args.directory/'dataset.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(curve_points=len(curves), reactions=len(selected), species=len(names),
        eligible_BH9=len(accepted), unsupported_HB_curves=len(unsupported_hb), selected_BH9=[r['identifier'] for r in selected])))


if __name__ == '__main__':
    main()
