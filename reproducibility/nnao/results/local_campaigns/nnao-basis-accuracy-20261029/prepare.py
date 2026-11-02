"""Freeze the approved 30 physical structures and reference definitions."""
from pathlib import Path
import collections
import hashlib
import json
import numpy as np
from ase.data.s22 import data as s22
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tests/comparisons'))
from nnao_basis_accuracy.protocols import fragment_indices

OUT = Path(__file__).resolve().parent
DATA = Path('/Volumes/TF/GradSCF_GMTKN55/data/v2')
PAPER = Path('/Users/jiaoyuan/Documents/GitHub/NNAO_paper')
BASES = ['sto-3g','sto-6g','3-21g','6-31g','6-31g(d,p)','6-31+g(d,p)',
    '6-311g(d,p)','6-311++g(2d,2p)','def2-svp','def2-svpd','def2-tzvp',
    'def2-tzvpp','def2-qzvp','def2-qzvpp','cc-pvdz','aug-cc-pvdz','cc-pvtz','cc-pvqz']
ISO = [3,14,15,24,25,26]
DIMERS = {1:'Ammonia_dimer',2:'Water_dimer',8:'Methane_dimer',9:'Ethene_dimer',
          10:'Benzene-methane_complex',16:'Ethene-ethyne_complex'}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def spectrum(row):
    z=np.asarray(row['atomic_numbers']);p=np.asarray(row['coords_angstrom'])
    groups=collections.defaultdict(list)
    for i in range(len(z)):
        for j in range(i):groups[tuple(sorted([int(z[i]),int(z[j])]))].append(float(np.linalg.norm(p[i]-p[j])))
    return tuple(sorted(map(int,z))),np.asarray([v for k in sorted(groups) for v in sorted(groups[k])])


def main():
    structures={r['structure_id']:r for r in map(json.loads,(DATA/'structures.jsonl').read_text().splitlines())}
    reactions={r['reaction_id']:r for r in map(json.loads,(DATA/'reactions.jsonl').read_text().splitlines())}
    rows=[]; observables=[]
    def add(identifier, source, symbols=None, coords=None, **extra):
        r=dict(structures[source])
        if symbols is not None:
            r.update(symbols=list(symbols),coords_angstrom=np.asarray(coords).tolist(),
                atomic_numbers=[{'H':1,'C':6,'N':7,'O':8}[s] for s in symbols],natoms=len(symbols))
        r.update(structure_id=identifier,source_structure_id=source,charge=0,spin=0,**extra)
        r['nelectron']=sum(r['atomic_numbers'])
        assert not(set(r['symbols'])-set('HCNO')) and r['nelectron']%2==0
        rows.append(r)
        return r
    for i in ISO:
        reference=reactions[f'ISO34/{i}']
        for t in reference['terms']:add(t['structure_id'],t['structure_id'],task='isomer')
        observables.append(dict(reference,category='isomer'))
    for i,name in DIMERS.items():
        sid=f'S22/{i:02d}'; r=structures[sid];counts=s22[name]['dimer atoms']
        assert len(r['symbols'])==sum(counts)
        # GMTKN files group atoms by element, not by molecular fragment.
        groups=fragment_indices(r)
        if [len(g) for g in groups]!=counts:groups=groups[::-1]
        assert [len(g) for g in groups]==counts
        add(sid,sid,task='dimer',fragment_indices=groups,label=name)
        for label,indices in zip('AB',groups):
            add(f'{sid}/{label}',sid,[r['symbols'][j] for j in indices],
                [r['coords_angstrom'][j] for j in indices],task='monomer',parent=sid,
                related_monomer_source_id=sid+('a' if i in [1,2,8,9] or label=='A' else 'b'))
        ref=reactions[f'S22/{i}']
        observables.append(dict(reaction_id=f'S22B/{i}',category='dimer',label=name,
            terms=[dict(structure_id=sid,coefficient=1.),dict(structure_id=sid+'/A',coefficient=-1.),
                   dict(structure_id=sid+'/B',coefficient=-1.)],
            reference_energy_kcal_mol=-ref['reference_energy_kcal_mol'],
            source_reference=ref,reference_doi='10.1063/1.3659142',
            source_sign_note='GMTKN uses positive dissociation energies; interaction energy is its negative'))
    assert len(rows)==30 and len({r['structure_id'] for r in rows})==30
    train_path=PAPER/'data/training/nnao500-20261021/references.json'
    train=json.loads(train_path.read_text())['structures']
    prepared=[(r,spectrum(r)) for r in train]
    for r in rows:
        z,d=spectrum(r)
        r['training_geometry_candidates']=[s['structure_id'] for s,(sz,sd) in prepared
            if sz==z and len(sd)==len(d) and np.allclose(sd,d,atol=1e-7,rtol=0)]
        r['training_source_id_matches']=[s['structure_id'] for s in train
            if s['structure_id'].removeprefix('GMTKN55-v2/') in
                [r['source_structure_id'],r.get('related_monomer_source_id')]]
    write(OUT/'inputs/structures.json',rows)
    write(OUT/'inputs/observables.json',observables)
    write(OUT/'inputs/settings.json',dict(classical_bases=BASES,reference_basis='aug-cc-pvqz',
        methods=['mp2','wb97m-v'],jkfit='def2-universal-jkfit',mp2fit='aug-cc-pvqz-ri',
        grid_level=3,nlc_grid_level=3,conv_tol=1e-10,conv_tol_grad=1e-7,max_cycle=200,
        frozen_core='one 1s MO per real C/N/O; no core from ghosts',dtype='float64',with_t2=False,
        main_cases=1620,reference_cases=60,cp_diagnostic_cases_upper_bound=888,
        gpu_cap=3,cpu_cap=64,main_score='0.5*(six-isomer MAE + six-dimer MAE)',
        classical_reference='GMTKN55 ISO34 and S22B',finite_differences=False))
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [DATA/'structures.jsonl',DATA/'reactions.jsonl',train_path]}
    write(OUT/'inputs/source-manifest.json',manifest)
    write(OUT/'inputs/selection-summary.json',dict(physical_structures=30,isomer_reactions=6,dimers=6,
        largest_natoms=max(r['natoms'] for r in rows),
        overlap_candidates={r['structure_id']:r['training_geometry_candidates'] for r in rows if r['training_geometry_candidates']},
        related_source_matches={r['structure_id']:r['training_source_id_matches'] for r in rows if r['training_source_id_matches']},
        overlap_policy='Keep and label, no held-out claim; distance spectrum match is conservative'))
    print((OUT/'inputs/selection-summary.json').read_text())


if __name__=='__main__':main()
