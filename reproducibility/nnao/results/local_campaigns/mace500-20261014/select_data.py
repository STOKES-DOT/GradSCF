"""Select 500 source-backed structures, prioritizing existing element diversity."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
SOURCE=REPO/'artifacts/main-group1000-core-sp6-d3-20261013/inputs/references.json'

def fingerprint(row):
    xyz=np.asarray(row['coords_angstrom']);symbols=row['symbols']
    pairs=sorted((tuple(sorted((symbols[i],symbols[j]))),round(float(np.linalg.norm(xyz[i]-xyz[j])),5))
                 for i in range(len(xyz)) for j in range(i))
    return hashlib.sha256(json.dumps([sorted(symbols),row['charge'],row['spin'],pairs]).encode()).hexdigest()

def main():
    data=json.loads(SOURCE.read_text())
    tests=[json.loads(s) for s in (REPO/'artifacts/nnao-relative200-20261014/test-selection.jsonl').read_text().splitlines()]
    excluded={fingerprint(r) for r in tests};seen=set();pool=[]
    for row in data['structures']:
        assert row['charge']==row['spin']==0 and row['natoms']<=30
        assert row['nelectron']==sum(row['atomic_numbers'])
        signature=fingerprint(row)
        if signature in excluded or signature in seen:continue
        seen.add(signature);pool.append(dict(row,selection_distance_signature=signature))
    expected=set(s for r in pool for s in r['symbols'])
    selected=[];ids=set();counts=Counter();formulas=Counter();subsets=Counter();sources=Counter()
    caps={'GMTKN55-v2':400,'QH9Stable':100}
    frequency=Counter(s for r in pool for s in set(r['symbols']))
    def score(r):
        noncommon=set(r['symbols'])-set('HCNOF')
        new=sum(s not in counts for s in set(r['symbols']))
        rarity=sum(1/(frequency[s]*(1+counts[s])) for s in noncommon)
        return (-new,-bool(noncommon),-rarity,formulas[r['formula']],subsets[r['subset']],
                hashlib.sha256(r['structure_id'].encode()).hexdigest())
    while len(selected)<500:
        available=[r for r in pool if r['structure_id'] not in ids and sources[r['source_dataset']]<caps[r['source_dataset']]]
        if not available:raise RuntimeError('Insufficient unique supported structures for the requested source quotas.')
        row=min(available,key=score)
        selected.append(row);ids.add(row['structure_id']);counts.update(set(row['symbols']))
        formulas[row['formula']]+=1;subsets[row['subset']]+=1;sources[row['source_dataset']]+=1
    selected.sort(key=lambda r:r['structure_id'])
    assert set(counts)==expected and len(ids)==500
    elements=[s for s in data['elements'] if s in counts]
    result=dict(data,elements=elements,basis_family='nnao_szp_l',structures=selected,
        training_ready=False,description='500 systems, full available period-1-to-4 main-group coverage; SCF gates pending.')
    (ROOT/'inputs').mkdir(parents=True,exist_ok=True)
    write=lambda name,x:(ROOT/'inputs'/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
    write('references.json',result)
    n=sum(r['natoms'] for r in selected);edges=0
    for r in selected:
        x=np.asarray(r['coords_angstrom']);distance=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
        edges+=int(((distance<5)&(~np.eye(len(x),dtype=bool))).sum())
    manifest=dict(count=500,elements=elements,element_count=len(elements),element_structure_counts=dict(counts),
        source_counts=dict(sources),max_atoms=max(r['natoms'] for r in selected),
        total_atoms=n,directed_edges_cutoff5=edges,average_neighbors=edges/n,
        original_200_retained=sum(bool(r.get('in_current_200')) for r in selected),
        benchmark_geometry_overlap=0,geometry_test='Element-pair distances rounded to 1e-5 Angstrom; not chemical-family independence.',
        missing_supported_element='Ga: absent from available GMTKN55 v2 structures; no synthetic replacement.',
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        selection_sha256=hashlib.sha256((ROOT/'inputs/references.json').read_bytes()).hexdigest(),
        unique_formulas=len(formulas),scf_validated=False)
    write('manifest.json',manifest)
    (ROOT/'inputs/train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    pilot=json.loads((REPO/'artifacts/main-group1000-core-sp6-d3-20261013/inputs/pilot.json').read_text())
    pilot.update(basis_family='nnao_szp_l',core_primitives='6-31g')
    write('pilot.json',pilot)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
