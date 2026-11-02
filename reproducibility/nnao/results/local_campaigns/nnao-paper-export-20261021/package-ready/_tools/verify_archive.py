"""Verify an assembled NNAO archive without rerunning electronic structure."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
args=parser.parse_args()
root=args.directory.resolve()
listed={}
for line in (root/'SHA256SUMS').read_text().splitlines():
    expected,name=line.split('  ',1)
    path=(root/name).resolve()
    assert path.is_relative_to(root),name
    assert digest(path)==expected,name
    listed[name]=expected
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
assert set(listed)==actual,'Unlisted or missing payload files'
catalog=json.loads((root/'catalog.json').read_text())
assert len(catalog['cases'])==13
structures=json.loads((root/'structure-index.json').read_text())
for g in structures:
    lines=(root/g['path']).read_text().splitlines()
    assert int(lines[0])==g['natoms']==len(lines)-2
    assert g['units']=='angstrom' and 'units=angstrom' in lines[1]
    assert all(len(line.split())==4 for line in lines[2:])
    assert np.isfinite([[float(v) for v in line.split()[1:]] for line in lines[2:]]).all()
    for pointer in g['source_records']:
        match=re.search(r'\.jsonl?(?=[:/]|$)',pointer)
        assert match and (root/pointer[:match.end()]).is_file(),pointer
for path in [root/'README.md',*(root/c['archive_path']/'README.md' for c in catalog['cases'])]:
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
        if not target.startswith(('http://','https://')):
            assert (path.parent/target).is_file(),(path,target)

case_root=root/'cases'
read=lambda p:json.loads(p.read_text())
core=read(case_root/'07_radial_bsse_polarizability/artifacts/nnao-core-benchmarks-20261020/full/results.json')
assert (core['passed'],core['failed'],len(core['results']))==(272,12,284)
curve=case_root/'08_distance_curves_and_barriers'
report=read(curve/'artifacts/nnao-curves-barriers-repair-20261021/merged/results.json')
assert (report['passed'],report['failed'],len(report['results']))==(852,0,852)
checks=read(curve/'artifacts/nnao-curves-barriers-repair-20261021/combined-retry/independent-check.json')
assert checks['passed']==31 and checks['failed']==0
states=list((curve/'independent-checked-states').glob('*.npz'))
assert len(states)==31
for path in states:
    with np.load(path,allow_pickle=False) as f:
        d=f['density_spin']; c=f['mo_coeff']; occ=f['mo_occ']
        assert d.ndim==3 and d.shape[0]==2 and d.shape[-1]==d.shape[-2]
        assert c.ndim in (2,3) and c.shape[-1]==d.shape[-1]
        assert all(np.isfinite(a).all() for a in [d,c,occ])
stretch=read(case_root/'05_ethane_cc_stretch/artifacts/nnao-ethane-dissociation-20261019/scan-summary.json')
assert len(stretch['points'])==56 and stretch['passed']==51
torsion=read(case_root/'02_ethane_torsion/artifacts/nnao-ethane-torsion-20261018/scan-summary.json')
assert len(torsion['points'])==25
family=read(case_root/'10_nine_family_free_contractions/artifacts/free-family9-20261014/verification-summary.json')
assert family['verified_cases']==63
model=root/'models/update-3351'
original=read(model/'snapshot.json'); derived=read(model/'inference-snapshot.json')
with np.load(model/'parameters.npz',allow_pickle=False) as f:
    assert f.files==['parameters']
    parameters=f['parameters']
    assert parameters.size==original['parameter_count'] and str(parameters.dtype)=='float64'
    assert hashlib.sha256(parameters.tobytes()).hexdigest()==original['parameter_sha256']
assert digest(model/'parameters.npz')==derived['checkpoint_sha256']
assert derived['original_checkpoint_sha256']==original['checkpoint_sha256']
print(json.dumps(dict(status='passed',checksummed_files=len(listed),cases=len(catalog['cases']),
    xyz_structures=len(structures),distinct_coordinate_state_signatures=len({g['geometry_sha256'] for g in structures}),
    independent_states=len(states),model_parameters=int(parameters.size),
    checkpoint_parameter_sha256=original['parameter_sha256']),indent=2))
