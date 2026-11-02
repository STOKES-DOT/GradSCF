"""Publish the exact validated source to the idle queued campaign checkout."""
import json
import os
from pathlib import Path
import re
import subprocess
from gradscf.model.nnao.runtime import atomic_json,file_hash,scientific_source_hash

root=Path.cwd()
parent=Path('/home/user_yuanjiao/private/homefile/GradSCF_main_group1000_sp6_d3_20261013')
out=root/'artifacts/pool-calibration-repair-20261014'
campaign=Path('artifacts/main-group1000-core-sp6-d3-20261013')
expected='325af8aa021bb2a73bcdfa4446db6c88b414795f1cb7f25e44745e6f25681b15'
assert scientific_source_hash(parent)==expected,'Parent changed while validating'
new_hash=scientific_source_hash(root)
def sources(base):
    package=base/'src/gradscf'
    return {str(p.relative_to(package)):file_hash(p) for p in package.rglob('*')
            if p.is_file() and p.suffix in {'.py','.json','.dat'}}
a,b=sources(parent),sources(root)
changed=sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k))
allowed=['model/nnao/training/pool.py','model/nnao/workflows/basis_grid.py','model/nnao/workflows/pool_calibration.py']
assert changed==allowed,changed
native=Path('src/gradscf/integrals/_native/lib/libgradscf_integrals.so')
assert file_hash(root/native)==file_hash(parent/native)
checks=[json.loads((out/f'{f}-passed.json').read_text()) for f in ['nnao_dzp_s','nnao_tzp_s']]
assert all(r['passed'] and r['source_sha256']==new_hash for r in checks)
rows=[json.loads(p.read_text()) for p in (root/campaign/'validation-pool-calibration-fixed').glob('*/passed.json')]
assert len(rows)==9 and len({r['family'] for r in rows})==9
assert all(r['passed'] and r['scientific_source_sha256']==new_hash for r in rows)
active=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-t','RUNNING','-o','%i'],text=True).split()
for identifier in active:
    job=subprocess.check_output(['scontrol','show','job',identifier,'-o'],text=True)
    match=re.search(r'WorkDir=(\S+)',job)
    assert not match or Path(match[1]).resolve()!=parent.resolve(),f'Parent checkout is active in {identifier}'
for relative in changed:
    destination=parent/'src/gradscf'/relative
    backup=out/'parent_sources'/relative
    backup.parent.mkdir(parents=True,exist_ok=True)
    backup.write_bytes(destination.read_bytes())
    temporary=destination.with_name(destination.name+'.gpu-repair.tmp')
    temporary.write_bytes((root/'src/gradscf'/relative).read_bytes())
    temporary.replace(destination)
assert scientific_source_hash(parent)==new_hash
gate=dict(passed=True,results=rows,calibration_checks=checks)
atomic_json(root/campaign/'validation-passed.json',gate)
atomic_json(parent/campaign/'validation-passed.json',gate)
atomic_json(out/'deployment.json',dict(passed=True,parent_source=expected,new_source=new_hash,
    changed=changed,native_sha256=file_hash(root/native),validation_families=9))
print(json.dumps({'deployed':str(parent),'source':new_hash,'calibration_checked':2,'training_replays':9}),flush=True)
