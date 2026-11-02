"""Freeze the metric fix separately from every active training source."""
import hashlib
import json
import os
from pathlib import Path
import shutil


ROOT=Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021')
CAMPAIGN=ROOT/'artifacts/nnao-family500-20261021'
HERE=CAMPAIGN/'repairs/c6cl6-20261026'
BASE=CAMPAIGN/'repairs/kr3-20261024/source'


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def source_hash(root):
    value=hashlib.sha256();package=root/'src/gradscf'
    for path in sorted(package.rglob('*')):
        if path.is_file() and path.suffix in {'.py','.json','.dat'}:
            value.update(str(path.relative_to(package)).encode());value.update(digest(path).encode())
    return value.hexdigest()


def main():
    assert not (HERE/'deployment.json').exists()
    old_hash=source_hash(BASE)
    assert old_hash=='7e5a22c4a17e3d727c03c121be4484e902080cef7b39b164783ce5fde23cdec2'
    assert digest(BASE/'src/gradscf/model/nnao/training/objective.py')==digest(HERE/'previous-objective.py')
    target=HERE/'source';assert not target.exists()
    target.mkdir()
    shutil.copytree(BASE/'src',target/'src',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(BASE/'pyproject.toml',target/'pyproject.toml')
    shutil.copy2(HERE/'fixed-objective.py',target/'src/gradscf/model/nnao/training/objective.py')
    changed=[str(p.relative_to(BASE/'src')) for p in (BASE/'src').rglob('*')
        if p.is_file() and p.suffix in {'.py','.json','.dat'}
        and digest(p)!=digest(target/'src'/p.relative_to(BASE/'src'))]
    assert changed==['gradscf/model/nnao/training/objective.py'],changed
    train=CAMPAIGN/'runs/nnao_tzp_l/train'
    assert not (train/'last.npz').exists(),'An accepted checkpoint needs explicit identity migration.'
    progress=json.loads((train/'progress.json').read_text())
    assert progress['update_step']==0 and progress['status']=='failed'
    archive=HERE/'before-restart';archive.mkdir()
    for name in ('progress.json','phase.json','batch_progress.json','metadata.json'):
        if (train/name).exists():shutil.copy2(train/name,archive/name)
    failed=train/'failed-update-000001.npz'
    os.link(failed,archive/failed.name)
    report=dict(status='prepared',base_source_sha256=old_hash,new_source_sha256=source_hash(target),
        changed_files=changed,source_directory=str(target),previous_job_id=104598,
        accepted_updates=0,seed=0,failed_parameters_sha256=digest(failed),
        all_aos_retained=True,scf_tolerances_unchanged=True,implicit_tolerance=1e-8,
        residual_metric='orthonormal_density',finite_differences=False)
    assert source_hash(BASE)==old_hash,'Active L-DZP source changed.'
    (HERE/'deployment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
