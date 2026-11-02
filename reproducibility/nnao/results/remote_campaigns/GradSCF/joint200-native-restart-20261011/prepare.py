"""Snapshot an accepted checkpoint and pin validated source before stopping a job."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

root = Path('/home/user_yuanjiao/private/homefile/GradSCF')
task = root/'artifacts/joint200-native-restart-20261011'
source = root/'releases/joint200-native-restart-20261011'
archive = task/'archive'
archive.mkdir(exist_ok=False)
old_dir = root/'artifacts/joint200-cache-20261011/training'
for name in ('last.npz', 'best.npz', 'metadata.json', 'progress.json', 'history.json', 'runtime.json'):
    shutil.copy2(old_dir/name, archive/name)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
with np.load(archive/'last.npz', allow_pickle=False) as last:
    metadata = json.loads(str(last['metadata'].item()))
    assert last['parameters'].shape == last['gradient'].shape == (7185,)
    assert np.isfinite(last['parameters']).all() and np.isfinite(last['gradient']).all()
    assert len(metadata['molecules']) == 200
    assert all(r['converged'] and r['stable'] for r in metadata['molecules'])
with np.load(archive/'best.npz', allow_pickle=False) as best:
    best_meta = json.loads(str(best['metadata'].item()))
    assert best_meta['identity'] == metadata['identity'] and best_meta['epoch'] <= metadata['epoch']
files = {}
for path in sorted(source.rglob('*')):
    relative = path.relative_to(source)
    if (not path.is_file() or any(p in {'__pycache__', '.git', 'build', 'lib'} for p in relative.parts)
            or path.suffix == '.pyc'):
        continue
    files[str(relative)] = digest(path)
native = source/'src/gradscf/integrals/_native/lib/libgradscf_integrals.so'
assert digest(native) == '7ef5cefc5098dfdfb5ffc0fdfb17b3b9cc896dc82c1a3b8c1f798d28b4310b6b'
for name in ('train_nnao_joint_ad.py', 'train_nnao_joint.py', 'train_nnao_pilot.py', 'optimize_methane_nnao.py'):
    assert digest(source/'tools'/name) == digest(root/'releases/joint200-cachefix-aca4a20b0bd00be2/tools'/name), name
deployment = dict(source=str(source), files=files, native_sha256=digest(native),
    old_native_sha256=metadata['identity']['native_library_sha256'],
    archive_files={p.name: digest(p) for p in archive.iterdir() if p.is_file()},
    accepted_epoch=metadata['epoch'], loss_hartree=metadata['loss_hartree'], old_job=95681,
    target_epochs=1000, native_threads=16, optimizer_preserved=True,
    validation_job=95883)
(task/'deployment.json').write_text(json.dumps(deployment, indent=2)+'\n')
print(json.dumps({k: v for k, v in deployment.items() if k not in {'files', 'archive_files'}}, indent=2))
