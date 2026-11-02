"""Preserve the stopped 24-structure run's accepted state, never its candidate."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

root = Path('/home/user_yuanjiao/private/homefile/GradSCF')
source = root/'artifacts/nnao-joint-native-24-20261010-v3/job-95230/training'
target = root/'artifacts/nnao-joint-reorg-20261010/previous-job-95230'
target.mkdir(parents=True, exist_ok=False)
record = {'job_id': 95230, 'original_directory': str(source), 'checkpoints': {}}
for name in ('last.npz', 'best.npz'):
    shutil.copy2(source/name, target/name)
    with np.load(target/name, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata'].item()))
        assert np.isfinite(z['parameters']).all() and np.isfinite(z['gradient']).all()
    record['checkpoints'][name] = dict(epoch=metadata['epoch'], loss_hartree=metadata['loss_hartree'],
        sha256=hashlib.sha256((target/name).read_bytes()).hexdigest())
for name in ('metadata.json','runtime.json','progress.json','history.json','parent-parity.json','gradient-checks.json'):
    if (source/name).is_file(): shutil.copy2(source/name, target/name)
(target/'preservation.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
