import hashlib,json,tarfile,shutil
from pathlib import Path
import numpy as np
root=Path('/home/user_yuanjiao/private/homefile/GradSCF')
task=root/'artifacts/nnao-joint200-20261011'
m=json.loads((task/'source-manifest.json').read_text());archive=task/m['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==m['archive_sha256']
source=root/'releases'/('joint200-'+m['source_digest'][:16])
source.mkdir(parents=True,exist_ok=False)
with tarfile.open(archive) as tar:
 assert {x.name for x in tar.getmembers()}==set(m['files'])
 assert all(x.isfile() and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts for x in tar.getmembers())
 tar.extractall(source)
for k,v in m['files'].items():assert hashlib.sha256((source/k).read_bytes()).hexdigest()==v,k
checkpoint=root/'artifacts/nnao-joint-reorg-20261010/job-95277/training/best.npz'
selection=root/'artifacts/nnao-stage2-20261007/selection/train.jsonl'
assert hashlib.sha256(selection.read_bytes()).hexdigest()=='9bf275038c21f3775111c749cf43287d1bc494503928a6079cd5d12fa0098f8f'
rows=[json.loads(x) for x in selection.read_text().splitlines() if x.strip()]
assert len(rows)==200 and len({x['structure_id'] for x in rows})==200
with np.load(checkpoint,allow_pickle=False) as z:
 meta=json.loads(str(z['metadata'].item()));assert meta['epoch']==1000 and z['parameters'].shape==(7185,)
 assert np.isfinite(z['parameters']).all()
for src,name in ((checkpoint,'initial.npz'),(selection,'train.jsonl')):
 target=task/name
 if target.exists():raise FileExistsError(target)
 shutil.copy2(src,target)
report=dict(source=str(source),source_digest=m['source_digest'],files_verified=len(m['files']),
 checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),parent_epoch=1000,
 train_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),train_count=200,heldout_count=0)
(task/'deployment.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
