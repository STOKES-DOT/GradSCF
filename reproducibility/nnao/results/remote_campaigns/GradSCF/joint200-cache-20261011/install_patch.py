import hashlib,json,shutil
from pathlib import Path
root=Path('/home/user_yuanjiao/private/homefile/GradSCF')
task=root/'artifacts/joint200-cache-20261011'
m=json.loads((task/'source-manifest.json').read_text())
source=root/'releases'/('joint200-cachefix-'+m['source_digest'][:16])
shutil.copytree(root/'artifacts/nnao-joint200-20261011/job-95573/source',source,
                ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
for name in m['patches']:shutil.copy2(task/'patch'/name,source/name)
for name,digest in m['files'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
library=source/'src/gradscf/integrals/_native/lib/libgradscf_integrals.so'
meta=json.loads((root/'artifacts/nnao-joint200-20261011/training/metadata.json').read_text())
assert hashlib.sha256(library.read_bytes()).hexdigest()==meta['native_library_sha256']
report=dict(source=str(source),source_digest=m['source_digest'],files_verified=len(m['files']),
             native_library_sha256=meta['native_library_sha256'])
(task/'deployment.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
