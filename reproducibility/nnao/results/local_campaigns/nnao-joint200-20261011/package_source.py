"""Archive current tracked sources plus the new regression; exclude unrelated WIP."""
import hashlib,json,subprocess,tarfile
from pathlib import Path
root=Path.cwd();out=root/'artifacts/nnao-joint200-20261011'
names=subprocess.check_output(['git','ls-files','-z']).decode().split('\0')
names=sorted(set(filter(None,names))|{'tests/test_df_rhf_stability.py'})
files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
archive=out/f'source-{digest[:16]}.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name in names:tar.add(root/name,arcname=name,recursive=False)
manifest=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
 source_digest=digest,files=files,archive=archive.name,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(out/'source-manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='files'}))
