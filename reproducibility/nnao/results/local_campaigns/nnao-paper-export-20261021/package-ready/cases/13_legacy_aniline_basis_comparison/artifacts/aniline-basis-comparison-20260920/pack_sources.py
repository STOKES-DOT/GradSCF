"""Package the current module layout for an isolated remote validation run."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path('artifacts/aniline-basis-comparison-20260920')
paths=[]
for directory in ['src','tools','tests']:
 for p in Path(directory).rglob('*'):
  if not p.is_file() or any(x in {'build','__pycache__','.git','.pytest_cache'} for x in p.parts):continue
  if p.name.startswith('._') or p.suffix in {'.pyc','.so','.dylib'}:continue
  if '/integrals/_native/lib/' in str(p):continue
  paths.append(p)
paths.append(Path('pyproject.toml'))
paths += [p for p in root.iterdir() if p.is_file() and p.suffix in {'.py','.json','.sh'} and p.name!='source-hashes.json']
m={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
manifest=root/'source-hashes.json';manifest.write_text(json.dumps(m,indent=2)+'\n')
with tarfile.open('/private/tmp/gradscf-aniline-20260920.tgz','w:gz') as t:
 for p in [*paths,manifest]:t.add(p,arcname=str(p))
print('Source files:',len(m))
