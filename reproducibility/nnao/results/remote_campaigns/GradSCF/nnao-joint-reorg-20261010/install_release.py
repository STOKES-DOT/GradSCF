"""Extract an exact committed archive into its formal server version directory."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path('/home/user_yuanjiao/private/homefile/GradSCF')
task = root/'artifacts/nnao-joint-reorg-20261010'
manifest = json.loads((task/'release-manifest.json').read_text())
archive = task/manifest['archive_name']
if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['archive_sha256']:
    raise RuntimeError('Archive transfer checksum mismatch.')
target = root/'releases'/manifest['commit']
target.mkdir(parents=True, exist_ok=False)
with tarfile.open(archive, 'r:gz') as bundle:
    entries = [member for member in bundle.getmembers() if member.isfile()]
    if {member.name for member in entries} != set(manifest['files']):
        raise RuntimeError('Archive file list differs from manifest.')
    bundle.extractall(target)
for name, digest in manifest['files'].items():
    if hashlib.sha256((target/name).read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'Extracted source checksum mismatch: {name}')
(task/'server-sync.json').write_text(json.dumps(dict(commit=manifest['commit'],
    source_directory=str(target), files_verified=len(manifest['files']),
    archive_sha256=manifest['archive_sha256']), indent=2)+'\n')
print((task/'server-sync.json').read_text())
