"""Freeze complete source/tools/tests without local binaries or experiment data."""
import hashlib
import json
from pathlib import Path
import shutil

task = Path(__file__).resolve().parent
root = task.parents[1]
target = task/'source'
if target.exists():
    raise FileExistsError('Snapshot is immutable; select a new directory.')
files = {}
for directory in ('src', 'tools', 'tests'):
    for path in sorted((root/directory).rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in {'__pycache__', '.git', '.pytest_cache'} or part.endswith('.egg-info') for part in relative.parts):
            continue
        if '_native' in relative.parts and any(part in {'build', 'lib'} for part in relative.parts[relative.parts.index('_native')+1:-1]):
            # Keep vendor/pyscf/.../lib sources; only direct native build/lib outputs are excluded.
            if relative.parts[relative.parts.index('_native')+1] in {'build', 'lib'}:
                continue
        if path.suffix in {'.pyc', '.pyo'} or path.name == '.DS_Store':
            continue
        destination = target/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        files[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
for name in ('pyproject.toml', 'README.md', 'AGENTS.md', 'LICENSE'):
    source = root/name
    if source.is_file():
        shutil.copy2(source, target/name)
        files[name] = hashlib.sha256(source.read_bytes()).hexdigest()
(task/'source-manifest.json').write_text(json.dumps({'files': files}, indent=2, sort_keys=True)+'\n')
print(json.dumps(dict(files=len(files), bytes=sum((target/p).stat().st_size for p in files))))
