"""Validate a committed release and the unrelated frozen training checkout."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--original', type=Path, required=True)
parser.add_argument('--baseline', type=Path, required=True)
parser.add_argument('--structures', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--native', action='store_true')
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
for name, expected in manifest['files'].items():
    if sha(args.source/name) != expected:
        raise RuntimeError(f'Release file differs from committed archive: {name}')
for folder in ('src','tools','tests'):
    for path in (args.source/folder).rglob('*.py'):
        if '__pycache__' not in path.parts and str(path.relative_to(args.source)) not in manifest['files']:
            raise RuntimeError(f'Unexpected Python source: {path}')
for name, expected in json.loads(args.baseline.read_text())['files'].items():
    if sha(args.original/name) != expected:
        raise RuntimeError(f'Unrelated training checkout changed: {name}')
if sha(args.original/'src/gradscf/integrals/_native/lib/libgradscf_integrals.so') != 'b0e65d075c537a3e56e499d7e8df85b8b493ed0e71b6f86ab82cda196a426fd4':
    raise RuntimeError('Unrelated native library changed.')
if sha(args.checkpoint) != 'ba93fb40efca3268962c7501f82627e188ed9509b6a73bd27a2df2170a3495de':
    raise RuntimeError('Original best975 checkpoint changed.')
if sha(args.structures) != 'cddae74070ee5053656c1e105be45e0b8dbd98ab15ec347d0fc87e4233405899':
    raise RuntimeError('The 24-structure selection changed.')
report = dict(commit=manifest['commit'], tree=manifest['tree'],
    release_files_verified=len(manifest['files']), source_manifest_sha256=sha(args.manifest),
    tools_verified=[name for name in manifest['files'] if name.startswith('tools/')],
    original_training_checkout_unchanged=True, structures_sha256=sha(args.structures),
    checkpoint_sha256=sha(args.checkpoint))
if args.native:
    report['native_library_sha256'] = sha(args.source/'src/gradscf/integrals/_native/lib/libgradscf_integrals.so')
print(json.dumps(report, indent=2, sort_keys=True))
