"""Check frozen run sources and the unrelated original training checkout."""
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
for root, manifest in ((args.source, args.manifest), (args.original, args.baseline)):
    for name, expected in json.loads(manifest.read_text())['files'].items():
        if sha(root/name) != expected:
            raise RuntimeError(f'Frozen source changed: {root/name}')
if sha(args.original/'src/gradscf/integrals/_native/lib/libgradscf_integrals.so') != 'b0e65d075c537a3e56e499d7e8df85b8b493ed0e71b6f86ab82cda196a426fd4':
    raise RuntimeError('Original native library changed.')
if sha(args.checkpoint) != 'ba93fb40efca3268962c7501f82627e188ed9509b6a73bd27a2df2170a3495de':
    raise RuntimeError('Original best975 checkpoint changed.')
if sha(args.structures) != 'cddae74070ee5053656c1e105be45e0b8dbd98ab15ec347d0fc87e4233405899':
    raise RuntimeError('Original 24 structures changed.')
record = dict(original_sources_unchanged=True, source_manifest_sha256=sha(args.manifest),
    structures_sha256=sha(args.structures), checkpoint_sha256=sha(args.checkpoint))
if args.native:
    record['native_library_sha256'] = sha(args.source/'src/gradscf/integrals/_native/lib/libgradscf_integrals.so')
print(json.dumps(record, indent=2))
