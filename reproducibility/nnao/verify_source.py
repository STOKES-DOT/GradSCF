"""Verify that the archived scientific files match the remote training snapshot."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(Path(__file__).with_name('source-manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = root/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            raise RuntimeError(f'Training source mismatch: {relative}')
    print(f"Verified {len(manifest['files'])} training-source files.")
    print('Scientific payload:', manifest['archived_scientific_payload_sha256'])


if __name__ == '__main__':
    main()
