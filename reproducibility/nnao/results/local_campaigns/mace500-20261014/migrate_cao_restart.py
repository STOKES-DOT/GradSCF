"""Explicit, gated source-identity migration for the failed CaO update-20 run.

Dry-run is the default. Stop the training process before using --apply. This
changes only checkpoint metadata.identity.scientific_source_sha256; arrays,
Adam bytes, epoch/batch cursor and all other metadata remain exactly intact.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np

PARENT = '37f917b3d144c8545f0c024910e0f5fa47ae3c753cf0d0ddbae4b85b8bb88ca6'
CHECKPOINTS = ('last.npz', 'best.npz', 'last_full_epoch.npz')
ARRAYS = ('parameters', 'gradient', 'optimizer')
REPRESENTATIVES = (
    'GMTKN55-v2/ALKBDE10/cao', 'GMTKN55-v2/HAL59/BrBr_OPH3',
    'GMTKN55-v2/HEAVYSB11/ge2h6', 'GMTKN55-v2/RG18/kr2',
    'GMTKN55-v2/ADIM6/AM2',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return digest(stream)


def source_hash(root):
    # Same file set, ordering and path encoding as nnao.runtime; no JAX import.
    value = hashlib.sha256()
    package = root / 'src/gradscf'
    require(package.is_dir(), 'Missing source checkout.')
    for path in sorted(package.rglob('*')):
        if path.is_file() and path.suffix in {'.py', '.json', '.dat'}:
            value.update(str(path.relative_to(package)).encode())
            value.update(file_hash(path).encode())
    return value.hexdigest()


def read_checkpoint(path):
    with zipfile.ZipFile(path) as saved:
        require(sorted(saved.namelist()) == sorted(
            [key + '.npy' for key in (*ARRAYS, 'metadata')]),
            f'{path}: unexpected NPZ members.')
        payloads = {}
        for key in ARRAYS:
            with saved.open(key + '.npy') as stream:
                payloads[key] = digest(stream)
        metadata = json.loads(str(np.load(io.BytesIO(saved.read('metadata.npy')),
                                          allow_pickle=False).item()))
        parameters = np.load(io.BytesIO(saved.read('parameters.npy')), allow_pickle=False)
        parameter_hash = hashlib.sha256(parameters.tobytes()).hexdigest()
    return dict(file_sha256=file_hash(path), metadata=metadata,
                array_npy_sha256=payloads, parameter_sha256=parameter_hash)


def check_fd(check, label):
    require(check.get('passed') is True, f'{label}: finite difference did not pass.')
    ad, fd = float(check['ad']), float(check['fd'])
    require(math.isfinite(ad) and math.isfinite(fd), f'{label}: nonfinite derivative.')
    require(abs(ad - fd) <= 2e-6 + 2e-5 * abs(fd), f'{label}: derivative mismatch.')


def validate_gate(gate, checkpoints, new_source, native_hash):
    require(gate.get('passed') is True and gate.get('status') == 'passed',
            'Numerical verification has not passed.')
    require(gate.get('source_sha256') == new_source and
            gate.get('parent_source_sha256') == PARENT, 'Verification source lineage mismatch.')
    require(gate.get('native_sha256') == native_hash, 'Verification native library mismatch.')
    for label, name in [('best', 'best.npz'), ('latest', 'last.npz')]:
        require(gate.get(label + '_parameter_sha256') == checkpoints[name]['parameter_sha256'],
                f'Verification used different {label} model parameters.')
    indexed = {}
    for check in gate['checks']:
        key = check['checkpoint'], check['structure_id']
        require(key not in indexed, f'Duplicate numerical check: {key}.')
        indexed[key] = check
    for identifier in REPRESENTATIVES:
        check = indexed.get(('best', identifier), {})
        require(check.get('converged') is True and check.get('stable') is True and
                check.get('replay_passed') is True, f'Best checkpoint replay failed: {identifier}.')
        energy, reference = float(check['energy_hartree']), float(check['reference_energy_hartree'])
        require(math.isfinite(energy) and math.isfinite(reference) and abs(energy - reference) <= 1e-8,
                f'Best checkpoint energy changed: {identifier}.')
    check = indexed.get(('last', REPRESENTATIVES[0]), {})
    require(check.get('converged') is True and check.get('stable') is True and
            math.isfinite(float(check.get('energy_hartree', float('nan')))),
            'Update-20 CaO is not finite, converged and stable.')
    coefficients = check.get('coefficient_checks', [])
    steps = {float(item['step']) for item in coefficients}
    require(len(steps) >= 2 and all(math.isfinite(h) and h > 0 for h in steps),
            'Require at least two distinct positive coefficient FD step sizes.')
    for item in coefficients:
        check_fd(item, 'Update-20 CaO coefficient')
    check_fd(check.get('mace_parameter_check', {}), 'Update-20 CaO MACE parameter')


def rewrite_metadata(source, destination, new_source):
    before = read_checkpoint(source)
    metadata = deepcopy(before['metadata'])
    metadata['identity']['scientific_source_sha256'] = new_source
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(destination, 'w') as new:
        for member in old.infolist():
            if member.filename == 'metadata.npy':
                buffer = io.BytesIO()
                np.save(buffer, np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
                        allow_pickle=False)
                new.writestr(member, buffer.getvalue())
            else:
                with old.open(member) as reader, new.open(member, 'w', force_zip64=True) as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
    after = read_checkpoint(destination)
    require(after['array_npy_sha256'] == before['array_npy_sha256'], 'Checkpoint array bytes changed.')
    require(after['metadata'] == metadata, 'Unexpected checkpoint metadata mutation.')
    return after


def migrate(args):
    base, root = args.artifact_root.resolve(), args.repo_root.resolve()
    train = base / 'train'
    verification = args.verification or base / 'cao-restart-verification.json'
    source = json.loads((base / 'source.json').read_text())
    new_source = source_hash(root)
    require(source['source_sha256'] == new_source and new_source != PARENT,
            'source.json must identify the verified, modified source checkout.')
    native_hash = file_hash(args.native_library)
    gate = json.loads(verification.read_text())
    checkpoints = {name: read_checkpoint(train / name) for name in CHECKPOINTS}
    identity = checkpoints['last.npz']['metadata']['identity']
    require(identity['scientific_source_sha256'] == PARENT, 'Expected unmigrated parent checkpoint.')
    for name, snapshot in checkpoints.items():
        metadata = snapshot['metadata']
        require(metadata['identity'] == identity, f'{name}: inconsistent scientific identity.')
        require(metadata['epoch'] == 1, f'{name}: expected completed epoch 1.')
        expected_step, expected_cursor = (20, 10) if name == 'last.npz' else (10, 0)
        require(metadata['update_step'] == expected_step and metadata['batch_cursor'] == expected_cursor,
                f'{name}: unexpected optimizer update or batch cursor.')
    require(checkpoints['best.npz']['array_npy_sha256'] ==
            checkpoints['last_full_epoch.npz']['array_npy_sha256'],
            'Best and last-full-epoch arrays disagree.')
    require(bool(identity.get('caches')), 'Checkpoint lacks integral cache provenance.')
    for manifest in identity['caches'].values():
        require(manifest['identity']['native_sha256'] == native_hash, 'Cached native library changed.')
    validate_gate(gate, checkpoints, new_source, native_hash)
    files = sorted(p for p in train.rglob('*') if p.is_file())
    files += sorted(base.glob('source*.json'))
    files += [verification]
    for name in ('smoke-passed.json', 'atomic-rescue-verification.json'):
        if (base / name).exists():
            files.append(base / name)
    files = sorted(set(files))
    require(all(p.is_relative_to(base) and not p.is_symlink() for p in files),
            'Archive inputs must be regular files within the artifact directory.')
    before_hashes = {str(p.relative_to(base)): file_hash(p) for p in files}
    record = dict(format='nnao-cao-checkpoint-migration-v1',
        status='dry_run', parent_source_sha256=PARENT, source_sha256=new_source,
        verification_sha256=file_hash(verification), native_sha256=native_hash,
        checkpoints={name: {key: value for key, value in snapshot.items() if key != 'metadata'}
                     for name, snapshot in checkpoints.items()},
        archived_files=before_hashes,
        changed_field='metadata.identity.scientific_source_sha256',
        preserved='all other metadata and exact parameters.npy, gradient.npy, optimizer.npy bytes')
    if not args.apply:
        return record
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    archive = base / ('migration-archive-' + stamp)
    archive.mkdir(exist_ok=False)
    for relative, expected_hash in before_hashes.items():
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / relative, destination)
        require(file_hash(destination) == expected_hash, f'Archive verification failed: {relative}.')
    staged = archive / 'migrated-checkpoints'
    staged.mkdir()
    for name in CHECKPOINTS:
        updated = rewrite_metadata(archive / 'train' / name, staged / name, new_source)
        record['checkpoints'][name]['migrated_file_sha256'] = updated['file_sha256']
    require(source_hash(root) == new_source, 'Source changed while preparing migration.')
    for relative, expected_hash in before_hashes.items():
        require(file_hash(base / relative) == expected_hash, f'Live file changed during migration: {relative}.')
    record.update(status='prepared', archive=str(archive), created_utc=stamp)
    record_path = archive / 'migration-record.json'
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    replaced = []
    try:
        for name in CHECKPOINTS:
            temporary = train / (name + '.cao-migration.tmp')
            with temporary.open('xb') as stream, (staged / name).open('rb') as reader:
                shutil.copyfileobj(reader, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, train / name)
            replaced.append(name)
            require(file_hash(train / name) == record['checkpoints'][name]['migrated_file_sha256'],
                    f'Installed checkpoint checksum mismatch: {name}.')
    except BaseException:
        for name in replaced:
            shutil.copy2(archive / 'train' / name, train / name)
        record['status'] = 'rolled_back'
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        raise
    record['status'] = 'applied'
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument('--artifact-root', type=Path, default=base)
    parser.add_argument('--repo-root', type=Path, default=base.parent.parent)
    parser.add_argument('--verification', type=Path)
    parser.add_argument('--native-library', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Archive originals and migrate; otherwise read-only.')
    args = parser.parse_args()
    result = migrate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
