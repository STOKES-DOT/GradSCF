"""Gated source-identity migration from accepted update 29; no finite differences.

Default is read-only. With training stopped, --apply archives all original train
files and source/gate evidence, then changes only the source identity in last,
best and last_full_epoch. The failed update-30 candidate is never promoted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

from migrate_cao_restart import file_hash, read_checkpoint, require, rewrite_metadata, source_hash

PARENT = '0a5510b43d00e003eb6d0f16258d48d14a5553d05bfda5b960c6467da85fefc5'
CHECKPOINTS = ('last.npz', 'best.npz', 'last_full_epoch.npz')
FAILED = 'failed-update-000030.npz'
REPRESENTATIVES = (
    'GMTKN55-v2/RG18/kr3', 'GMTKN55-v2/ALKBDE10/cao', 'GMTKN55-v2/HAL59/BrBr_OPH3',
)


def failed_candidate(path):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata'].item()))
        parameters = saved['parameters']
        require(np.isfinite(parameters).all(), 'Failed candidate parameters are nonfinite.')
        parameter_hash = hashlib.sha256(parameters.tobytes()).hexdigest()
    require(metadata.get('accepted_updates') == 29 and
            metadata.get('phase') == 'candidate_batch_check',
            'Expected unaccepted candidate evaluated after update 29.')
    require(metadata.get('structure_id') == REPRESENTATIVES[0],
            'Failed candidate snapshot does not identify Kr3.')
    return dict(file_sha256=file_hash(path), parameter_sha256=parameter_hash, metadata=metadata)


def validate_gate(gate, checkpoints, failed, new_source, native_hash):
    require(gate.get('passed') is True and gate.get('status') == 'passed',
            'Analytic-backward verification has not passed.')
    require(gate.get('finite_differences') is False,
            'Restart requires the explicitly requested no-finite-difference verification.')
    require(gate.get('source_sha256') == new_source and
            gate.get('parent_source_sha256') == PARENT, 'Verification source lineage mismatch.')
    require(gate.get('native_sha256') == native_hash, 'Verification native library mismatch.')
    expected_hashes = dict(last=checkpoints['last.npz']['parameter_sha256'],
                           best=checkpoints['best.npz']['parameter_sha256'],
                           failed=failed['parameter_sha256'])
    require(gate.get('parameter_sha256') == expected_hashes,
            'Verification does not match accepted/best/failed parameter vectors.')
    indexed = {}
    for check in gate['checks']:
        key = check['checkpoint'], check['structure_id']
        require(key not in indexed, f'Duplicate numerical check: {key}.')
        indexed[key] = check
    expected = {(label, identifier) for label in ('last', 'best') for identifier in REPRESENTATIVES}
    expected.add(('failed', REPRESENTATIVES[0]))
    require(set(indexed) == expected, 'Expected exactly the seven approved restart checks.')
    for key, check in indexed.items():
        require(all(check.get(field) is True for field in
                    ('converged', 'stable', 'finite_gradient', 'finite_network_gradient')),
                f'Forward/stability/analytic-backward check failed: {key}.')
        energy = float(check['energy_hartree'])
        require(math.isfinite(energy), f'Nonfinite energy: {key}.')
        if key[0] == 'best':
            reference = float(check['reference_energy_hartree'])
            require(check.get('replay_passed') is True and math.isfinite(reference) and
                    abs(energy - reference) <= 1e-9, f'Best energy replay changed: {key}.')


def migrate(args):
    base, root = args.artifact_root.resolve(), args.repo_root.resolve()
    train = base / 'train'
    verification = (args.verification or base / 'kr3-restart-verification.json').resolve()
    source_bytes = (base / 'source.json').read_bytes()
    source = json.loads(source_bytes)
    new_source = source_hash(root)
    require(source['source_sha256'] == new_source and new_source != PARENT,
            'source.json must identify the verified, modified checkout.')
    native_hash = file_hash(args.native_library)
    gate_bytes = verification.read_bytes()
    gate = json.loads(gate_bytes)
    verification_hash = hashlib.sha256(gate_bytes).hexdigest()
    checkpoints = {name: read_checkpoint(train / name) for name in CHECKPOINTS}
    failed = failed_candidate(train / FAILED)
    identity = checkpoints['last.npz']['metadata']['identity']
    require(identity['scientific_source_sha256'] == PARENT, 'Expected unmigrated parent checkpoint.')
    for name, snapshot in checkpoints.items():
        metadata = snapshot['metadata']
        require(metadata['identity'] == identity, f'{name}: inconsistent scientific identity.')
        require(metadata['epoch'] == 2, f'{name}: expected completed epoch 2.')
        step, cursor = (29, 9) if name == 'last.npz' else (20, 0)
        require(metadata['update_step'] == step and metadata['batch_cursor'] == cursor,
                f'{name}: unexpected optimizer update or batch cursor.')
    require(checkpoints['best.npz']['array_npy_sha256'] ==
            checkpoints['last_full_epoch.npz']['array_npy_sha256'],
            'Best and last-full-epoch array bytes disagree.')
    require(bool(identity.get('caches')), 'Checkpoint lacks integral cache provenance.')
    for manifest in identity['caches'].values():
        require(manifest['identity']['native_sha256'] == native_hash, 'Cached native library changed.')
    validate_gate(gate, checkpoints, failed, new_source, native_hash)
    files = set(p for p in train.rglob('*') if p.is_file())
    for pattern in ('source*.json', '*verification*.json', '*gate*.json', '*passed*.json'):
        files.update(base.glob(pattern))
    files.add(verification)
    require(all(p.is_relative_to(base) and not p.is_symlink() for p in files),
            'Archive evidence must be regular files within the artifact directory.')
    before_hashes = {str(p.relative_to(base)): file_hash(p) for p in sorted(files)}
    for name, snapshot in {**checkpoints, FAILED: failed}.items():
        require(before_hashes['train/' + name] == snapshot['file_sha256'],
                f'Checkpoint changed during verification: {name}.')
    require(before_hashes['source.json'] == hashlib.sha256(source_bytes).hexdigest(),
            'Source record changed during inspection.')
    require(before_hashes[str(verification.relative_to(base))] == verification_hash,
            'Verification report changed during inspection.')
    record = dict(format='nnao-kr3-checkpoint-migration-v1', status='dry_run',
        parent_source_sha256=PARENT, source_sha256=new_source, finite_differences=False,
        native_sha256=native_hash, verification_sha256=verification_hash,
        resume=dict(epoch=2, update_step=29, batch_cursor=9, failed_candidate_promoted=False),
        failed_candidate=failed,
        checkpoints={name: {key: value for key, value in snapshot.items() if key != 'metadata'}
                     for name, snapshot in checkpoints.items()},
        archived_files=before_hashes, changed_field='metadata.identity.scientific_source_sha256',
        preserved='all other metadata and exact parameters.npy, gradient.npy, optimizer.npy bytes')
    if not args.apply:
        return record
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    archive = base / ('kr3-migration-archive-' + stamp)
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
    require(file_hash(args.native_library) == native_hash, 'Native library changed during migration.')
    for relative, expected_hash in before_hashes.items():
        require(file_hash(base / relative) == expected_hash, f'Live file changed during migration: {relative}.')
    record.update(status='prepared', archive=str(archive), created_utc=stamp)
    record_path = archive / 'migration-record.json'
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    replaced = []
    try:
        for name in CHECKPOINTS:
            temporary = train / (name + '.kr3-migration.tmp')
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
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
        raise
    record['status'] = 'applied'
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument('--artifact-root', type=Path, default=base)
    parser.add_argument('--repo-root', type=Path, default=base.parent.parent)
    parser.add_argument('--verification', type=Path)
    parser.add_argument('--native-library', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Archive and migrate; otherwise read-only.')
    print(json.dumps(migrate(parser.parse_args()), indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
