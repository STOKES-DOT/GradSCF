"""Isolate the one-file fix and migrate only the failed lane's accepted state."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np


ROOT = Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021/repairs/ne6-20261023/source')
CAMPAIGN = Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021')
HERE = CAMPAIGN/'repairs/kr3-20261024'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def source_hash(root):
    value = hashlib.sha256(); package = root/'src/gradscf'
    for path in sorted(package.rglob('*')):
        if path.is_file() and path.suffix in {'.py', '.json', '.dat'}:
            value.update(str(path.relative_to(package)).encode())
            value.update(digest(path).encode())
    return value.hexdigest()


def migrate_checkpoint(family, old_hash, new_hash, update, cursor):
    output = CAMPAIGN/'runs'/family/'train'
    checkpoint = output/'last.npz'
    old_checkpoint_hash = digest(checkpoint)
    archive = HERE/f'{family}-accepted-update{update}-original.npz'
    os.link(checkpoint, archive)
    for name in ('progress.json', 'phase.json', 'history.json'):
        shutil.copy2(output/name, HERE/(family+'-before-'+name))
    array_hashes = {}
    temporary = checkpoint.with_name('last-ne6-migration.npz')
    with zipfile.ZipFile(checkpoint) as source, zipfile.ZipFile(temporary, 'w', allowZip64=True) as destination:
        metadata = json.loads(str(np.lib.format.read_array(source.open('metadata.npy'), allow_pickle=False).item()))
        assert metadata['update_step'] == update and metadata['batch_cursor'] == cursor
        assert metadata['identity']['scientific_source_sha256'] == old_hash
        before = json.loads(json.dumps(metadata))
        metadata['identity']['scientific_source_sha256'] = new_hash
        reverted = json.loads(json.dumps(metadata))
        reverted['identity']['scientific_source_sha256'] = old_hash
        assert reverted == before
        for info in source.infolist():
            if info.filename == 'metadata.npy':
                data = io.BytesIO()
                np.save(data, np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
                destination.writestr(info, data.getvalue())
            else:
                with source.open(info) as incoming, destination.open(info, 'w', force_zip64=True) as outgoing:
                    digestor = hashlib.sha256()
                    for block in iter(lambda:incoming.read(1024*1024), b''):
                        digestor.update(block);outgoing.write(block)
                    array_hashes[info.filename] = digestor.hexdigest()
    with zipfile.ZipFile(temporary) as saved:
        for name, expected in array_hashes.items():
            digestor = hashlib.sha256()
            with saved.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024), b''):digestor.update(block)
            assert digestor.hexdigest() == expected, name
    os.replace(temporary, checkpoint)
    return dict(accepted_update=update,batch_cursor=cursor,optimizer_reset=False,
        original_checkpoint_sha256=old_checkpoint_hash,
        migrated_checkpoint_sha256=digest(checkpoint),unchanged_npy_payloads=array_hashes,
        checkpoint_identity_changes=['scientific_source_sha256'],old_archive=str(archive))


def main():
    target = HERE/'source'
    assert not target.exists() and not (HERE/'deployment.json').exists()
    config = json.loads((CAMPAIGN/'repairs/ne6-20261023/deployment.json').read_text())
    old_hash = source_hash(ROOT)
    assert old_hash == config['new_source_sha256']
    original = ROOT/'src/gradscf/scf/stability.py'
    assert digest(original) == digest(HERE/'previous-stability.py')
    target.mkdir()
    shutil.copytree(ROOT/'src', target/'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'pyproject.toml', target/'pyproject.toml')
    shutil.copy2(HERE/'fixed-stability.py', target/'src/gradscf/scf/stability.py')
    new_hash = source_hash(target)
    changed = [str(p.relative_to(ROOT/'src')) for p in (ROOT/'src').rglob('*')
               if p.is_file() and p.suffix in {'.py', '.json', '.dat'}
               and digest(p) != digest(target/'src'/p.relative_to(ROOT/'src'))]
    assert changed == ['gradscf/scf/stability.py'], changed
    checkpoints={family:migrate_checkpoint(family,old_hash,new_hash,update,cursor)
        for family,update,cursor in (('nnao_dzp_s',470,0),('nnao_dzp_m',16,6))}
    assert source_hash(ROOT) == old_hash, 'The other two training lanes must retain their frozen source.'
    report = dict(status='prepared', old_source_sha256=old_hash, new_source_sha256=new_hash,
        changed_files=changed, source_directory=str(target),
        checkpoints=checkpoints,stability_tolerance=1e-7,finite_differences=False)
    (HERE/'deployment.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':main()
