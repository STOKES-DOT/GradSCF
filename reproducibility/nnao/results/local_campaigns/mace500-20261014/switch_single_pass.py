"""Archive a stopped run, then change only its last checkpoint's source identity."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil

from migrate_cao_restart import file_hash,read_checkpoint,rewrite_metadata,source_hash

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
PARENT='55b628b381fe84ee226c2882a1b095bce02ba0e32fc9634dec9865b93ad65e7f'
RECORD=BASE/'single-pass-switch.json'
NATIVE=ROOT/'src/gradscf/integrals/_native/lib/libgradscf_integrals.so'

def inventory():
    package=ROOT/'src/gradscf'
    return {str(p.relative_to(package)):file_hash(p) for p in sorted(package.rglob('*'))
            if p.is_file() and p.suffix in {'.py','.json','.dat'}}

def write_record(record):
    temporary=RECORD.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    os.replace(temporary,RECORD)

def archive():
    assert source_hash(ROOT)==PARENT,'Unexpected parent sources'
    before=read_checkpoint(BASE/'train/last.npz')
    metadata=before['metadata']
    assert metadata['identity']['scientific_source_sha256']==PARENT
    native=file_hash(NATIVE)
    assert all(m['identity']['native_sha256']==native for m in metadata['identity']['caches'].values())
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    destination=BASE/f'single-pass-archive-{stamp}'
    destination.mkdir(exist_ok=False)
    files=sorted(p for p in (BASE/'train').rglob('*') if p.is_file())
    files+=list(BASE.glob('source*.json'))
    for name in ['restart-kr3-98645.log','restart_kr3.py','restart_kr3.slurm']:
        if (BASE/name).exists():files.append(BASE/name)
    hashes={}
    for path in files:
        target=destination/path.relative_to(BASE);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        digest=file_hash(path);assert file_hash(target)==digest
        hashes[str(path.relative_to(BASE))]=digest
    assert hashes['train/last.npz']==before['file_sha256'],'Checkpoint changed while archiving'
    original=inventory()
    for name in ['model/nnao/training/minibatch.py','model/nnao/workflows/fixed_pool.py']:
        target=destination/'src/gradscf'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'src/gradscf'/name,target)
    record=dict(status='archived',archive=str(destination),parent_job_id=98645,parent_source_sha256=PARENT,
        native_sha256=native,scientific_files_before=original,archived_files=hashes,
        checkpoint={key:before[key] for key in ['file_sha256','array_npy_sha256','parameter_sha256']},
        resume={key:metadata[key] for key in ['epoch','update_step','batch_cursor','loss_hartree']})
    write_record(record)
    print(json.dumps({key:record[key] for key in ['status','archive','resume']}),flush=True)

def migrate():
    record=json.loads(RECORD.read_text());assert record['status']=='archived'
    expected=json.loads((BASE/'single-pass-source.json').read_text())
    current=inventory();old=record['scientific_files_before']
    changed=sorted(k for k in set(old)|set(current) if old.get(k)!=current.get(k))
    assert changed==sorted(expected['changed_files']),changed
    assert current==expected['scientific_files'],'Uploaded sources differ from tested checkout'
    source=source_hash(ROOT);assert source==expected['source_sha256']
    assert file_hash(NATIVE)==record['native_sha256']
    checkpoint=BASE/'train/last.npz'
    assert file_hash(checkpoint)==record['checkpoint']['file_sha256']
    destination=Path(record['archive'])
    before=read_checkpoint(checkpoint)
    staged=destination/'migrated-last.npz'
    after=rewrite_metadata(checkpoint,staged,source)
    assert after['array_npy_sha256']==before['array_npy_sha256']
    updated=json.loads((destination/'source.json').read_text())
    updated.update(source_sha256=source,parent_source_sha256=PARENT,
                   training_policy='single_pass',check_parameter_gradients=False)
    record.update(status='prepared',source_sha256=source,changed_scientific_files=changed,
        migrated_checkpoint_sha256=after['file_sha256'],preserved_array_npy_sha256=after['array_npy_sha256'])
    write_record(record)
    os.replace(staged,checkpoint)
    (BASE/'source.json').write_text(json.dumps(updated,indent=2)+'\n')
    assert file_hash(checkpoint)==after['file_sha256']
    for name in ['best.npz','last_full_epoch.npz']:
        if (BASE/'train'/name).exists():
            assert file_hash(BASE/'train'/name)==record['archived_files']['train/'+name]
    record['status']='applied';write_record(record)
    print(json.dumps({key:record[key] for key in ['status','source_sha256','resume']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['archive','migrate'])
    args=parser.parse_args()
    archive() if args.action=='archive' else migrate()
