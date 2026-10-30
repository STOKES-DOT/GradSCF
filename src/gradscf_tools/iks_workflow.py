"""CPU workflow orchestration; all electronic structure remains in GradSCF."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .pbe_recording import canonical_hash


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_teacher_jobs(input_root):
    root = Path(input_root)
    qualification = json.loads((root/'qualification/qualification.json').read_text())
    if (qualification.get('status') != 'qualified' or not
            (qualification.get('local_file_provenance') or {}).get('archive_sha256_verified')):
        raise ValueError('A teacher batch requires the verified complete source archive')
    ids = qualification['molecule_ids']
    rows = {row['molecule_id']:row for row in qualification['rows']}
    if len(ids) != 30 or len(set(ids)) != 30 or set(rows) != set(ids):
        raise ValueError('Require exactly30 distinct frozen raw inputs')
    jobs = []
    for name in ids:
        row = rows[name]
        npz = root/'raw_subset'/row['npz']
        metadata = npz.with_suffix('.metadata.json')
        if row['qualification']['status'] != 'qualified' or not metadata.is_file():
            raise ValueError('Every raw input needs physical qualification and metadata')
        if file_hash(npz) != row['npz_sha256']:
            raise ValueError('Raw input SHA256 hash changed after qualification')
        jobs.append(dict(molecule_id=name, sample_id=row['sample_id'], npz=str(npz),
                         metadata=str(metadata), npz_sha256=row['npz_sha256']))
    return jobs


def run_teacher_batch(input_root, output_root, protocol_path, source_root, *, timeout=600, seed_root=None):
    """Sequential subprocesses bound JAX compiler residency and retain failures."""
    jobs = plan_teacher_jobs(input_root)
    output, source = Path(output_root), Path(source_root)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(protocol_path).read_text())
    body = {key:value for key,value in protocol.items() if key != 'sha256'}
    if protocol['sha256'] != canonical_hash(body):
        raise ValueError('Reference protocol changed after declaration')
    split = json.loads((Path(input_root)/'qualification/sample_manifest.json').read_text())['split']
    roles = {name:role for role,names in split.items() for name in names}
    rows = []
    manifest_path = output/'reference_manifest.json'
    generator = source/'tools/cafes_wuyang_teacher.py'
    generator_hash = file_hash(generator)
    for job in jobs:
        name = job['molecule_id']
        folder = output/name
        folder.mkdir(exist_ok=True)
        target = folder/'teacher.json'
        row = dict(molecule_id=name, sample_id=job['sample_id'], split=roles[name],
                   raw_npz_sha256=job['npz_sha256'], teacher_json=str(target))
        if target.exists():
            prior = json.loads(target.read_text())
            reusable = (prior.get('valid') and prior.get('workflow_protocol_sha256') == protocol['sha256']
                        and prior.get('raw_npz_sha256') == job['npz_sha256']
                        and prior.get('generator_sha256') == generator_hash)
            if not reusable:
                raise ValueError('Preserve an earlier attempt; use a new batch output directory')
            row.update(valid=True, reused=True, teacher_json_sha256=file_hash(target))
        else:
            command = [sys.executable, str(generator), '--sample-npz', job['npz'],
                '--sample-meta', job['metadata'], '--output', str(target),
                '--max-memory-mb','6000', '--grid-level',str(protocol['grid_level']),
                '--block-size','4096','--guide','pbe','--density-metric',protocol['density_metric'],
                '--method',protocol.get('method','wu_yang'),
                '--regularization',str(protocol['regularization'])]
            if seed_root is not None:
                seed = Path(seed_root)/name/'teacher.json'
                if seed.is_file() and json.loads(seed.read_text()).get('gap',0.) > protocol.get('gap_min_hartree',1e-7):
                    command += ['--initial-teacher',str(seed)]
            started = time.perf_counter()
            with (folder/'stdout.log').open('w') as stdout, (folder/'stderr.log').open('w') as stderr:
                try:
                    result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            row.update(exit_code=code, elapsed_seconds=time.perf_counter()-started, reused=False, valid=False)
            if target.exists():
                data = json.loads(target.read_text())
                data.update(workflow_protocol_sha256=protocol['sha256'], raw_npz_sha256=job['npz_sha256'],
                            generator_sha256=generator_hash)
                target.write_text(json.dumps(data,indent=2)+'\n')
                q = data['quality_settings']
                compatible = (q['regularization'] == protocol['regularization']
                    and q['gradient_tolerance'] == protocol['gradient_max']
                    and q['density_error_kind'] == protocol['density_metric']
                    and data['grid_level'] == protocol['grid_level']
                    and data.get('inverse_method','wu_yang') == protocol.get('method','wu_yang')
                    and q['diagnostic_tolerance'] == protocol['density_tolerance'])
                row.update(valid=bool(code == 0 and compatible and data['valid']),
                    teacher_json_sha256=file_hash(target),
                    relative_density_error=data.get('relative_density_error'),
                    absolute_density_error=data.get('density_error'), gap=data.get('gap'),
                    gradient_norm=data.get('gradient_norm'))
        rows.append(row)
        complete = len(rows) == len(jobs)
        record = dict(status='qualified' if complete and all(r['valid'] for r in rows) else
                             ('failed' if complete else 'preparing'),
            protocol_sha256=protocol['sha256'], generator_sha256=generator_hash,
            source_root=str(source), input_qualification_sha256=file_hash(
                Path(input_root)/'qualification/qualification.json'),
            split=split, rows=rows, completed=len(rows), total=len(jobs))
        manifest_path.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(molecule_id=name, valid=row['valid'], completed=len(rows), total=len(jobs))),flush=True)
    return record


def select_references(manifest, *, splits=('train','validation'), frozen_winner=None):
    """Keep held-out references sealed until a matching winner is frozen."""
    if manifest.get('status') != 'qualified':
        raise ValueError('All30 references must qualify before baseline or fitting')
    split = manifest['split']
    if set(split) != {'train','validation','test'} or {k:len(v) for k,v in split.items()} != {
            'train':20,'validation':5,'test':5}:
        raise ValueError('Require the frozen20/5/5 split')
    expected = {name:role for role,names in split.items() for name in names}
    rows = manifest['rows']
    if len(expected) != 30 or len(rows) != 30 or len({r['molecule_id'] for r in rows}) != 30:
        raise ValueError('Reference IDs and split roles must be unique')
    if any(not r['valid'] or expected.get(r['molecule_id']) != r['split'] for r in rows):
        raise ValueError('Reference validity or split role disagrees with the frozen manifest')
    requested = set(splits)
    if not requested or requested-{'train','validation','test'}:
        raise ValueError('Select explicit train, validation or test roles')
    if 'test' in requested and (not frozen_winner or frozen_winner.get('status') != 'frozen'
            or frozen_winner.get('protocol_sha256') != manifest['protocol_sha256']):
        raise ValueError('The test set is sealed until a matching winner is frozen')
    return [dict(row) for row in rows if row['split'] in requested]
