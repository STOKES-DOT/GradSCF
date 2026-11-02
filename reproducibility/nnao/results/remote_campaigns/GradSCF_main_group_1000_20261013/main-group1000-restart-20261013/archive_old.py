"""Archive the superseded eight-job campaign before explicitly cancelling it."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

base = Path('/home/user_yuanjiao/private/homefile/GradSCF_basis_grid_20261013')
source = base/'artifacts/basis-grid-20261013'
out = base/'artifacts/retired-200-grid-20261013'
out.mkdir(exist_ok=False)
jobs = json.loads((source/'submitted-jobs.json').read_text())
assert {r['job_id'] for r in jobs} == set(range(96633, 96641))
assert all(r['job_id'] != 96510 for r in jobs)
records = []
for row in jobs:
    status = subprocess.run(['scontrol', 'show', 'job', str(row['job_id']), '-o'],
                            text=True, capture_output=True, check=True).stdout
    assert f"JobName={row['family']} " in status
    assert str(base.resolve()) in status or str(base) in status
    files = [source/row['script'].split('/')[-1], source/f"{row['family']}-{row['job_id']}.log"]
    run = source/'runs'/row['family']
    if run.exists():
        files += [p for p in run.rglob('*') if p.is_file() and 'cache' not in p.relative_to(run).parts]
    saved = []
    for p in files:
        if not p.exists():
            continue
        relative = p.relative_to(source)
        target = out/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = p.read_bytes()
        target.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == sha
        saved.append(dict(path=str(relative), bytes=len(payload), sha256=sha))
    records.append(dict(**row, slurm_before=status.strip(), archived=saved))
report = dict(time=datetime.datetime.now().isoformat(), retained_job=96510,
    action='archive_before_cancel', jobs=records, original_caches_retained=True)
(out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
subprocess.run(['scancel', *[str(r['job_id']) for r in jobs]], check=True)
remaining = subprocess.check_output(['squeue','-u','user_yuanjiao','-h','-o','%i %T %j'], text=True)
assert any(line.startswith('96510 ') for line in remaining.splitlines())
assert not any(line.split()[0] in {str(r['job_id']) for r in jobs}
               and 'COMPLETING' not in line for line in remaining.splitlines())
(out/'queue-after.txt').write_text(remaining)
print(json.dumps(dict(archive=str(out), cancelled=[r['job_id'] for r in jobs],
                     retained_job=96510, files=sum(len(r['archived']) for r in records)), indent=2))
