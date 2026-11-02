"""Submit only the two approved A100 profiles, without altering other jobs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/user_yuanjiao/private/homefile/nnao-dzpm-hidden-l50-20261030')


def main():
    destination = ROOT/'submitted.json'
    if destination.exists():
        raise RuntimeError('Submission record exists; inspect it before resubmitting.')
    queue = subprocess.run(['squeue', '-u', 'user_yuanjiao', '-h', '-o', '%i|%j|%T|%C'],
        check=True, capture_output=True, text=True).stdout
    rows = [line.split('|') for line in queue.splitlines()]
    if any('nnao50-hidden-' in row[1] for row in rows):
        raise RuntimeError('A matching training job already exists.')
    if {row[0] for row in rows} != {'112916', '112917'}:
        raise RuntimeError('Queue changed; review CPU headroom/dependencies before submitting.')
    record = dict(submitted_at=datetime.now(timezone.utc).isoformat(),
        preceding_jobs=['112916', '112917'],
        dependency_reason='Existing 64-CPU task chain finishes before the two 32-CPU training jobs.',
        jobs={})
    for profile in ('L1', 'L2'):
        result = subprocess.run(['sbatch', '--parsable', '--dependency=afterany:112916:112917',
            '--chdir='+str(ROOT), str(ROOT/(profile+'.slurm'))], cwd=str(ROOT),
            env=dict(os.environ, PWD=str(ROOT)), check=True, capture_output=True, text=True)
        job = result.stdout.strip().splitlines()[-1].split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Unrecognized sbatch response: '+result.stdout)
        record['jobs'][profile] = dict(job_id=job, sbatch_stdout=result.stdout,
            sbatch_stderr=result.stderr)
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(destination)
        print(profile, job, flush=True)
        time.sleep(3)


if __name__ == '__main__':
    main()
