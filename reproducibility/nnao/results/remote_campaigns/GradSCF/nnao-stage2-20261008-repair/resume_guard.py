"""Allow continuation only after this training job's confirmed time limit."""
import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np


def checkpoint_metadata(path):
    with np.load(path, allow_pickle=False) as saved:
        return json.loads(str(saved['metadata'].item()))


def resume_decision(run_dir, parent_log, parent_job, target_epochs):
    run_dir = Path(run_dir)
    last = run_dir / 'last.npz'
    if not last.is_file():
        return dict(action='skip', reason='No accepted checkpoint to continue.')
    checkpoint = checkpoint_metadata(last)
    epoch = checkpoint['epoch']
    if type(epoch) is not int or epoch < 0:
        return dict(action='blocked', reason='Invalid checkpoint epoch.')
    if epoch >= target_epochs:
        return dict(action='skip', epoch=epoch, reason='Target epochs already completed.')
    summary = run_dir / 'summary.json'
    if summary.is_file() and json.loads(summary.read_text()).get('status') == 'failed':
        return dict(action='skip', epoch=epoch, reason='Prior numerical failure requires investigation.')
    log = Path(parent_log)
    lines = log.read_text(errors='replace').splitlines() if log.is_file() else []
    job_pattern = re.compile(r'\b(?:JOB|STEP)\s+' + re.escape(str(parent_job)) + r'(?:[.\s])')
    if not any(job_pattern.search(line) and 'DUE TO TIME LIMIT' in line.upper() for line in lines):
        return dict(action='skip', epoch=epoch, reason='No confirmed time-limit exit for the parent job.')
    for name in ('best.npz', 'best_validation.npz'):
        path = run_dir / name
        if not path.is_file():
            return dict(action='blocked', epoch=epoch, reason=f'Missing required checkpoint: {name}.')
        if checkpoint_metadata(path).get('identity') != checkpoint.get('identity'):
            return dict(action='blocked', epoch=epoch, reason=f'Checkpoint identity differs: {name}.')
    return dict(action='resume', epoch=epoch, reason='Accepted checkpoint after confirmed time limit.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--parent-log', type=Path, required=True)
    parser.add_argument('--parent-job', required=True)
    parser.add_argument('--target-epochs', type=int, default=1000)
    parser.add_argument('--attempt-dir', type=Path, required=True)
    args = parser.parse_args()
    args.attempt_dir.mkdir(parents=True, exist_ok=True)
    result = resume_decision(args.run_dir, args.parent_log, args.parent_job, args.target_epochs)
    if result['action'] == 'resume':
        archive = args.attempt_dir / 'previous-attempt'
        archive.mkdir(exist_ok=True)
        for name in ('metadata.json', 'status.txt', 'summary.json', 'progress.json',
                     'validation_history.json', 'batch_progress.json', 'heldout_progress.json'):
            path = args.run_dir / name
            if path.is_file():
                shutil.copy2(path, archive / name)
        shutil.copy2(args.parent_log, archive / args.parent_log.name)
    (args.attempt_dir / 'guard.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return {'resume': 0, 'skip': 10, 'blocked': 2}[result['action']]


if __name__ == '__main__':
    raise SystemExit(main())
