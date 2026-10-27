"""Join checkpointed retry phases without recomputing accepted cases."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnao_reactivity.merge import merge_records


def join_records(previous, continuation, expected_keys):
    accepted = {r['key'] for r in previous if r['status'] == 'passed'}
    if any(r['key'] in accepted for r in continuation):
        raise ValueError('Continuation must preserve already accepted cases.')
    known = {r['key']: r for r in previous}
    base = [known.get(k, dict(key=k, status='pending')) for k in expected_keys]
    combined = merge_records(base, continuation)
    if any(r['status'] == 'pending' for r in combined):
        raise ValueError('Continuation is missing requested cases.')
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--continuation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old = json.loads((args.previous/'results.json').read_text())
    new = json.loads((args.continuation/'results.json').read_text())
    old_cfg = json.loads((args.previous/'configuration.json').read_text())
    new_cfg = json.loads((args.continuation/'configuration.json').read_text())
    for name in ['parent_sha256', 'dataset_sha256', 'checkpoint_sha256']:
        if old_cfg[name] != new_cfg[name]:
            raise ValueError('Retry phase provenance differs: '+name)
    if len(new['results']) != new['requested_cases']:
        raise ValueError('Continuation did not complete.')
    rows = join_records(old['results'], new['results'], old_cfg['keys'])
    report = {**old, 'results': rows, 'job_id': new['job_id'],
        'previous_jobs': [old['job_id'], new['job_id']],
        'elapsed_seconds': old['elapsed_seconds']+new['elapsed_seconds'],
        'isolated_results': {**old['isolated_results'], **new['isolated_results']}}
    report['passed'] = sum(r['status'] == 'passed' for r in rows)
    report['failed'] = len(rows)-report['passed']
    report['status'] = 'complete' if not report['failed'] else 'complete_with_failures'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report['phase_reports'] = {str(p): digest(p/'results.json') for p in [args.previous, args.continuation]}
    args.output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = args.continuation if row['key'] in set(new_cfg['keys']) else args.previous
        suffix = '-state.npz' if row['task'] == 'species' else '-dimer-state.npz'
        name = row['key']+suffix
        if (source/name).is_file():
            shutil.copy2(source/name, args.output/name)
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'configuration.json').write_text(json.dumps({**old_cfg, 'continuation': new_cfg}, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], passed=report['passed'], failed=report['failed'],
                         retained_accepted_cases=sum(r['status']=='passed' for r in old['results']))))


if __name__ == '__main__':
    main()
