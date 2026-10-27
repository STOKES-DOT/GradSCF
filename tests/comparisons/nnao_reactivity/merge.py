"""Create a separately versioned benchmark view from explicit native retries."""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnao_reactivity.protocols import reaction_profile


def merge_records(parent, retries):
    original = {r['key']: r for r in parent}
    overrides = {}
    for row in retries:
        key = row['key']
        if key not in original:
            raise ValueError('Unknown retry case: '+key)
        if key in overrides:
            raise ValueError('Duplicate retry case: '+key)
        old = original[key]
        if row['status'] == old['status'] == 'passed' and 'energy_hartree' in row and 'energy_hartree' in old:
            if row['energy_hartree'] > old['energy_hartree']+5e-8:
                raise ValueError('Retry energy is higher than the accepted parent energy.')
        overrides[key] = row
    return [deepcopy(overrides.get(row['key'], row)) for row in parent]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--retry', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads((args.parent/'results.json').read_text())
    retry = json.loads((args.retry/'results.json').read_text())
    retry_config = json.loads((args.retry/'configuration.json').read_text())
    config = json.loads((args.parent/'configuration.json').read_text())
    data = json.loads(args.dataset.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(args.parent/'results.json') != retry_config['parent_sha256']:
        raise ValueError('Parent report changed after retry selection.')
    if len(retry['results']) != retry['requested_cases']:
        raise ValueError('Retry did not finish every requested case.')
    if digest(args.dataset) != retry_config['dataset_sha256']:
        raise ValueError('Dataset differs from the native retry.')
    report = {**parent, 'results': merge_records(parent['results'], retry['results'])}
    report['isolated_results'] = {**parent['isolated_results'], **retry['isolated_results']}
    report['passed'] = sum(r['status'] == 'passed' for r in report['results'])
    report['failed'] = len(report['results'])-report['passed']
    report['status'] = 'complete' if not report['failed'] else 'complete_with_failures'
    report['repair'] = dict(parent_sha256=digest(args.parent/'results.json'),
        retry_sha256=digest(args.retry/'results.json'), retry_cases=len(retry['results']),
        retried_case_keys=[r['key'] for r in retry['results']],
        elapsed_seconds=retry['elapsed_seconds'], job_id=retry['job_id'],
        untouched_cases=len(report['results'])-len(retry['results']),
        scope='Real internal HF stability on retried cases only; global HF minima are not certified')
    good = {(r['identifier'], r['basis']): r for r in report['results']
            if r['task'] == 'species' and r['status'] == 'passed'}
    report['reactions'] = []
    for definition in data['reactions']:
        for basis in config['bases']:
            missing = [name for name in definition['species'] if (name, basis) not in good]
            row = dict(identifier=definition['identifier'], category=definition['category'], basis=basis)
            if missing:
                row.update(status='failed', missing_species=missing)
            else:
                energies = {name: good[name, basis]['energy_hartree'] for name in definition['species']}
                row.update(reaction_profile(definition['reactants'], definition['transition'], definition['products'], energies),
                    status='passed', reference_forward_kcal_mol=definition['reference_forward_kcal_mol'],
                    reference_reverse_kcal_mol=definition['reference_reverse_kcal_mol'],
                    reference_reaction_kcal_mol=definition['reference_reaction_kcal_mol'])
            report['reactions'].append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    config['native_retry'] = retry_config
    (args.output/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], passed=report['passed'], failed=report['failed'],
                         repaired_cases=len(retry['results']))))


if __name__ == '__main__':
    main()
