"""Summarize saved shared-execution evidence; Python standard library only."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    raw = args.report / 'raw'
    summaries, parity = [], []
    for name in ('water', 'benzene', 'benzene-ri', 'benzene-adjacent'):
        paths = [raw / f'{variant}-{name}' for variant in ('parent', 'candidate')]
        if not all((path / 'status.json').exists() for path in paths):
            continue
        data = [records(path / 'measurements.jsonl') for path in paths]
        statuses = [json.loads((path / 'status.json').read_text()) for path in paths]
        runtimes = [json.loads((path / 'runtime.json').read_text()) for path in paths]
        matched = all(runtimes[0][key] == runtimes[1][key]
                      for key in ('input_sha256', 'native_sha256', 'probe_sha256',
                                  'affinity', 'thread_environment', 'versions', 'devices'))
        ends = [[row for row in rows if row['stage'] == 'end_to_end'] for rows in data]
        certificates = [[row for row in rows if row['stage'] == 'certification'] for rows in data]
        if not all(ends) or len(ends[0]) != len(ends[1]):
            raise ValueError(f'{name}: incomplete clean repetitions')
        errors = [max(abs(a-b) for a,b in zip(left['excitations_ha'], right['excitations_ha']))
                  for left,right in zip(*ends)]
        accepted = (matched and all(row.get('status') == 'ok' and row.get('scientific_pass') is True for row in statuses) and max(errors) < 1e-9 and all(row['public_converged'] and row['finite_outputs']
                    for group in ends for row in group) and
                    all(group and group[-1]['scientific_pass'] for group in certificates))
        parity.append(dict(case=name, accepted=accepted, matched_execution_settings=matched, excitation_error_ha=max(errors),
                           certification_scope='final clean repeat only',
                           residuals=[group[-1]['residual_per_root'] if group else None for group in certificates]))
        for stage in ('scf', 'response_reference', 'response_solve', 'oscillator_strength', 'end_to_end'):
            samples = [[row['wall_seconds'] for row in rows if row['stage'] == stage] for rows in data]
            for state in ('first_call', 'warm_new_objects'):
                groups = [values[:1] if state == 'first_call' else values[1:] for values in samples]
                if not all(groups):
                    continue
                medians = [statistics.median(values) for values in groups]
                summaries.append(dict(case=name, stage=stage, state=state, accepted_final_state=accepted,
                    parent_samples=json.dumps(groups[0]), candidate_samples=json.dumps(groups[1]),
                    parent_median_s=medians[0], candidate_median_s=medians[1],
                    time_ratio_candidate_parent=medians[1]/medians[0] if accepted else None))
    if summaries:
        with (args.report/'timings.csv').open('w') as stream:
            writer=csv.DictWriter(stream, fieldnames=summaries[0].keys()); writer.writeheader(); writer.writerows(summaries)
    (args.report/'response_parity.json').write_text(json.dumps(parity, indent=2)+'\n')
    compile_path=raw/'candidate-compile'/'compile_events.jsonl'
    if compile_path.exists():
        events=records(compile_path)
        counts={str(repeat):dict(count=sum(row['repeat']==repeat for row in events),
                                seconds=sum(row.get('wall_seconds',0) for row in events if row['repeat']==repeat))
                for repeat in (0,1)}
        (args.report/'compile_counts.json').write_text(json.dumps(dict(
            case='benzene public full TDDFT; fresh SCF facade each repeat', counts=counts,
            scope='instrumented response.kernel only; not clean timing', events='raw/candidate-compile/compile_events.jsonl'),indent=2)+'\n')
    memory=[]
    for path in sorted(raw.glob('*.process.json')):
        row=json.loads(path.read_text())
        memory.append(dict(run=path.name.removesuffix('.process.json'),
            process_status=row['status'], returncode=row['returncode'],
            wall_seconds=row['wall_seconds'],
            sampled_group_rss_peak_bytes=row['sampled_group_rss_peak_bytes'],
            child_ru_maxrss_bytes=row['child_ru_maxrss_bytes'],
            sample_seconds=row['sample_seconds']))
    if memory:
        with (args.report/'process_memory.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=memory[0].keys());writer.writeheader();writer.writerows(memory)
    suites=[]
    for path in sorted(raw.glob('*.xml')):
        root=ET.parse(path).getroot()
        suites.append(dict(file=str(path.relative_to(args.report)),
            cases=[dict(name=f"{node.get('classname')}::{node.get('name')}",
                        status='failed' if node.find('failure') is not None or node.find('error') is not None
                               else 'skipped' if node.find('skipped') is not None else 'passed')
                   for node in root.iter('testcase')]))
    (args.report/'test_receipts.json').write_text(json.dumps(suites,indent=2)+'\n')
    # RED receipts remain above, but never enter the final acceptance count.
    selected = ('green.xml', 'policy-green.xml', 'integration.xml', 'ad-families.xml',
                'force-training.xml', 'gpu-ad.xml', 'isolated-test_streamed_response_grid.xml',
                'isolated-test_workflow_reuse_integration.xml', 'isolated-bse.xml',
                'candidate-direct-reuse.xml', 'direct-coordinate-ad.xml', 'isolated-reuse.xml')
    latest = {}
    for name in selected:
        match = next((suite for suite in suites if Path(suite['file']).name == name), None)
        if match is None:
            continue
        for case in match['cases']:
            latest[case['name']] = dict(case, receipt=match['file'])
    acceptance = dict(
        note='Unique test identities after explicit supersession; CPU/GPU duplicates count once. '
             'Known parent failures remain failures. The compiler crash remains separate.',
        counts={status: sum(case['status'] == status for case in latest.values())
                for status in ('passed', 'failed', 'skipped')},
        tests=list(latest.values()))
    (args.report/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    print(json.dumps(dict(pairs=parity, test_suites=len(suites),
                         final_unique_tests=acceptance['counts']),indent=2))


if __name__=='__main__':
    main()
