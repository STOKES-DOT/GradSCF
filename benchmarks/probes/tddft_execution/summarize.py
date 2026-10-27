"""Rebuild machine-readable summaries without editing original run records."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, records, compiles, workload = [], [], [], []
    for path in sorted(args.evidence.glob('*/*/measurements.jsonl')):
        run = str(path.parent.relative_to(args.evidence))
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        process_path = path.with_name('process.json')
        process = json.loads(process_path.read_text()) if process_path.exists() else {}
        groups = defaultdict(list)
        for row in rows:
            row = dict(row, run_id=run, evidence_path=str(path.relative_to(args.evidence)),
                       process_status=process.get('process_status', 'incomplete'),
                       scientific_pass=process.get('scientific_pass'),
                       accepted_for_comparison=process.get('accepted_for_comparison', False),
                       residual_per_root=process.get('residual_per_root'),
                       sampled_clean_rss_peak_bytes=process.get('sampled_clean_rss_peak_bytes'))
            records.append(row)
            if row['wall_seconds'] is not None:
                groups[row['stage']].append(row)
        for stage, samples in groups.items():
            warm = [x['wall_seconds'] for x in samples if x.get('repeat', 0)>0]
            first = samples[0]['wall_seconds']
            summaries.append(dict(run_id=run, case_id=samples[0]['case_id'], variant=samples[0]['variant'],
                device=samples[0]['device'], stage=stage, first_seconds=first,
                warm_samples_seconds=json.dumps(warm), warm_count=len(warm),
                warm_median_seconds=statistics.median(warm) if warm else None,
                warm_min_seconds=min(warm) if warm else None, warm_max_seconds=max(warm) if warm else None,
                sampled_clean_rss_peak_bytes=process.get('sampled_clean_rss_peak_bytes'),
                process_status=process.get('process_status', 'incomplete'),
                scientific_pass=process.get('scientific_pass'),
                accepted_for_comparison=process.get('accepted_for_comparison', False)))
        for name, target in [('compile_events.jsonl', compiles), ('solver_events.jsonl', workload)]:
            source = path.with_name(name)
            if source.exists():
                target.extend(dict(json.loads(s), run_id=run) for s in source.read_text().splitlines())
    for name, rows in [('measurements', records), ('compile_events', compiles), ('solver_events', workload)]:
        (args.output/(name+'.jsonl')).write_text(''.join(json.dumps(x, allow_nan=False)+'\n' for x in rows))
    if summaries:
        with (args.output/'summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(summaries)


if __name__ == '__main__':
    main()
