"""Aggregate local XC receipts with stdlib only; preserve all raw evidence.

Ratios are after/before and require both sides accepted. Kernel observations
are separate from complete SCF; process RSS sampling and ru_maxrss are distinct.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rawdir', type=Path)
    parser.add_argument('outdir', type=Path)
    args = parser.parse_args()
    raw, out = args.rawdir.resolve(), args.outdir.resolve()
    if out == raw or raw in out.parents:
        parser.error('Output must be outside the immutable raw directory')
    out.mkdir(parents=True, exist_ok=True)
    warnings, measurements, comparisons = [], [], []

    def read(path, jsonl=False):
        if not path.exists():
            return [] if jsonl else {}
        if not jsonl:
            try:
                return json.loads(path.read_text())
            except (ValueError, OSError) as error:
                warnings.append(dict(path=str(path), error=str(error)))
                return {}
        rows = []
        for index, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    warnings.append(dict(path=str(path), error=f'incomplete JSONL line {index}'))
        return rows

    def stats(values):
        values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
        return dict(samples=values, median=statistics.median(values) if values else None,
                    min=min(values) if values else None, max=max(values) if values else None)

    def status(receipt, complete, failed=False):
        if failed or (receipt and (receipt.get('returncode') != 0 or receipt.get('status') != 'ok')):
            return 'failed'
        return 'accepted' if complete and receipt.get('returncode') == 0 and receipt.get('status') == 'ok' else 'pending'

    def record(scope, device, variant, folder, receipt, state, metrics, **details):
        row = dict(scope=scope, device=device, variant=variant, source_directory=str(folder) if folder.exists() else None,
            status=state, metrics={key: stats(value) for key, value in metrics.items()},
            process_receipt=receipt or None, process_sampled_rss_peak_bytes=receipt.get('sampled_group_rss_peak_bytes'),
            process_child_ru_maxrss_bytes=receipt.get('child_ru_maxrss_bytes'),
            memory_scope='Process peaks are cumulative; kernel variants in one process do not have isolated RSS peaks', **details)
        measurements.append(row)
        return row

    def compare(scope, device, before, after):
        state = 'failed' if 'failed' in (before['status'], after['status']) else (
            'accepted' if before['status'] == after['status'] == 'accepted' else 'pending')
        for metric in sorted(set(before['metrics']) | set(after['metrics'])):
            left, right = before['metrics'].get(metric, stats([])), after['metrics'].get(metric, stats([]))
            ratio = right['median']/left['median'] if state == 'accepted' and left['median'] not in (None, 0) and right['median'] is not None else None
            comparisons.append(dict(scope=scope, device=device, before=before['variant'], after=after['variant'],
                metric=metric, status=state, before_samples=left['samples'], after_samples=right['samples'],
                before_median=left['median'], before_min=left['min'], before_max=left['max'],
                after_median=right['median'], after_min=right['min'], after_max=right['max'], ratio_after_before=ratio))

    for device in ('cpu', 'gpu'):
        folder = raw/f'final-local-{device}'
        summary, receipt = read(folder/'summary.json'), read(folder.with_suffix('.process.json'))
        selected = ('admitted_cpu_cache' if device == 'cpu' else 'raw_stream_4096', 'local_4096_cap512_tol1e-14')
        entries = {row['variant']: row for row in summary.get('measurements', [])}
        normalized = {}
        for name in dict.fromkeys((*entries, *selected)):
            row = entries.get(name, {})
            complete = summary.get('complete') is True and summary.get('fixed_density_accuracy_pass') is True and row.get('accuracy_pass') is True and len(row.get('warm_seconds', [])) == 3
            normalized[name] = record('fixed_density_pbe0_xc', device, name, folder, receipt,
                status(receipt, complete, row.get('accuracy_pass') is False),
                dict(warm_operator_seconds=row.get('warm_seconds', []), first_compiled_operator_seconds=[row.get('first_seconds')],
                     preparation_seconds=[row.get('preparation_seconds')], lower_seconds=[row.get('lower_seconds')],
                     compile_seconds=[row.get('compile_seconds')]),
                energy_error_hartree=row.get('energy_error_hartree'), potential_maxabs_error_hartree=row.get('potential_maxabs_error_hartree'))
        compare('fixed_density_pbe0_xc', device, *(normalized[name] for name in selected))

        pair = []
        for side in ('before', 'after'):
            folder = raw/'scf-paired-118316'/f'{device}-{side}'
            summary, receipt = read(folder/'summary.json'), read(folder.with_suffix('.process.json'))
            events = read(folder/'measurements.jsonl', True)
            rows = summary.get('records') or [event for event in events if event.get('stage') == 'scf_result']
            complete = (len(rows) == 4 and {r.get('repeat') for r in rows} == {0, 1, 2, 3}
                and summary.get('scientific_pass') is True and summary.get('smoke') is False
                and all(r.get('scientific_pass') is True for r in rows))
            metrics = {}
            for key in ('total_seconds', 'cpu_prepare_seconds', 'stage_seconds', 'solver_seconds'):
                metrics['cold_'+key] = [r.get(key) for r in rows if r.get('repeat') == 0]
                metrics['warm_'+key] = [r.get(key) for r in rows if r.get('repeat') in (1, 2, 3)]
            metrics['process_sampled_rss_peak_bytes'] = [receipt.get('sampled_group_rss_peak_bytes')]
            metrics['process_child_ru_maxrss_bytes'] = [receipt.get('child_ru_maxrss_bytes')]
            metrics['reported_cumulative_ru_maxrss_bytes'] = [r.get('peak_rss_bytes') for r in rows]
            pair.append(record('complete_pbe0_scf', device, side, folder, receipt,
                status(receipt, complete, any(r.get('scientific_pass') is False for r in rows)), metrics,
                records_observed=len(rows) if rows or summary or events else None,
                energy_hartree=[r.get('energy_hartree') for r in rows],
                orbital_residual_hartree=[r.get('orbital_residual_hartree') for r in rows]))
        compare('complete_pbe0_scf', device, *pair)

    jobs = sorted(path for path in raw.glob('whole-mgga-*') if path.is_dir()) or [raw/'whole-mgga-pending']
    for job in jobs:
        for device in ('cpu', 'gpu'):
            folder = job/device
            summary, receipt = read(folder/'summary.json'), read(folder.with_suffix('.process.json'))
            events = read(folder/'measurements.jsonl', True)
            pair = []
            for hinted in (False, True):
                row = next((r for r in summary.get('records', []) if r.get('hinted') is hinted), {})
                complete = summary.get('accuracy_pass') is True and len(summary.get('records', [])) == 2 and len(row.get('warm_seconds', [])) == 3
                pair.append(record('fixed_pbe0_density_r2scan_xc:'+job.name, device, 'with_hints' if hinted else 'without_hints', folder, receipt,
                    status(receipt, complete, summary.get('accuracy_pass') is False),
                    dict(warm_operator_seconds=row.get('warm_seconds', []), first_compiled_operator_seconds=[row.get('first_seconds')],
                         lower_seconds=[row.get('lower_seconds')], compile_seconds=[row.get('compile_seconds')],
                         shared_provider_preparation_seconds=[e.get('seconds') for e in events if e.get('stage') == 'provider_preparation']),
                    exact_guard_matches=summary.get('exact_guard_matches'),
                    energy_error_hartree=summary.get('energy_error_hartree'), potential_maxabs_error_hartree=summary.get('potential_maxabs_error_hartree')))
            compare('fixed_pbe0_density_r2scan_xc:'+job.name, device, *pair)

    (out/'measurements.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in measurements))
    with (out/'before_after.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]), lineterminator="\n")
        writer.writeheader()
        for row in comparisons:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    summary = dict(raw_directory=str(raw), ratio_definition='after/before; smaller is faster; null until both sides accepted',
        acceptance='Complete SCF requires exactly repeats 0..3, each scientific_pass, summary pass, and process returncode=0/status=ok',
        scopes_separate=True,
        timing_notes={'first_compiled_operator_seconds': 'First execution after explicit lowering/compilation; not process cold start',
                      'kernel_preparation_seconds': 'Additional variant preparation; shared basis/grid preparation is in raw events',
                      'cold_total_seconds': 'First complete SCF call after imports/native registration, including its preparation and solver compilation'},
        measurements=measurements, comparisons=comparisons, parse_warnings=warnings)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(measurements=len(measurements), comparisons=len(comparisons), parse_warnings=len(warnings), output=str(out))))


if __name__ == '__main__':
    main()
