"""Summarize synced direct-J/K evidence; stdlib only, no scientific execution."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

MATCH = ('workers', 'affinity', 'config_sha256', 'density_sha256', 'private_output_bytes',
         'cutoff', 'omega', 'dtype', 'nao', 'host', 'jax')
VARIANTS = ('basis_screened', 'density_screened')
KINDS = ('value', 'geometry', 'coefficients', 'log_exponents')


def read(path):
    if not path.exists(): return None
    try: return json.loads(path.read_text())
    except (ValueError, OSError) as error: return {'read_error': str(error)}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def safe(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    return value


def receipt_ok(receipt):
    return isinstance(receipt, dict) and receipt.get('status') == 'ok' and receipt.get('returncode') == 0


def stats(record, runtime):
    record, runtime = record or {}, runtime or {}
    samples = record.get('warm_seconds')
    valid = isinstance(samples, list) and len(samples) >= 3 and all(finite(x) and x > 0 for x in samples)
    return dict(warm_seconds=samples, warm_count=len(samples) if isinstance(samples, list) else None,
        warm_median=statistics.median(samples) if valid else None, warm_min=min(samples) if valid else None,
        warm_max=max(samples) if valid else None, first_compiled_seconds=record.get('first_compiled_seconds'),
        lower_seconds=record.get('lower_seconds'), compile_seconds=record.get('compile_seconds'),
        preparation_seconds=runtime.get('preparation_seconds'), ru_maxrss_bytes=record.get('ru_maxrss_bytes'))


def run(raw, name):
    runtime, summary, receipt = (read(raw/name/'runtime.json'), read(raw/name/'summary.json'), read(raw/(name+'.process.json')))
    records = summary.get('records', []) if isinstance(summary, dict) else []
    state = 'pending' if any(x is None for x in (runtime, summary, receipt)) else 'failed'
    if receipt_ok(receipt) and isinstance(runtime, dict) and 'read_error' not in runtime and summary and summary.get('complete') is True:
        if len(records) == 2 and {r.get('variant') for r in records} == set(VARIANTS): state = 'complete'
    if receipt is not None and not receipt_ok(receipt): state = 'failed'
    return dict(source=name, state=state, runtime=runtime, receipt=receipt, records={r['variant']: r for r in records if 'variant' in r})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rawdir', type=Path); parser.add_argument('outdir', type=Path)
    args = parser.parse_args(); raw, out = args.rawdir, args.outdir
    amendment_path = raw.parent/'sampling_amendment.md'
    amendment = amendment_path.read_text() if amendment_path.exists() else None
    measurements, pairs = [], []
    specifications = [('final', 'forward-final-before', 'forward-final-after'),
                      ('exploratory64', 'forward-before', 'forward-screen-after'),
                      ('exploratory64', 'forward-repeat-before', 'forward-repeat-after')]
    for category, before_name, after_name in specifications:
        before, after = run(raw, before_name), run(raw, after_name)
        for variant in VARIANTS:
            b, a = before['records'].get(variant), after['records'].get(variant)
            bs, ass = stats(b, before['runtime']), stats(a, after['runtime'])
            reasons = [side+':'+data['state'] for side, data in (('before', before), ('after', after)) if data['state'] != 'complete']
            br, ar = before['runtime'] or {}, after['runtime'] or {}
            reasons += ['metadata:'+key for key in MATCH if br.get(key) is None or ar.get(key) is None or br[key] != ar[key]]
            if b is None or a is None: reasons.append('variant_missing_or_no_overlap')
            if bs['warm_median'] is None or ass['warm_median'] is None: reasons.append('insufficient_or_invalid_warm_samples')
            error = (a or {}).get('maxabs_reference_error')
            if not finite(error) or error < 0: reasons.append('missing_or_invalid_reference_error')
            for side, data in (('before', bs), ('after', ass)):
                if any(not finite(data[k]) or data[k] < 0 for k in ('first_compiled_seconds', 'lower_seconds', 'compile_seconds', 'preparation_seconds')):
                    reasons.append(side+':invalid_separate_timing')
            if category == 'final' and bs['warm_count'] != ass['warm_count'] and not amendment: reasons.append('sampling_amendment_missing')
            accepted = not reasons
            state = 'accepted' if accepted else ('failed' if 'failed' in (before['state'], after['state']) else 'pending' if 'pending' in (before['state'], after['state']) else 'rejected')
            row = dict(type='forward_pair', category=category, pair=before_name+' -> '+after_name, variant=variant,
                state=state, accepted=accepted, reasons=reasons, ratio_after_before=ass['warm_median']/bs['warm_median'] if accepted else None,
                maxabs_reference_error=error, before=bs, after=ass, before_runtime=before['runtime'], after_runtime=after['runtime'],
                before_receipt=before['receipt'], after_receipt=after['receipt'], sampling_amendment=amendment if category == 'final' else None)
            measurements.append(row); pairs.append(row)
    product_path = raw/'products.jsonl'
    product_read_error = False
    try: product_data = [json.loads(line) for line in product_path.read_text().splitlines() if line.strip()] if product_path.exists() else []
    except (ValueError, OSError): product_data, product_read_error = [], True
    runtime = next((r for r in product_data if r.get('type') == 'runtime'), None)
    product_summary = next((r for r in product_data if r.get('type') == 'summary'), None)
    product_receipt = read(raw/'products.process.json')
    products = [r for r in product_data if r.get('type') == 'measurement']
    complete = (len(products) == 8 and {(r.get('kind'), r.get('cutoff')) for r in products} == {(k, c) for k in KINDS for c in (0., 1e-12)})
    unique_metadata = sum(r.get('type') == 'runtime' for r in product_data) == sum(r.get('type') == 'summary' for r in product_data) == 1
    products_ok = receipt_ok(product_receipt) and complete and unique_metadata and bool(product_summary and product_summary.get('complete') is True and product_summary.get('consistency_passed') is True)
    for row in products:
        errors = row.get('max_abs_error_vs_zero_cutoff', [])
        gates = (runtime or {}).get('gates', {})
        limits = [gates.get('value_absolute')] + ([] if row['kind'] == 'value' else [gates.get('directional_derivative_absolute')])
        accepted = products_ok and row.get('gate_passed') is True and stats(row, runtime)['warm_median'] is not None
        accepted = accepted and all(finite(row.get(k)) and row[k] >= 0 for k in ('first_compiled_seconds', 'lower_seconds', 'compile_seconds'))
        accepted = accepted and len(errors) == len(limits) and all(finite(e) and finite(g) and 0 <= e <= g for e, g in zip(errors, limits))
        accepted = bool(accepted and finite(row.get('value')) and (row['kind'] == 'value' or finite(row.get('directional_derivative'))))
        measurements.append(dict(row, type='products_measurement', accepted=accepted, runtime=runtime, receipt=product_receipt,
                                 sample_statistics=stats(row, runtime), comparison='cutoff comparison only; not before/after or SCF'))
    product_rows = [r for r in measurements if r['type'] == 'products_measurement']
    product_state = 'accepted' if products_ok and all(r['accepted'] for r in product_rows) else 'pending' if product_receipt is None or product_summary is None else 'failed'
    if product_read_error or (product_receipt is not None and not receipt_ok(product_receipt)): product_state = 'failed'
    groups = {}
    for path in sorted(raw.glob('*.xml')):
        try:
            cases = list(ET.parse(path).getroot().iter('testcase'))
            counts = {key: sum(case.find(tag) is not None for case in cases) for key, tag in (('failed', 'failure'), ('errors', 'error'), ('skipped', 'skipped'))}
            groups[path.stem] = dict(tests=len(cases), passed=len(cases)-sum(counts.values()), **counts, receipt=read(path.with_suffix('.process.json')))
        except (ET.ParseError, OSError) as error: groups[path.stem] = dict(tests=None, state='unreadable', reason=str(error))
    out.mkdir(parents=True, exist_ok=True)
    (out/'measurements.jsonl').write_text(''.join(json.dumps(safe(row), allow_nan=False)+'\n' for row in measurements))
    fields = ['category', 'pair', 'variant', 'state', 'accepted', 'ratio_after_before', 'maxabs_reference_error']
    columns = list(stats(None, None))
    fields += [side+'_'+key for side in ('before', 'after') for key in columns]+['reasons']
    with (out/'before_after.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        for row in pairs:
            flat = {key: row[key] for key in fields if key in row}
            flat.update({side+'_'+key: json.dumps(safe(value)) if isinstance(value, list) else value for side in ('before', 'after') for key, value in row[side].items()})
            flat['reasons'] = '; '.join(row['reasons']); writer.writerow(safe(flat))
    final = [r for r in pairs if r['category'] == 'final']
    summary = dict(rawdir=str(raw.resolve()), final_forward_accepted=all(r['accepted'] for r in final),
        forward_pairs=[{k: r[k] for k in ('category', 'pair', 'variant', 'state', 'accepted', 'ratio_after_before', 'reasons')} for r in pairs],
        products=dict(state=product_state, measurements=len(products), scope='cutoff comparison; no SCF'),
        test_groups=groups, test_group_note='Counts overlap; do not sum groups.', sampling_amendment=amendment,
        acceptance_note='Forward accepted means complete matched comparison, not a performance-target verdict or overall project acceptance.',
        timing_note='Ratios use synchronized warm medians only; first/lower/compile/preparation/RSS remain separate. No whole-process ratio.')
    (out/'summary.json').write_text(json.dumps(safe(summary), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
