"""Check identical saved inputs, full outputs and AD gates before timing ratios."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results', type=Path)
parser.add_argument('allocation')
parser.add_argument('output', type=Path)
args = parser.parse_args()
comparisons = []
for before_path in sorted(args.results.glob(f'spin-*-candidate-grid-{args.allocation}.jsonl')):
    after_path = before_path.with_name(before_path.name.replace('candidate-grid', 'candidate-quota-v1'))
    rows = [[json.loads(line) for line in path.read_text().splitlines()]
            for path in (before_path, after_path)]
    for key in ('kernel', 'nao', 'input_sha256', 'affinity', 'device', 'dtype'):
        assert rows[0][0][key] == rows[1][0][key], key
    complete = all(any(row['stage'] == 'process_peak' for row in run) for run in rows)
    gates = all(row.get('scientific_pass') is not False for run in rows for row in run)
    with np.load(before_path.with_suffix('.npz')) as a, np.load(after_path.with_suffix('.npz')) as b:
        assert a.files == b.files
        error = max(float(np.max(np.abs(a[key]-b[key]))) for key in a.files)
        equivalent = all(np.allclose(a[key], b[key], atol=3e-10, rtol=2e-11) for key in a.files)
    samples = [[row['wall_seconds'] for row in run if row['stage'] == 'action' and row['repeat'] > 0]
               for run in rows]
    gradients = [{row['channel']: row['gradient'] for row in run if row['stage'] == 'value_and_grad'}
                 for run in rows]
    assert gradients[0].keys() == gradients[1].keys()
    derivative_errors = {key: abs(value-gradients[1][key]) for key, value in gradients[0].items()}
    derivative_pass = all(np.isclose(value, gradients[1][key], atol=3e-9, rtol=2e-9)
                          for key, value in gradients[0].items())
    passed = bool(complete and gates and equivalent and derivative_pass and all(len(x) >= 10 for x in samples))
    medians = [float(np.median(x)) for x in samples]
    metadata = rows[0][0]
    comparisons.append(dict(kernel=metadata['kernel'], nao=metadata['nao'],
        device=metadata['device'], input_sha256=metadata['input_sha256'],
        before=str(before_path), after=str(after_path), scientific_pass=passed,
        array_parameter_gradient_errors=derivative_errors,
        full_output_maxabs_error=error, before_warm_seconds=samples[0], after_warm_seconds=samples[1],
        before_median_seconds=medians[0], after_median_seconds=medians[1],
        time_ratio_after_before=medians[1]/medians[0] if passed else None,
        compile_seconds=[next(r['wall_seconds'] for r in run if r['stage'] == 'compile') for run in rows],
        xla_temp_bytes=[next(r['xla_temp_bytes'] for r in run if r['stage'] == 'compile') for run in rows],
        cpu_peak_rss_bytes=[next(r['cpu_peak_rss_bytes'] for r in run if r['stage'] == 'process_peak') for run in rows]))
assert len(comparisons) == 4, 'Incomplete planned set; do not publish a partial aggregate.'
args.output.write_text(json.dumps(comparisons, indent=2)+'\n')
print(json.dumps(comparisons, indent=2))
raise SystemExit(0 if all(row['scientific_pass'] for row in comparisons) else 1)
