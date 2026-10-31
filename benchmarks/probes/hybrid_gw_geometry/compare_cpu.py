"""Analyze saved worker and coordinate-product probes; no new integrals."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results', type=Path)
parser.add_argument('allocation')
parser.add_argument('output', type=Path)
args = parser.parse_args()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def warm(data):
    samples = [r['wall_seconds'] for r in data if r['repeat'] > 0]
    assert len(samples) >= 3
    return dict(samples_seconds=samples, median_seconds=float(np.median(samples)),
        average_busy_logical_cpus=float(np.median([
            (r['cpu_user_seconds']+r['cpu_system_seconds'])/r['wall_seconds']
            for r in data if r['repeat'] > 0])),
        process_peak_rss_bytes=max(r['process_peak_rss_bytes'] for r in data))


base = args.results/f'phi-svp-worker-budget-{args.allocation}.jsonl'
worker_rows = rows(base)
worker_results = []
reference = base.with_name(base.stem+'-w8-o0.0.npz')
for workers in (8, 32, 64):
    group = [r for r in worker_rows if r['workers'] == workers]
    with np.load(reference) as a, np.load(base.with_name(base.stem+f'-w{workers}-o0.0.npz')) as b:
        error = max(float(np.max(np.abs(a[k]-b[k]))) for k in a.files)
    worker_results.append(dict(workers=workers, **warm(group),
        full_output_maxabs_error=error, scientific_pass=error <= 3e-10,
        logical_affinity_count=len(group[0]['affinity']),
        physical_core_count=group[0]['physical_core_count'],
        planned_native_workspace_bytes=group[0]['planned_native_workspace_bytes']))

paths = [args.results/f'phi-svp-integral-vjp-{v}-{args.allocation}.jsonl'
         for v in ('candidate-grid', 'candidate-quota-v1')]
groups = [rows(p) for p in paths]
for key in ('affinity', 'density_sha256', 'case_sha256', 'workers', 'cutoff', 'omega', 'exchange_fraction'):
    assert groups[0][0][key] == groups[1][0][key], key
with np.load(paths[0].with_name(paths[0].stem+'-w32-o0.0.npz')) as a, \
     np.load(paths[1].with_name(paths[1].stem+'-w32-o0.0.npz')) as b:
    error = float(np.max(np.abs(a['gradient']-b['gradient'])))
    passed = bool(np.allclose(a['gradient'], b['gradient'], atol=3e-8, rtol=3e-9))
before, after = [warm(group) for group in groups]
result = dict(scope='fixed-density integral products; not full molecular forces',
    workers=worker_results, coordinate_vjp=dict(before=before, after=after,
        full_gradient_maxabs_error=error, scientific_pass=passed,
        time_ratio_after_before=after['median_seconds']/before['median_seconds'] if passed else None,
        native_sha256=[group[0]['native_sha256'] for group in groups]))
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
raise SystemExit(0 if passed and all(r['scientific_pass'] for r in worker_results) else 1)
