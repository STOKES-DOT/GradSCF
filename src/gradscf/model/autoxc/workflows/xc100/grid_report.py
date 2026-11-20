"""Validate saved grid-block measurements and render scientific scaling plots."""
import csv
import json
from pathlib import Path

import numpy as np

from .grid_benchmark import compare_arrays


def render_plot(rows, sweep, output):
    """Render saved scalar measurements; no molecular calculation is performed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    labels = {'010': r'BH$_3$', '017': r'CF$_4$', '065': r'C$_6$H$_6$'}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.6), constrained_layout=True)
    panels = [('warm_assembly_seconds', 'Warm AO assembly [s]\n(sum of stage medians)', True),
              ('compile_seconds', 'Cold tracing + compilation [s]\n(matrix and directions)', False),
              ('gpu_peak_gib', 'JAX peak live allocations [GiB]', True),
              ('rss_peak_gib', 'Process peak RSS [GiB]', False)]
    for identifier, color in zip(sweep['ids'], ('#0072B2','#D55E00','#009E73')):
        selected = [row for row in rows if row['system_id']==identifier and row['status']=='ok']
        for ax, (field, ylabel, logarithmic) in zip(axes.flat, panels):
            x, y = [r['block_size'] for r in selected], [r[field] for r in selected]
            if field=='warm_assembly_seconds':
                spread = [[r[field]-r['warm_assembly_min_seconds'] for r in selected],
                          [r['warm_assembly_max_seconds']-r[field] for r in selected]]
                ax.errorbar(x, y, yerr=spread, fmt='-o', capsize=3,
                            color=color, label=labels[identifier], linewidth=1.8, markersize=4)
            else:
                ax.plot(x, y, '-o', color=color, label=labels[identifier], linewidth=1.8, markersize=4)
            ax.set_xscale('log', base=2)
            ax.set_xticks(sweep['blocks'], [str(b) for b in sweep['blocks']], rotation=25)
            ax.set_xlim(min(sweep['blocks'])/1.18, max(sweep['blocks'])*1.18)
            if logarithmic: ax.set_yscale('log')
            ax.set_xlabel('Grid points per block'); ax.set_ylabel(ylabel)
            ax.grid(True, alpha=.22)
    axes[0,0].legend(frameon=False)
    figure.suptitle('XC100: grid-block scaling at fixed basis, quadrature and candidate', fontsize=13)
    figure.savefig(output/'grid_block_scaling.png', dpi=220)
    figure.savefig(output/'grid_block_scaling.pdf')
    plt.close(figure)


def build_report(root):
    from ...history import write_json

    root = Path(root)
    output = root/'results'
    sweep = json.loads((output/'sweep.json').read_text())
    labels = {'010': 'BH3', '017': 'CF4', '065': 'C6H6'}
    failures = {(r['system_id'], r['block_size']): r for r in sweep['failures']}
    rows, checks = [], {}
    for identifier in sweep['ids']:
        baseline = output/identifier/'block-256.npz'
        reference = dict(np.load(baseline, allow_pickle=False)) if baseline.exists() else None
        baseline_record = output/identifier/'block-256.json'
        identity = json.loads(baseline_record.read_text()) if baseline_record.exists() else None
        records = {}
        for block in sweep['blocks']:
            path = output/identifier/f'block-{block}.json'
            row = {'system_id': identifier, 'molecule': labels[identifier], 'block_size': block,
                   'status': 'failed', 'warm_assembly_seconds': None, 'compile_seconds': None,
                   'warm_assembly_min_seconds': None, 'warm_assembly_max_seconds': None,
                   'first_assembly_seconds': None, 'score_gradient_seconds': None,
                   'density_diagnostic_seconds': None, 'gpu_peak_gib': None, 'rss_peak_gib': None,
                   'speedup_vs256': None, 'max_matrix_error_hartree': None,
                   'max_gradient_error': None}
            if not path.exists():
                row['error'] = failures.get((identifier, block), {}).get('error', 'No result file')
                rows.append(row)
                continue
            record = json.loads(path.read_text())
            validation = {}
            if reference is not None and identity is not None:
                with np.load(record['arrays'], allow_pickle=False) as arrays:
                    for key in ('potential', 'offset', 'directions', 'reward_gradient'):
                        validation[key] = compare_arrays(arrays[key], reference[key],
                            atol=1e-7 if key=='reward_gradient' else 1e-8,
                            rtol=1e-8 if key=='reward_gradient' else 1e-9)
                fields = ('source','benchmark_source','matrix_source','archive_sha256',
                          'metadata_sha256','baseline_cache_sha256','candidate_digest',
                          'parameters','parameter_order','nao','grid_points','grid_level',
                          'device','node','job_id','float64','versions')
                changed = [key for key in fields if record.get(key)!=identity.get(key)]
                validation['provenance'] = {'passed': not changed, 'changed_fields': changed}
            valid = (record['status']=='ok' and bool(validation) and
                     all(check['passed'] for check in validation.values()))
            checks[f'{identifier}/{block}'] = validation
            timing = record['timings']
            row.update(status='ok' if valid else 'numerical_mismatch',
                grid_points=record['grid_points'], grid_level=record['grid_level'],
                nao=record['nao'], grid_blocks=record['grid_blocks'],
                warm_assembly_seconds=sum(timing[k]['warm_median_seconds'] for k in ('matrix','directions')),
                warm_assembly_min_seconds=sum(min(timing[k]['warm_seconds']) for k in ('matrix','directions')),
                warm_assembly_max_seconds=sum(max(timing[k]['warm_seconds']) for k in ('matrix','directions')),
                compile_seconds=sum(timing[k]['trace_lower_seconds']+timing[k]['compile_seconds']
                                    for k in ('matrix','directions')),
                first_assembly_seconds=sum(timing[k]['first_execution_seconds'] for k in ('matrix','directions')),
                score_gradient_seconds=timing['score_gradient']['warm_median_seconds'],
                density_diagnostic_seconds=timing['density_diagnostic']['warm_median_seconds'],
                gpu_peak_gib=record['gpu_peak_bytes']/1024**3,
                rss_peak_gib=record['rss_peak_gib'],
                max_matrix_error_hartree=max((validation.get(k,{}).get('max_abs_error') or 0.)
                                            for k in ('potential','offset','directions')),
                max_gradient_error=validation.get('reward_gradient',{}).get('max_abs_error'))
            records[block] = row
            rows.append(row)
        if 256 in records and records[256]['status']=='ok':
            baseline_time = records[256]['warm_assembly_seconds']
            for row in records.values():
                if row['status']=='ok':
                    row['speedup_vs256'] = baseline_time/row['warm_assembly_seconds']

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output/'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    write_json(output/'summary.json', {'rows': rows, 'block256_checks': checks, 'sweep': sweep})

    render_plot(rows, sweep, output)

    lines = ['# XC grid-block benchmark', '',
        'Fixed cc-pVTZ, original XC100 grids, float64 and fitted round-2 formula/parameters. '
        'One V100; each target/block pair is a fresh process. All numerical work used Slurm.', '',
        'Warm assembly is the sum of median matrix and parameter-direction stage times '
        '(three synchronized repetitions per stage), not full end-to-end fitting time. '
        'Error bars bound the observed sums of per-stage minima/maxima; they are not '
        'confidence intervals. Compilation is timed separately. PBE matrices are reused unchanged.', '',
        'The GPU metric is JAX peak live-allocation bytes, not total nvidia-smi usage. '
        'Process RSS includes preparation and compiler memory. Driver caches and node '
        'activity outside this allocation are not fully isolated.', '',
        'Block-256 matrix/gradient comparisons and original round-2 scalar checks must '
        'pass before a point receives a speedup. Failed/timeout points remain in CSV.', '']
    for identifier in sweep['ids']:
        lines += [f'## {labels[identifier]}', '',
                  '| Block | Warm assembly (s) | Compile (s) | GPU peak (GiB) | RSS (GiB) | Speedup | Status |',
                  '|---:|---:|---:|---:|---:|---:|---|']
        for row in (r for r in rows if r['system_id']==identifier):
            show = lambda key: f'{row[key]:.4g}' if row.get(key) is not None else '—'
            lines.append(f"| {row['block_size']} | {show('warm_assembly_seconds')} | "
                         f"{show('compile_seconds')} | {show('gpu_peak_gib')} | "
                         f"{show('rss_peak_gib')} | {show('speedup_vs256')} | {row['status']} |")
        lines.append('')
    lines += ['These three molecules have different AO counts and grids. The curves measure '
              'block-size sensitivity and do not establish an asymptotic basis-size exponent.', '']
    (output/'REPORT.md').write_text('\n'.join(lines))
    return rows
