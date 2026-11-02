"""Measured dissociation curves and stationary-state reaction energy diagrams."""
from collections import Counter
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

import numpy as np

FOCAL = ['6-31g(d,p)', 'cc-pvdz', 'cc-pvtz', 'NNAO/MACE']
COLORS = {'6-31g(d,p)': '#8D933E', 'cc-pvdz': '#5C8C9E',
          'cc-pvtz': '#28699C', 'NNAO/MACE': '#D87632', 'Reference': '#20252A'}


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def formula(row):
    counts = Counter(row['symbols'])
    order = ([s for s in ['C', 'H'] if s in counts] + sorted(counts.keys()-{'C', 'H'}))
    if 'C' not in counts:
        order = sorted(counts)
    text = ''.join(s+(str(counts[s]) if counts[s] != 1 else '') for s in order)
    if row['charge']:
        q = row['charge']
        text += f'({abs(q) if abs(q) != 1 else ""}{"+" if q > 0 else "-"})'
    if row['spin']:
        text += f' [2S+1={row["spin"]+1}]'
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    os.environ.setdefault('MPLCONFIGDIR', str(root/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads((root/'results.json').read_text())
    dataset = json.loads(args.dataset.read_text())
    config = json.loads((root/'configuration.json').read_text())
    good = [r for r in report['results'] if r['status'] == 'passed']
    bases = [b for b in FOCAL if b in config['bases']]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    written = []
    def save(fig, name):
        for extension in ['png', 'pdf']:
            path = root/f'{name}.{extension}'
            fig.savefig(path, dpi=190, facecolor='white')
            written.append(str(path))
        plt.close(fig)

    curve_rows = [r for r in good if r['task'] == 'curve']
    fields = ['dataset', 'group', 'scale', 'basis', 'model_interaction_kcal_mol',
              'cp_interaction_kcal_mol', 'bsse_kcal_mol', 'reference_kcal_mol', 'seconds']
    write_csv(root/'interaction-cases.csv', [{k: r[k] for k in fields} for r in curve_rows])
    for dataset_name in sorted({r['dataset'] for r in curve_rows}):
        groups = list(dict.fromkeys(r['group'] for r in dataset['curves'] if r['dataset'] == dataset_name))
        for value, suffix, mode in [('model_interaction_kcal_mol', 'model', 'Independent dimer and isolated-monomer predictions'),
                                     ('cp_interaction_kcal_mol', 'cp-control', 'Frozen dimer-basis counterpoise diagnostic')]:
            fig, axes = plt.subplots((len(groups)+1)//2, 2, figsize=(12, 3.6*((len(groups)+1)//2)), squeeze=False)
            for ax, group in zip(axes.flat, groups):
                expected = [r for r in dataset['curves'] if r['dataset'] == dataset_name and r['group'] == group]
                rows = [r for r in curve_rows if r['dataset'] == dataset_name and r['group'] == group]
                scales = sorted(r['scale'] for r in expected)
                ref = {r['scale']: r['reference_kcal_mol'] for r in expected}
                ax.plot(scales, [ref[x] for x in scales], 'k--', lw=1.6, label='Correlated reference')
                for basis in bases:
                    available = {r['scale']: r[value] for r in rows if r['basis'] == basis}
                    y = [available.get(x, np.nan) for x in scales]
                    ax.plot(scales, y, marker='s' if basis == 'NNAO/MACE' else 'o', markersize=3.5,
                            ls='--' if basis == 'NNAO/MACE' else '-', color=COLORS[basis], label=basis)
                ax.axhline(0., color='#A9ADB0', lw=.7)
                ax.set_title(expected[0]['label'].replace('--', ' + '))
                ax.set_xlabel('Original intermolecular distance scale')
                ax.set_ylabel('Interaction energy (kcal/mol)')
                ax.grid(color='#E5E8EA', lw=.6)
            for ax in axes.flat[len(groups):]:
                ax.set_visible(False)
            axes.flat[0].legend(frameon=False, fontsize=8)
            fig.suptitle(f'{dataset_name}: representative distance curves / DF-HF', fontsize=14)
            fig.text(.08, .018, mode+'. Missing calculations remain gaps.\n'
                     'HF lacks correlated dispersion; the correlated reference is not a pure basis-error target.', fontsize=8.5)
            fig.tight_layout(rect=(0, .065, 1, .95))
            save(fig, f'{dataset_name}-{suffix}')

    reactions = [r for r in report.get('reactions', []) if r['status'] == 'passed']
    columns = ['identifier', 'category', 'basis', 'forward_kcal_mol', 'reverse_kcal_mol',
               'reaction_kcal_mol', 'reference_forward_kcal_mol', 'reference_reverse_kcal_mol',
               'reference_reaction_kcal_mol']
    write_csv(root/'reaction-cases.csv', [{k: r[k] for k in columns} for r in reactions])

    def profile(ax, definition, annotate=False):
        rows = [r for r in reactions if r['identifier'] == definition['identifier']]
        reference = [0., definition['reference_forward_kcal_mol'], definition['reference_reaction_kcal_mol']]
        series = [('Reference', reference)] + [(b, next((r['relative_kcal_mol'] for r in rows if r['basis'] == b), None)) for b in bases]
        for basis, levels in series:
            if levels is None:
                continue
            color = COLORS[basis]
            for i, energy in enumerate(levels):
                ax.hlines(energy, i-.15, i+.15, color=color, lw=2)
            for i in range(2):
                ax.plot([i+.15, i+1-.15], levels[i:i+2], color=color, lw=1.3,
                        ls='--' if basis in ['Reference', 'NNAO/MACE'] else '-')
            ax.plot([], [], color=color, ls='--' if basis in ['Reference', 'NNAO/MACE'] else '-', label=basis)
        ax.axhline(0., color='#B8BFC2', lw=.6)
        ax.set_xticks([0, 1, 2], ['Reactants', r'TS $^\ddagger$', 'Products'])
        ax.set_xlim(-.3, 2.6 if annotate else 2.3)
        ax.set_ylabel(r'$\Delta E$ (kcal/mol)')
        ax.set_title(definition['identifier']+' / class '+definition['category']+'\n'+definition.get('category_name', ''))
        ax.grid(axis='y', color='#E5E8EA', lw=.6)
        if annotate:
            for x, lower, upper, label in [(.55, 0., reference[1], r'$\Delta E_f^\ddagger$'),
                    (1.45, reference[2], reference[1], r'$\Delta E_r^\ddagger$'),
                    (2.35, 0., reference[2], r'$\Delta E_{rxn}$')]:
                ax.annotate('', xy=(x, upper), xytext=(x, lower),
                            arrowprops=dict(arrowstyle='<->', color='#40464A', lw=.9))
                ax.text(x+.05, (lower+upper)/2, label, fontsize=9, va='center')

    shown = [r for r in dataset['reactions'] if any(x['identifier'] == r['identifier'] for x in reactions)]
    if shown:
        nrows = (len(shown)+2)//3
        fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.6*nrows), squeeze=False)
        for ax, definition in zip(axes.flat, shown):
            profile(ax, definition)
        for ax in axes.flat[len(shown):]:
            ax.set_visible(False)
        axes.flat[0].legend(frameon=False, fontsize=8)
        title = ('BH9 published reference profiles: layout preview, no NNAO results'
                 if report['status'] == 'reference_only' else 'BH9 representative electronic reaction energy profiles')
        fig.suptitle(title, fontsize=15)
        fig.text(.08, .02, 'Each method uses its own reactant zero. Connectors are schematic: these are R/TS/P single points, not IRC curves.\n'
                 'No ZPE, thermal or entropy terms. Reference barriers and reaction energies are rounded independently.', fontsize=8.5)
        fig.tight_layout(rect=(0, .07, 1, .95))
        save(fig, 'BH9-reaction-profiles')
        for definition in shown:
            fig, (ax, table_ax) = plt.subplots(2, 1, figsize=(9, 7.5), gridspec_kw={'height_ratios': [3, 1.2]})
            profile(ax, definition, annotate=True)
            ax.legend(frameon=False, fontsize=8, ncol=3)
            def side(terms):
                return ' + '.join((str(n)+' ' if n != 1 else '')+formula(dataset['structures'][name]) for name, n in terms.items())
            ax.set_xlabel(side(definition['reactants'])+'  →  '+side(definition['products']), fontsize=9)
            table_rows = [['Reference', f'{definition["reference_forward_kcal_mol"]:.2f}',
                f'{definition["reference_reverse_kcal_mol"]:.2f}', f'{definition["reference_reaction_kcal_mol"]:.2f}']]
            for row in reactions:
                if row['identifier'] == definition['identifier'] and row['basis'] != 'Reference':
                    table_rows.append([row['basis']]+[f'{row[k]:.2f}' for k in ['forward_kcal_mol', 'reverse_kcal_mol', 'reaction_kcal_mol']])
            table_ax.axis('off')
            table_ax.table(cellText=table_rows, colLabels=['Method/basis', 'Forward', 'Reverse', 'Reaction'], loc='center', cellLoc='center')
            fig.text(.1, .025, 'Electronic energies, kcal/mol. Reference arrows; schematic connectors, not IRC.\n'
                     'Reference closure differs by at most 0.01 kcal/mol due to published rounding.', fontsize=8.5)
            fig.tight_layout(rect=(0, .075, 1, 1))
            save(fig, definition['identifier']+'-profile')

    metrics = []
    for basis in config['bases']:
        for quantity in ['forward', 'reverse', 'reaction']:
            rows = [r for r in reactions if r['basis'] == basis]
            anchor = {r['identifier']: r for r in reactions if r['basis'] == 'cc-pvtz'}
            paired = [r for r in rows if r['identifier'] in anchor]
            if rows:
                key = quantity+'_kcal_mol'
                metrics.append(dict(basis=basis, quantity=quantity, cases=len(rows),
                    mae_vs_correlated_reference=float(np.mean([abs(r[key]-r['reference_'+key]) for r in rows])),
                    anchor_cases=len(paired), mae_vs_HF_ccpVTZ=float(np.mean([abs(r[key]-anchor[r['identifier']][key]) for r in paired])) if paired else None))
    write_csv(root/'reaction-metrics.csv', metrics)
    summary = dict(status=report['status'], passed=report['passed'], failed=report['failed'],
        curves=len(curve_rows), reaction_profiles=len(reactions), figures=written,
        sources_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [root/'results.json', root/'configuration.json', args.dataset, Path(__file__)]})
    (root/'plot-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'sources_sha256'}))


if __name__ == '__main__':
    main()
