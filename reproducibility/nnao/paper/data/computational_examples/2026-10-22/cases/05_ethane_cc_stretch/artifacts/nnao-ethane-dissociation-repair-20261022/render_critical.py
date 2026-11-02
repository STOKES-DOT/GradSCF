"""Show the corrected critical interval and the actual endpoint gradients."""
from pathlib import Path
import csv
import json
import os
import numpy as np

OUT = Path(__file__).resolve().parent
BENCHMARK = OUT.parent / 'nnao-ethane-dissociation-benchmark-20261022'
os.environ.setdefault('MPLCONFIGDIR', str(BENCHMARK / '.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    rows = list(csv.DictReader((BENCHMARK / 'curves.csv').open()))
    audit = json.loads((OUT / 'archived-native-audit.json').read_text())['results']
    native = json.loads((BENCHMARK / 'nnao-results.json').read_text())['points']
    critical = [r for r in native if 2.7 <= r['distance_angstrom'] <= 2.9]
    assert len(critical) == 5 and all(r['gradient_norm'] <= 1e-5 for r in critical)
    styles = [('6-31G(d,p)', '#4A6342', ':'), ('cc-pVDZ', '#B75B7E', '--'),
              ('cc-pVTZ', '#252A31', '-'), ('NNAO / MACE', '#D7752B', '-'),
              ('NNAO / free contraction', '#246999', '--')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), gridspec_kw={'width_ratios': [1.25, 1.]})
    for name, color, style in styles:
        selected = [r for r in rows if r['basis'] == name and 2.55 <= float(r['distance_angstrom']) <= 3.1]
        axes[0].plot([float(r['distance_angstrom']) for r in selected],
            [float(r['plot_energy_hartree']) for r in selected], color=color, linestyle=style,
            linewidth=1.8, label=name)
    axes[0].scatter([r['distance_angstrom'] for r in critical], [r['free_energy_hartree'] for r in critical],
                    color='#246999', s=28, zorder=4)
    axes[0].set(xlabel='C-C separation (Angstrom)', ylabel='SCF total energy (Eh)',
                title='Corrected energy curves, 2.55-3.10 Angstrom')
    axes[0].legend(frameon=False, fontsize=8.8, loc='upper left')
    x = [r['distance_angstrom'] for r in critical]
    old = {r['index']: r for r in audit}
    axes[1].semilogy(x, [old[r['index']]['gradient_norm'] for r in critical],
                     color='#858C94', marker='x', linestyle='--', label='Archived coefficients, re-evaluated')
    axes[1].semilogy(x, [r['gradient_norm'] for r in critical],
                     color='#246999', marker='o', label='Reoptimized coefficients')
    axes[1].axhline(1e-5, color='#252A31', linestyle=':', linewidth=1.2, label='Acceptance: 1e-5')
    axes[1].set(xlabel='C-C separation (Angstrom)', ylabel='Canonical coefficient gradient norm (Eh)',
                title='All five endpoints now meet the criterion', xticks=x, ylim=(3e-7, 1.))
    axes[1].legend(frameon=False, fontsize=8.5, loc='upper left')
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(color='#E5E8EA', linewidth=.65)
    fig.suptitle('Ethane C-C scan: SCF branch repair and contraction convergence', fontsize=14, y=.98)
    fig.text(.075, .035,
        'Same geometry, 44-AO NNAO, fixed exponents/core and frozen MACE update 3351; restricted DF-RHF.\n'
        'Gradients use the existing implicit SCF, with per-radial unit-norm raw coefficients. No interpolated energies or finite differences.',
        fontsize=8.5, color='#59636B')
    fig.subplots_adjust(left=.075, right=.98, bottom=.21, top=.84, wspace=.28)
    for ext in ('png', 'pdf'):
        fig.savefig(BENCHMARK / f'ethane-cc-critical-repaired.{ext}', dpi=220, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
