"""Plot available, explicitly qualified C-C-stretch results after download."""
from pathlib import Path
import csv
import importlib.util
import json
import os

import numpy as np

OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

spec = importlib.util.spec_from_file_location('radial_reconstruction',
    OUT.parent/'nnao-ethane-radial-20261019'/'plot_radial.py')
radial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(radial)


def main():
    rows = json.loads((OUT/'structures.json').read_text())
    distances = np.array([row['distance_angstrom'] for row in rows])
    data, moments, energies, results = {}, [], [], []
    baseline = {}
    for i, row in enumerate(rows):
        folder = OUT/f'distance-{i:03d}'
        result = json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {'status': 'pending'}
        results.append(result)
        path = folder/'frozen-bases.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        for method, label in radial.METHODS.items():
            record = next((record for record in records if record['basis'] == label), None)
            # A MACE function exists before SCF. Free curves require checked endpoints.
            if record is None or (method == 'Free' and result['status'] != 'passed'):
                continue
            assert record['nao'] == 44
            np.testing.assert_allclose(record['coords_angstrom'], row['coords_angstrom'], atol=1e-12, rtol=0)
            for l, shell_index in [(0, 1), (1, 2)]:
                alpha, raw, weights = radial.normalized_weights(record['atom_shells'][0][shell_index])
                assert record['atom_shells'][0][shell_index][0] == l
                if l not in baseline:
                    assert i == 0 and method == 'MACE'
                    main_lobe = radial.R*radial.evaluate(l, alpha, weights, radial.R)
                    weights *= 1 if main_lobe[np.argmax(abs(main_lobe))] > 0 else -1
                    baseline[l] = (alpha.copy(), weights.copy())
                np.testing.assert_array_equal(alpha, baseline[l][0])
                if weights@radial.moment_matrix(l, alpha)@baseline[l][1] < 0:
                    weights *= -1
                function = radial.evaluate(l, alpha, weights, radial.R)
                data[method, l, i] = function
                mean = float(weights@radial.moment_matrix(l, alpha, 1)@weights)
                norm_error = abs(float(weights@radial.moment_matrix(l, alpha)@weights)-1)
                assert norm_error < 1e-12
                moments.append(dict(method=method, channel='sp'[l],
                    distance_angstrom=row['distance_angstrom'], mean_radius_bohr=mean,
                    norm_error=norm_error, paired_result_status=result['status']))
        energies.append(dict(distance_angstrom=row['distance_angstrom'], status=result['status'],
            mace_energy_hartree=result.get('mace_energy_hartree'),
            free_energy_hartree=result.get('free_energy_hartree'),
            gap_kcal_mol=result.get('gap_kcal_mol'), gradient_norm=result.get('gradient_norm')))
    if not moments:
        raise RuntimeError('No frozen radial functions available yet.')
    for name, records in [('radial-metrics.csv', moments), ('energy-comparison.csv', energies)]:
        with (OUT/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    selected = [0]+[int(np.argmin(abs(distances-d))) for d in [2, 2.5, 3, 4, 5, 8]]
    normalizer = matplotlib.colors.Normalize(distances[0], 8)
    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey='col')
    for method_index, method in enumerate(['MACE', 'Free']):
        for l in (0, 1):
            ax = axes[method_index, l]
            for i in selected:
                if (method, l, i) in data:
                    ax.plot(radial.R, data[method, l, i], lw=1.7,
                            color=cmap(normalizer(distances[i])))
            ax.set_title(f'{method}: C0 valence {"sp"[l]}')
            ax.set_xlim(0, 6)
            ax.set_ylabel(r'$R_l(r)$ ($a_0^{-3/2}$)')
            ax.grid(color='#E6E9EB', lw=.6)
            if method_index:
                ax.set_xlabel(r'Distance from C0, $r$ ($a_0$)')
    fig.subplots_adjust(left=.09, right=.86, top=.88, bottom=.13, wspace=.25, hspace=.3)
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=normalizer, cmap=cmap),
                       cax=fig.add_axes([.89, .2, .02, .6]))
    bar.set_label('C-C separation (Angstrom)')
    passed = sum(record['status'] == 'passed' for record in results)
    fig.suptitle(f'Ethane C-C stretch: radial functions ({passed}/56 checked pairs)', fontsize=14)
    fig.text(.09, .025, 'Rigid methyl fragments; restricted DF-RHF, 44 AOs, fixed exponents and core.\n'
             'Free curves: gradient-converged pairs only. Restricted-branch pressure test, not physical homolytic dissociation.', fontsize=8.2)
    for extension in ['png', 'pdf']:
        fig.savefig(OUT/f'carbon-radial-functions.{extension}', dpi=200, facecolor='white')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.7))
    for l in (0, 1):
        for method in ['MACE', 'Free']:
            y = np.full(len(rows), np.nan)
            for item in moments:
                if item['channel'] == 'sp'[l] and item['method'] == method:
                    i = int(np.argmin(abs(distances-item['distance_angstrom'])))
                    y[i] = item['mean_radius_bohr']
            axes[0, l].plot(distances, y, color=radial.COLORS[method],
                marker='s' if method == 'MACE' else 'o', markersize=3, lw=1.7, label=method)
        axes[0, l].set_title(f'C0 valence {"sp"[l]}')
        axes[0, l].set_ylabel(r'$\langle r\rangle$ ($a_0$)')
        axes[0, l].ticklabel_format(axis='y', useOffset=False)
    for method, key in [('MACE', 'mace_energy_hartree'), ('Free', 'free_energy_hartree')]:
        y = np.array([record[key] if record['status'] == 'passed' else np.nan for record in energies])
        axes[1, 0].plot(distances, (y-y[0])*627.5094740631, color=radial.COLORS[method],
                       marker='s' if method == 'MACE' else 'o', markersize=3, label=method)
    y = np.array([record['gap_kcal_mol'] if record['status'] == 'passed' else np.nan for record in energies])
    axes[1, 1].plot(distances, y, color='#485661', marker='o', markersize=3)
    axes[1, 0].set_ylabel(r'$E(R)-E(R_0)$ (kcal/mol)')
    axes[1, 1].set_ylabel(r'$E_{MACE}-E_{free}$ (kcal/mol)')
    for ax in axes.flat:
        ax.set_xlabel('C-C separation (Angstrom)')
        ax.grid(color='#E6E9EB', lw=.6)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f'RHF pressure test: radial response and energies ({passed}/56 checked pairs)', fontsize=13)
    fig.text(.1, .025, ('All 56 pairs meet the endpoint criterion; no missing points or interpolated energies.\n'
             if passed == len(rows) else 'Missing or unconverged pairs remain gaps; no interpolation across failed points.\n')+
             'Energy curves use their own initial reference. Large-R RHF is not the physical dissociation limit.', fontsize=8.5)
    fig.subplots_adjust(left=.1, right=.98, top=.9, bottom=.14, wspace=.3, hspace=.3)
    for extension in ['png', 'pdf']:
        fig.savefig(OUT/f'carbon-radial-response.{extension}', dpi=200, facecolor='white')
    plt.close(fig)
    print(json.dumps(dict(paired_passed=passed, requested=56, radial_records=len(moments),
                         pending=sum(r['status'] == 'pending' for r in results))))


if __name__ == '__main__':
    main()
