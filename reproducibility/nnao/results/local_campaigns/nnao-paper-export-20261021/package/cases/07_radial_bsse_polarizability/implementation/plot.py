"""Source-backed plots of completed benchmark cases; failures remain visible."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

import numpy as np
from scipy.integrate import simpson


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


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
    good = [row for row in report['results'] if row['status'] == 'passed']
    bases = config['bases']+(['aug-cc-pv5z'] if any(row['basis'] == 'aug-cc-pv5z' for row in good) else [])
    colors = dict(zip(['sto-6g', 'STO-6G+P', '6-31g(d,p)', 'def2-svp', 'cc-pvdz',
                      'cc-pvtz', 'ma-def2-qzvpp', 'NNAO/MACE', 'aug-cc-pv5z'],
                     ['#A6ADB3', '#8753A0', '#8D933E', '#B46A8C', '#5C8C9E',
                      '#28699C', '#26313A', '#D87632', '#111111']))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        for extension in ['png', 'pdf']:
            fig.savefig(root/f'{name}.{extension}', dpi=190, facecolor='white')
        plt.close(fig)

    def style(ax):
        ax.grid(color='#E5E8EA', lw=.65)
        ax.set_axisbelow(True)

    atomic = [row for row in good if row['task'] == 'atoms']
    labels = [row['label'] for row in dataset['atoms'] if any(r['label'] == row['label'] for r in atomic)]
    atom_table = []
    if labels:
        fig, axes = plt.subplots((len(labels)+1)//2, min(2, len(labels)),
                                 figsize=(11, 4.0*((len(labels)+1)//2)), squeeze=False)
        for ax, label in zip(axes.flat, labels):
            reference = next((r for r in atomic if r['label'] == label and r['basis'] == 'aug-cc-pv5z'), None)
            reference_data = np.load(root/reference['radial_file']) if reference else None
            for row in [r for r in atomic if r['label'] == label]:
                with np.load(root/row['radial_file']) as data:
                    radius, values = data['radius_bohr'], data['normalized_radial_density']
                    ax.plot(radius, values, color=colors[row['basis']], lw=2 if row['basis'] == 'NNAO/MACE' else 1.3,
                            ls='--' if row['basis'] == 'NNAO/MACE' else '-', label=row['basis'])
                    error = None
                    if reference_data is not None:
                        np.testing.assert_array_equal(radius, reference_data['radius_bohr'])
                        error = float(simpson(abs(values-reference_data['normalized_radial_density']), x=radius))
                    atom_table.append(dict(atom=label, basis=row['basis'], nao=row['nao'],
                        energy_hartree=row['energy_hartree'], radial_electron_number=row['radial_electron_number'],
                        mean_radius_bohr=row['mean_radius_bohr'], spin_squared=row['spin_squared'],
                        L1_normalized_density_difference_vs_reference=error))
            if reference_data is not None:
                reference_data.close()
            ax.set_title(label+' atom: angularly averaged electron density')
            ax.set_xlim(0, 8 if label.startswith('H') else 5)
            ax.set_xlabel(r'Radius ($a_0$)')
            ax.set_ylabel(r'$P(r)/N_e$ ($a_0^{-1}$)')
            style(ax)
        for ax in axes.flat[len(labels):]:
            ax.set_visible(False)
        axes.flat[0].legend(frameon=False, fontsize=8.5)
        fig.suptitle('Atomic radial density: explicit charge and spin, DF-HF', fontsize=14)
        fig.text(.09, .025, 'Every P(r) integrates to the electron number; displayed shapes are normalized per electron.\n'
                 'MACE atomic charge/spin states extrapolate beyond its neutral closed-shell molecular training.',
                 fontsize=8.5, color='#59636B')
        fig.tight_layout(rect=(0, .10, 1, .94))
        save(fig, 'atomic-radial-density')
        write_csv(root/'atomic-summary.csv', atom_table)

    cp = [row for row in good if row['task'] == 'bsse']
    if cp:
        cp_bases = [basis for basis in bases if any(row['basis'] == basis for row in cp)]
        available = [{r['structure_id'] for r in cp if r['basis'] == b} for b in cp_bases]
        common = sorted(set.intersection(*available), key=lambda k: int(k.split('/')[-1]))
        if not common:
            raise ValueError('No common converged CP cases for an aggregate comparison.')
        means, percentages = [], []
        for basis in cp_bases:
            selected = [r for r in cp if r['basis'] == basis and r['structure_id'] in common]
            means.append(np.mean([r['bsse_kcal_mol'] for r in selected]))
            percentages.append(np.mean([r['bsse_percent_reference_binding'] for r in selected]))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
        for ax, values, title, unit in zip(axes, [means, percentages],
            ['Mean orbital BSSE', 'Mean relative CP correction'], ['kcal/mol', '% of reference binding magnitude']):
            ax.bar(np.arange(len(cp_bases)), values, color=[colors[b] for b in cp_bases], width=.65)
            ax.set_xticks(np.arange(len(cp_bases)), cp_bases, rotation=32, ha='right')
            ax.set_ylabel(unit)
            ax.set_title(title)
            ax.set_ylim(bottom=0)
            style(ax)
        fig.suptitle(f'S22 counterpoise: {len(common)} common converged dimers', fontsize=14)
        fig.text(.09, .025, 'Frozen dimer coefficients, frozen fragment geometry and one common auxiliary metric in all five energies.\n'
                 'Percentage denominator: upstream reference dissociation magnitude. These are HF CP corrections, not binding-energy errors.',
                 fontsize=8.2, color='#59636B')
        fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.35, wspace=.35)
        save(fig, 's22-counterpoise')
        write_csv(root/'s22-summary.csv', [dict(basis=b, common_dimers=len(common),
            mean_bsse_kcal_mol=float(v), mean_relative_bsse_percent=float(p))
            for b, v, p in zip(cp_bases, means, percentages)])
        write_csv(root/'s22-cases.csv', [{k: row[k] for k in ['structure_id', 'label', 'basis', 'nao',
            'bsse_kcal_mol', 'bsse_percent_reference_binding', 'uncorrected_interaction_kcal_mol',
            'cp_interaction_kcal_mol']} for row in cp])

    polar = [row for row in good if row['task'] == 'polarizability']
    polar_labels = [row['label'] for row in dataset['polarizability']
                    if any(r['label'] == row['label'] for r in polar)]
    if polar:
        fig, ax = plt.subplots(figsize=(11, 5.7))
        for basis in bases:
            values = [next((r['isotropic_au'] for r in polar if r['basis'] == basis and r['label'] == label), np.nan)
                      for label in polar_labels]
            if not np.isfinite(values).any():
                continue
            ax.plot(np.arange(len(polar_labels)), values, color=colors[basis],
                    marker='s' if basis == 'NNAO/MACE' else 'o', markersize=4,
                    ls='--' if basis == 'NNAO/MACE' else '-', lw=1.8, label=basis)
        ax.set_xticks(np.arange(len(polar_labels)), polar_labels, rotation=32, ha='right')
        ax.set_ylabel(r'Isotropic static polarizability ($a_0^3$)')
        ax.set_title('Zero-field linear response in fixed predicted or conventional bases')
        ax.legend(frameon=False, fontsize=8.5, ncol=2)
        style(ax)
        fig.text(.09, .025, 'Available completed cases from the nine-molecule dataset; independent PubChem 3D conformers, HF electronic structure.\n'
                 'MACE coefficients are held fixed under field. No field-adaptive MACE response or finite fields are introduced.',
                 fontsize=8.5, color='#59636B')
        fig.subplots_adjust(left=.09, right=.98, top=.88, bottom=.3)
        save(fig, 'static-polarizability')
        write_csv(root/'polarizability-summary.csv', [dict(molecule=row['label'], basis=row['basis'],
            nao=row['nao'], isotropic_bohr3=row['isotropic_au'],
            max_response_residual=max(row['response_residuals']), seconds=row['seconds']) for row in polar])

    summary = dict(status=report['status'], passed=report['passed'], failed=report['failed'],
        requested=report['requested_cases'], phase=config['profile'],
        atomic_records=len(atomic), cp_records=len(cp), polar_records=len(polar),
        errors=[{k: row.get(k) for k in ['task', 'structure_id', 'basis', 'error']}
                for row in report['results'] if row['status'] != 'passed'],
        inputs_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [root/'results.json', root/'configuration.json', args.dataset, Path(__file__)]})
    (root/'plot-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
