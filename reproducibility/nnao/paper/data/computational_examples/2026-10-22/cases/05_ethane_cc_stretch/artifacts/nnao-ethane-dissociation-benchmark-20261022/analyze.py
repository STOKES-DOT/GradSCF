"""Reprocess the archived NNAO scan and aligned conventional-basis results."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'nnao-ethane-dissociation-20261019'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

EH_TO_KCAL = 627.5094740631
STYLES = {
    'STO-3G': ('#A8ADB3', ':', 'x'),
    'STO-6G': ('#737B84', '--', '+'),
    'STO-6G + C(d), H(p)': ('#B18A4B', '-.', 'v'),
    '3-21G': ('#89926A', '--', '^'),
    '6-31G(d,p)': ('#4A6342', ':', 'v'),
    'cc-pVDZ': ('#B75B7E', '--', 'D'),
    'cc-pVTZ': ('#252A31', '-', None),
    'NNAO / MACE': ('#D7752B', '-', 's'),
    'NNAO / free contraction': ('#246999', '--', 'o'),
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    geometry = read(SOURCE / 'structures.json')
    x = np.array([r['distance_angstrom'] for r in geometry])
    assert len(x) == 56 and np.all(np.diff(x) > 0)
    nnao_path = OUT / 'nnao-results.json'
    repaired = nnao_path.exists()
    nnao = read(nnao_path if repaired else SOURCE / 'scan-summary.json')
    conventional = read(OUT / 'traditional-results.json')
    assert conventional['status'] == 'complete' and len(conventional['results']) == 392
    assert conventional['identity']['geometry_sha256'] == digest(SOURCE / 'structures.json')
    points = sorted(nnao['points'], key=lambda r: r['distance_angstrom'])
    np.testing.assert_allclose([r['distance_angstrom'] for r in points], x, atol=1e-12, rtol=0)
    curves = {name: np.full(len(x), np.nan) for name in STYLES}
    raw = []
    for row in conventional['results']:
        i = row['index']
        assert row['structure_id'] == geometry[i]['structure_id']
        assert abs(row['distance_angstrom'] - x[i]) < 1e-12
        accepted = row['status'] == 'passed'
        if accepted:
            assert row['converged'] and row['internal_stable'] and row['orbital_residual'] < 1e-7
            curves[row['basis']][i] = row['energy_hartree']
        raw.append(dict(basis=row['basis'], index=i, distance_angstrom=x[i],
            energy_hartree=row.get('energy_hartree'), accepted=accepted, status=row['status'],
            nao=row.get('nao'), orbital_residual=row.get('orbital_residual'),
            coefficient_gradient_norm=None, source='traditional-results.json'))
    rejected = []
    for i, point in enumerate(points):
        summary = point if repaired else read(SOURCE / point['folder'] / 'summary.json')
        for label, energy_key, diagnostic_key in (
            ('NNAO / MACE', 'mace_energy_hartree', 'mace_scf'),
            ('NNAO / free contraction', 'free_energy_hartree', 'free_scf' if repaired else 'best_scf')):
            scf = summary[diagnostic_key]
            scf_ok = scf['converged'] and scf['stable'] and scf['orbital_residual'] < 1e-7
            accepted = bool(scf_ok and (label.endswith('MACE') or point['free_gradient_converged']))
            if accepted:
                curves[label][i] = point[energy_key]
            if label.endswith('contraction') and not accepted:
                rejected.append(float(x[i]))
            raw.append(dict(basis=label, index=i, distance_angstrom=x[i], energy_hartree=point[energy_key],
                accepted=accepted, status='passed' if accepted else 'coefficient_not_converged',
                nao=point['nao'], orbital_residual=scf['orbital_residual'],
                coefficient_gradient_norm=point.get('gradient_norm') if label.endswith('contraction') else None,
                source=f'nnao-results.json/point-{i:03d}' if repaired else
                    str((SOURCE / point['folder'] / 'summary.json').relative_to(OUT.parent))))
    assert np.isfinite(curves['NNAO / MACE']).sum() == 56
    assert np.isfinite(curves['NNAO / free contraction']).sum() == (56 if repaired else 51)
    assert np.isfinite(curves['cc-pVTZ']).sum() == 56
    rows = []
    for record in raw:
        name, i = record['basis'], record['index']
        record['plot_energy_hartree'] = float(curves[name][i]) if record['accepted'] else None
        record['relative_energy_kcal_mol'] = float((curves[name][i] - curves[name][0]) * EH_TO_KCAL) if record['accepted'] else None
        record['minus_cc_pvtz_kcal_mol'] = float((curves[name][i] - curves['cc-pVTZ'][i]) * EH_TO_KCAL) if record['accepted'] else None
        rows.append(record)
    rows.sort(key=lambda r: (list(STYLES).index(r['basis']), r['index']))
    table(OUT / 'curves.csv', rows)
    summaries = []
    for name, energy in curves.items():
        selected = [r for r in rows if r['basis'] == name]
        summaries.append(dict(basis=name, nao=selected[0]['nao'], accepted_points=int(np.isfinite(energy).sum()),
            requested_points=len(x), initial_energy_hartree=float(energy[0]),
            final_energy_hartree=float(energy[-1]), stretch_8A_kcal_mol=float((energy[-1] - energy[0]) * EH_TO_KCAL),
            final_minus_cc_pvtz_kcal_mol=float((energy[-1] - curves['cc-pVTZ'][-1]) * EH_TO_KCAL)))
    table(OUT / 'summary.csv', summaries)
    # Independently check the new CPU reference energies at the archived torsion geometry.
    old = read(OUT.parent / 'nnao-ethane-torsion-20261018/traditional-results.json')['results']
    old += read(OUT.parent / 'nnao-ethane-sto6g-polarized-20261020/results.json')['results']
    baseline = {r['basis'].lower(): r['energy_hartree'] for r in old if r['angle_deg'] == 0}
    errors = {n: float(v[0] - baseline[n.lower()]) for n, v in curves.items() if not n.startswith('NNAO')}
    assert max(abs(v) for v in errors.values()) < 1e-8
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#687078', 'text.color': '#24292F', 'axes.labelcolor': '#24292F'})
    names = list(curves)
    focus = ['6-31G(d,p)', 'cc-pVDZ', 'cc-pVTZ', 'NNAO / MACE', 'NNAO / free contraction']
    nao = {r['basis']: r['nao'] for r in summaries}
    for relative, stem in ((False, 'ethane-cc-absolute'), (True, 'ethane-cc-relative')):
        fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.6))
        for ax, selected, title in zip(axes, (names, focus), ('All nine curves', 'Polarized bases and NNAO')):
            for name in selected:
                energy = curves[name]
                y = (energy - energy[0]) * EH_TO_KCAL if relative else energy
                color, linestyle, marker = STYLES[name]
                ax.plot(x, y, color=color, linestyle=linestyle, marker=marker, markevery=3,
                    markersize=3.7, linewidth=2.0 if name.startswith('NNAO') or name == 'cc-pVTZ' else 1.15,
                    label=f'{name} ({nao[name]} AO)')
            if rejected:
                ax.axvspan(min(rejected)-.025, max(rejected)+.025, facecolor='#E8EAED', alpha=.65, zorder=-2)
            ax.set_xlim(1.48, 8.08)
            ax.set_xlabel('C-C separation (Angstrom)')
            ax.set_ylabel(r'$E(R)-E(R_0)$ (kcal/mol)' if relative else 'SCF total energy (Eh)')
            ax.set_title(title, fontsize=11.5, pad=10)
            if not relative:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.set_axisbelow(True)
            ax.grid(color='#E5E8EA', linewidth=.65)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(.51, .91),
                   frameon=False, fontsize=9.1, columnspacing=1.8)
        fig.suptitle('Ethane C-C stretch: ' + ('relative energies' if relative else 'absolute energies'),
                     fontsize=15, y=.99)
        fig.text(.07, .035,
            '56 rigid geometries; restricted DF-RHF / def2-universal-jkfit; MACE update 3351, fixed exponents and core.\n'
            + ('Each curve uses its own E(R0), R0 = 1.525976 Angstrom. ' if relative else 'Original Hartree energies; no energy shifts. ')
            + ('All 56 free-contraction endpoints meet the gradient criterion.\n' if repaired else
               'Gray band: five unconverged free-contraction endpoints (kept as gaps).\n')
            + ('Multiple restricted-SCF seeds checked; ' if repaired else 'Local restricted-SCF solutions; ')
            + 'this pressure test is not the physical homolytic dissociation limit.',
            fontsize=8.2, color='#59636B')
        fig.subplots_adjust(left=.07, right=.985, bottom=.20, top=.72, wspace=.24)
        for ext in ('png', 'pdf'):
            fig.savefig(OUT / f'{stem}.{ext}', dpi=220, facecolor='white')
        plt.close(fig)
    provenance = dict(source_scan_sha256=digest(SOURCE / 'scan-summary.json'),
        geometry_sha256=digest(SOURCE / 'structures.json'), traditional_results_sha256=digest(OUT / 'traditional-results.json'),
        plot_source_sha256=digest(Path(__file__)), curve_count=len(curves), points_per_curve=56,
        rejected_free_distances_angstrom=rejected, baseline_cpu_minus_archived_gpu_hartree=errors,
        recomputed_nnao=repaired, conventional_backend='CPU PySCF 2.9.0',
        nnao_results_sha256=digest(nnao_path) if repaired else None,
        branch_policy=conventional['identity'].get('branch_policy', 'hcore-only'),
        requested_blas_thread_limit=4, local_pyscf_openmp_available=False,
        units=dict(distance='angstrom', absolute_energy='hartree', relative_energy='kcal/mol'),
        hartree_to_kcal_mol=EH_TO_KCAL, summary=summaries)
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(curves=len(curves), conventional_passed=conventional['passed'],
        nnao_mace_passed=56, nnao_free_accepted=int(np.isfinite(curves['NNAO / free contraction']).sum()),
        baseline_max_abs_error_hartree=max(abs(v) for v in errors.values()),
        summary=summaries), indent=2))


if __name__ == '__main__':
    main()
