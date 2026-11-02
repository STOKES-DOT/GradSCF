"""Audit numerical alignment, accepted-point masks and standalone exports."""
import csv
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'nnao-ethane-dissociation-20261019'


def main():
    data = json.loads((ROOT / 'traditional-results.json').read_text())
    rows = list(csv.DictReader((ROOT / 'curves.csv').open()))
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    assert data['status'] == 'complete' and data['passed'] == 392 and data['failed'] == 0
    assert len(rows) == 504 and len({(r['basis'], r['index']) for r in rows}) == 504
    assert sum(r['accepted'] == 'True' for r in rows) == 499
    names = list(dict.fromkeys(r['basis'] for r in rows))
    for name in names:
        group = sorted([r for r in rows if r['basis'] == name], key=lambda r: int(r['index']))
        assert len(group) == 56
        initial = float(group[0]['energy_hartree'])
        for row in group:
            if row['accepted'] == 'True':
                energy = float(row['energy_hartree'])
                assert float(row['plot_energy_hartree']) == energy
                np.testing.assert_allclose(float(row['relative_energy_kcal_mol']),
                    (energy - initial) * 627.5094740631, atol=1e-10, rtol=0)
            else:
                assert name == 'NNAO / free contraction'
                assert row['plot_energy_hartree'] == row['relative_energy_kcal_mol'] == ''
    rejected = sorted(float(r['distance_angstrom']) for r in rows if r['accepted'] != 'True')
    np.testing.assert_allclose(rejected, [2.70, 2.75, 2.80, 2.85, 2.90], atol=1e-12, rtol=0)
    assert hashlib.sha256((SOURCE / 'scan-summary.json').read_bytes()).hexdigest() == provenance['source_scan_sha256']
    assert (ROOT / 'structures.json').read_bytes() == (SOURCE / 'structures.json').read_bytes()
    assert all(r['converged'] and r['internal_stable'] and r['orbital_residual'] < 1e-7 for r in data['results'])
    exports = {}
    for stem in ('ethane-cc-relative', 'ethane-cc-absolute'):
        with Image.open(ROOT / f'{stem}.png') as image:
            assert image.width > 2000 and image.height > 1000
            exports[stem] = dict(png_size=list(image.size))
        with fitz.open(ROOT / f'{stem}.pdf') as document:
            assert len(document) == 1
            text = document[0].get_text()
            assert 'Ethane C-C stretch' in text and '144 AO' in text
            review = ROOT / 'review'
            review.mkdir(exist_ok=True)
            document[0].get_pixmap(matrix=fitz.Matrix(1.25, 1.25)).save(review / f'{stem}.png')
            exports[stem]['pdf_pages'] = len(document)
    report = dict(status='passed', conventional_passed=392, raw_rows=504, accepted_rows=499,
        nnao_mace_points=56, nnao_free_points=51, missing_free_distances_angstrom=rejected,
        maximum_conventional_residual=max(r['orbital_residual'] for r in data['results']),
        original_nnao_scan_unchanged=True, absolute_energies_unshifted=True,
        relative_energy_definition='E(R)-E(R0), independently for each curve', exports=exports)
    (ROOT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
