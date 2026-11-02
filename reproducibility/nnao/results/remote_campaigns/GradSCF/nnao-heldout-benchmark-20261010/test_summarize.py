"""A numerical schema fixture checks partial-failure report semantics."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class PartialEvidenceReport(unittest.TestCase):
    def test_failed_energy_is_not_scored_and_frozen_gain_has_the_correct_sign(self):
        path = Path(__file__).with_name('summarize.py')
        if not path.is_file():
            self.fail('Held-out benchmark result collector is not implemented.')
        spec = importlib.util.spec_from_file_location('heldout_summary', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structures = [dict(structure_id=f'fixture/{i}', symbols=['H', 'H']) for i in (1, 2)]
            selection = root / 'structures.jsonl'
            selection.write_text(''.join(json.dumps(row) + '\n' for row in structures))
            native = dict(status='complete', auxbasis='def2-universal-jkfit', dtype='float64',
                          cart=False, structure_count=2, checkpoint_epoch=41, results=[])
            gpu4 = dict(status='failed', settings=dict(auxbasis='def2-universal-jkfit',
                                                     dtype='float64', cartesian=False), results=[])
            for structure, anchor in zip(structures, (-10., -20.)):
                identifier = structure['structure_id']
                for index, basis in enumerate(module.BASES):
                    energy = anchor + .01 * index
                    if basis == 'cc-pvtz': energy = anchor
                    if basis == 'NNAO-reference': energy = anchor + .1
                    if basis == 'NNAO-epoch41': energy = anchor + .05
                    failed = identifier == 'fixture/2' and basis == 'NNAO-epoch41'
                    gpu4['results'].append(dict(structure_id=identifier, basis=basis,
                        status='failed' if failed else 'passed', energy_hartree=energy,
                        nao=8, primitive_nao=24, setup_seconds=.1,
                        timings=[dict(index=i, scf_seconds=1. + i, scf_cycles=10) for i in range(2)]))
                    if basis.startswith('NNAO-'):
                        native['results'].append(dict(structure_id=identifier, basis=basis,
                            status='passed', energy_hartree=energy, natoms=2, nao=8, primitive_nao=24,
                            integral_generation_seconds=.5, basis_binding_and_to_cpu_seconds=.03,
                            host_device_transfer_seconds=.02,
                            nn_forward_timings=[dict(index=i, elapsed_seconds=.01) for i in range(3)],
                            scf_timings=[dict(index=i, elapsed_seconds=2. + i,
                                              scf=dict(scf_cycles=10)) for i in range(3)]))
            native_file = root / 'forward.json'; gpu_file = root / 'gpu4.json'
            native_file.write_text(json.dumps(native)); gpu_file.write_text(json.dumps(gpu4))
            output = root / 'report'
            result = module.main(['--native', str(native_file), '--gpu4', str(gpu_file),
                                  '--structures', str(selection), '--output-dir', str(output)])
            self.assertEqual(result, 0)
            with (output / 'comparison.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 28)
            good = next(row for row in rows if row['structure_id'] == 'fixture/1'
                        and row['basis'] == 'NNAO-epoch41')
            self.assertAlmostEqual(float(good['above_cc_pvtz_millihartree']), 50.)
            self.assertAlmostEqual(float(good['reference_to_epoch41_gain_millihartree']), 50.)
            bad = next(row for row in rows if row['structure_id'] == 'fixture/2'
                       and row['basis'] == 'NNAO-epoch41')
            self.assertEqual(bad['comparison_status'], 'failed')
            self.assertEqual(bad['energy_hartree'], '')
            self.assertEqual(bad['above_cc_pvtz_millihartree'], '')
            self.assertEqual(bad['reference_to_epoch41_gain_millihartree'], '')
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['passed_energy_cases'], 27)
            self.assertEqual(summary['paired_gain_structures'], 1)
            self.assertFalse(summary['selection_manifest_matches_structures'])
            for filename in ('README.md', 'energy-gap.png', 'energy-gap.pdf', 'timing.png', 'timing.pdf'):
                self.assertGreater((output / filename).stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
