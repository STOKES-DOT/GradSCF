"""Narrow provenance guards and a numerical reporting fixture, without SCF."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


DIRECTORY = Path(__file__).resolve().parent
BN = DIRECTORY / 'job-95106/native/forward.json'
BG = DIRECTORY / 'job-95106/gpu4/comparison.json'


def module():
    path = DIRECTORY / 'summarize_pilot_best.py'
    if not path.is_file():
        raise AssertionError('Independent old-pilot comparison collector is not implemented.')
    spec = importlib.util.spec_from_file_location('pilot_report', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def reports(mod):
    """Synthetic pilot energies are 20 mHa below the actual 41-epoch baseline."""
    bn = json.loads(BN.read_text()); bg = json.loads(BG.read_text())
    new_native = copy.deepcopy(bn)
    new_native.update(checkpoint_profile='pilot-best975', checkpoint_epoch=975,
        model_training_structure_count=24, checkpoint_sha256=mod.PILOT_CHECKPOINT,
        passed=10, failed=0)
    new_native['results'] = [row for row in new_native['results'] if row['basis'] == mod.STAGE2]
    new_gpu = copy.deepcopy(bg)
    new_gpu.update(passed=10, failed=0, target_cases=10, completed_cases=10)
    new_gpu['results'] = [row for row in new_gpu['results'] if row['basis'] == mod.STAGE2]
    profile = dict(checkpoint_profile='pilot-best975', checkpoint_epoch=975,
        model_training_structure_count=24, checkpoint_sha256=mod.PILOT_CHECKPOINT,
        native_source_sha256=mod.SCIENCE_HASH)
    new_gpu['model_profiles'] = {mod.PILOT: profile}
    for report, timing_key in ((new_native, 'scf_timings'), (new_gpu, 'timings')):
        for row in report['results']:
            row['basis'] = mod.PILOT; row['energy_hartree'] -= .02
            for timing in row[timing_key]: timing['energy_hartree'] -= .02
            if report is new_gpu:
                row['model_profile'] = profile
                row['native_energy_hartree'] = row['energy_hartree']
    return bn, bg, new_native, new_gpu


class PilotReportGuards(unittest.TestCase):
    def test_reserialized_baseline_bytes_are_rejected_before_reporting(self):
        mod = module()
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / 'changed.json'
            changed.write_text(BN.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'baseline artifact bytes'):
                mod.main(['--native', str(BN), '--gpu4', str(BG),
                          '--baseline-native', str(changed), '--baseline-gpu4', str(BG),
                          '--output-dir', str(Path(temporary) / 'report')])

    def test_wrong_source_data_or_model_profile_is_rejected(self):
        mod = module(); bn, bg, n, g = reports(mod)
        for field, value in (('structures_sha256', 'different'),
                             ('checkpoint_epoch', 1000), ('model_training_structure_count', 200)):
            bad = copy.deepcopy(n); bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                mod.validate_reports(bn, bg, bad, g)
        bad = copy.deepcopy(n); bad['provenance']['repaired_scientific_source_sha256'] = 'old'
        with self.assertRaisesRegex(ValueError, 'source'):
            mod.validate_reports(bn, bg, bad, g)

    def test_same_nn_dimensions_and_passed_baseline_are_required(self):
        mod = module(); bn, bg, n, g = reports(mod)
        for field in ('nao', 'primitive_nao'):
            bad = copy.deepcopy(n); bad['results'][0][field] += 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'AO|primitive'):
                mod.validate_reports(bn, bg, bad, g)
        bad = copy.deepcopy(bg); bad['results'][0]['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'baseline'):
            mod.validate_reports(bn, bad, n, g)

    def test_failed_pilot_row_has_no_score_or_training_gain(self):
        mod = module(); bn, bg, n, g = reports(mod)
        g['results'][0]['status'] = 'failed'; g['failed'] = 1; g['passed'] = 9
        mod.validate_reports(bn, bg, n, g)
        structures = [json.loads(line) for line in (DIRECTORY / 'test-selection.jsonl').read_text().splitlines()]
        rows = mod.collect_reports(bn, bg, n, g, structures)
        pilot = [row for row in rows if row['basis'] == mod.PILOT]
        self.assertEqual(len(rows), 150)
        self.assertEqual(pilot[0]['comparison_status'], 'failed')
        self.assertIsNone(pilot[0]['energy_hartree'])
        self.assertIsNone(pilot[0]['energy_reduction_vs_nnao_reference_millihartree'])
        self.assertIsNone(pilot[0]['pilot975_reduction_vs_stage2epoch41_millihartree'])
        self.assertAlmostEqual(pilot[1]['pilot975_reduction_vs_stage2epoch41_millihartree'], 20.)

    def test_numerical_fixture_generates_separate_fifteen_basis_report(self):
        mod = module(); bn, bg, n, g = reports(mod)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); nf = root / 'forward.json'; gf = root / 'comparison.json'
            nf.write_text(json.dumps(n)); gf.write_text(json.dumps(g))
            out = root / 'report-pilot-best975'
            result = mod.main(['--native', str(nf), '--gpu4', str(gf),
                              '--baseline-native', str(BN), '--baseline-gpu4', str(BG),
                              '--output-dir', str(out)])
            self.assertEqual(result, 0)
            with (out / 'comparison.csv').open() as stream: rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 150)
            pilot = next(row for row in rows if row['basis'] == mod.PILOT)
            self.assertEqual(pilot['model_training_structure_count'], '24')
            self.assertEqual(pilot['checkpoint_epoch'], '975')
            self.assertAlmostEqual(float(pilot['pilot975_reduction_vs_stage2epoch41_millihartree']), 20.)
            for name in ('README.md', 'summary.json', 'energy-gap.png', 'energy-gap.pdf', 'timing.png', 'timing.pdf'):
                self.assertGreater((out / name).stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
