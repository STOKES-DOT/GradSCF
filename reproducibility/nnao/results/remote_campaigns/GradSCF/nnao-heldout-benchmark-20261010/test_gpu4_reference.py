"""Input and evidence guards without importing CuPy or initializing CUDA."""
import importlib.util
from pathlib import Path
import unittest


def script():
    path = Path(__file__).with_name('gpu4_reference.py')
    if not path.is_file():
        raise AssertionError('Independent GPU4PySCF benchmark is not implemented.')
    spec = importlib.util.spec_from_file_location('gpu4_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def structure():
    return dict(structure_id='case/1', symbols=['H', 'H'],
                coords_angstrom=[[0., 0., 0.], [0., 0., .74]], charge=0, spin=0)


def frozen():
    return dict(structure_id='case/1', basis='NNAO-epoch41',
                atom_labels=['H0', 'H1'],
                atom_shells=[[[0, [1., .5]]], [[0, [1., .7]]]],
                native_energy_hartree=-1.1)


class ReferenceGuards(unittest.TestCase):
    def test_pilot975_profile_is_distinct_and_does_not_change_default_bases(self):
        module = script()
        self.assertIn('NNAO-pilot-best975', module.CUSTOM_BASES)
        self.assertEqual(len(module.DEFAULT_BASES), 14)
        self.assertNotIn('NNAO-pilot-best975', module.DEFAULT_BASES)
        row = frozen(); row.update(basis='NNAO-pilot-best975', checkpoint_profile='pilot-best975',
            checkpoint_epoch=975, model_training_structure_count=24,
            checkpoint_sha256='pilot-weight-hash', native_source_sha256='current-science')
        profiles = module.frozen_profiles([row])
        self.assertEqual(profiles['NNAO-pilot-best975']['checkpoint_epoch'], 975)
        self.assertEqual(profiles['NNAO-pilot-best975']['model_training_structure_count'], 24)
        self.assertNotIn('epoch41', module.profile_note(profiles))

    def test_inconsistent_frozen_profile_cannot_be_silently_combined(self):
        module = script()
        first = frozen(); first.update(checkpoint_epoch=41, checkpoint_sha256='first')
        second = dict(first, structure_id='case/2', checkpoint_sha256='changed')
        self.assertTrue(hasattr(module, 'frozen_profiles'), 'Frozen provenance is not yet classified.')
        with self.assertRaisesRegex(ValueError, 'profile|provenance'):
            module.frozen_profiles([first, second])

    def test_missing_native_basis_fails_only_custom_case_and_standard_case_continues(self):
        module = script()
        self.assertTrue(hasattr(module, 'run_case'), 'Missing frozen custom cases still abort the full benchmark.')
        computed = []
        def operation():
            computed.append('standard')
            return dict(status='passed', energy_hartree=-1.1)
        missing = module.run_case(structure(), 'NNAO-epoch41', None, operation)
        self.assertEqual(missing['status'], 'failed')
        self.assertEqual(missing['error_type'], 'MissingNativeBasis')
        self.assertNotIn('energy_hartree', missing)
        self.assertEqual(computed, [])
        standard = module.run_case(structure(), 'cc-pvtz', None, operation)
        self.assertEqual(standard['status'], 'passed')
        self.assertEqual(computed, ['standard'])

    def test_present_frozen_geometry_is_validated_independently(self):
        module = script()
        bad = frozen(); bad['symbols'] = ['C', 'H']
        with self.assertRaisesRegex(ValueError, 'symbols'):
            module.frozen_basis(structure(), bad)
        bad = frozen(); bad['coords_angstrom'] = [[0., 0., 0.], [0., 0., .75]]
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            module.frozen_basis(structure(), bad)

    def test_distinct_atom_contractions_keep_distinct_labels(self):
        module = script()
        labels, basis = module.frozen_basis(structure(), frozen())
        self.assertEqual(labels, ['H0', 'H1'])
        self.assertEqual(basis['H0'][0][1][1], .5)
        self.assertEqual(basis['H1'][0][1][1], .7)
        duplicate = frozen(); duplicate['atom_labels'] = ['H', 'H']
        with self.assertRaisesRegex(ValueError, 'unique'):
            module.frozen_basis(structure(), duplicate)

    def test_wrong_atom_element_and_empty_contraction_are_rejected(self):
        module = script()
        bad = frozen(); bad['atom_labels'][0] = 'C0'
        with self.assertRaisesRegex(ValueError, 'element'):
            module.frozen_basis(structure(), bad)
        bad = frozen(); bad['atom_labels'][0] = '0H'
        with self.assertRaisesRegex(ValueError, 'element'):
            module.frozen_basis(structure(), bad)
        bad = frozen(); bad['atom_shells'][0][0].append([.2])
        with self.assertRaisesRegex(ValueError, 'primitive'):
            module.frozen_basis(structure(), bad)
        bad = frozen(); bad['atom_shells'][0][0][1][1] = 0.
        with self.assertRaisesRegex(ValueError, 'zero'):
            module.frozen_basis(structure(), bad)

    def test_only_current_passed_evidence_can_be_reused(self):
        module = script()
        saved = dict(status='passed', signature='current', timings=[{}, {}])
        self.assertTrue(module.reusable_case(saved, 'current'))
        for bad in (dict(saved, signature='old'), dict(saved, status='failed'),
                    dict(saved, timings=[{}])):
            self.assertFalse(module.reusable_case(bad, 'current'))

    def test_native_comparison_is_strict_and_finite(self):
        module = script()
        self.assertAlmostEqual(module.native_energy_error(-1.1, frozen()), 0.)
        self.assertAlmostEqual(module.native_energy_error(-1.1 + 1e-8, frozen()), 1e-8)
        for energy in (-1.1 + 2e-7, float('nan')):
            with self.assertRaisesRegex(ValueError, 'energy'):
                module.native_energy_error(energy, frozen())

    def test_finite_basis_anchor_covers_only_matching_structure(self):
        module = script()
        rows = [dict(structure_id='a', basis='cc-pvtz', status='passed', energy_hartree=-10.),
                dict(structure_id='a', basis='NNAO-epoch41', status='passed', energy_hartree=-9.99),
                dict(structure_id='b', basis='sto-3g', status='passed', energy_hartree=-1.),
                dict(structure_id='a', basis='bad', status='failed')]
        module.add_reference_differences(rows)
        self.assertAlmostEqual(rows[1]['above_cc_pvtz_millihartree'], 10.)
        self.assertNotIn('above_cc_pvtz_millihartree', rows[2])
        self.assertNotIn('above_cc_pvtz_millihartree', rows[3])


if __name__ == '__main__':
    unittest.main()
