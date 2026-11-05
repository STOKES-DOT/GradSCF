"""Metadata-only profiling contracts; run with python -S, without JAX/engines."""
import ast
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

PROFILE = Path(__file__).resolve().parents[1]/'scf_solver/profile_stages.py'
spec = importlib.util.spec_from_file_location('scf_stage_contract', PROFILE)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def required(name):
    function = getattr(profile, name, None)
    assert callable(function), f'Profiling metadata contract {name} is not implemented'
    return function


def configuration(method='RKS', representation='df'):
    return dict(method=method, representation=representation, xc='pbe',
        max_cycle=200, conv_tol=1e-10, density_tol=1e-8, gradient_tol=1e-7,
        max_memory=180000., threads=64, auxbasis='weigend', init_guess='atom',
        df_tol=1e-12, direct_scf_tol=1e-12)


class Facade:
    def __init__(self):
        self.df_tol, self.direct_scf_tol, self.direct_scf_workers = 1e-10, 0., 1
        self.jk_backend, self.only_dfj = 'full', False

    def density_fit(self, auxbasis):
        self.auxbasis, self.jk_backend = auxbasis, 'df'
        return self

    def direct_scf(self):
        self.jk_backend = 'direct'
        return self

    def _config(self):
        return SimpleNamespace(df_tol=self.df_tol, direct_scf_tol=self.direct_scf_tol,
            direct_scf_workers=self.direct_scf_workers, only_dfj=self.only_dfj,
            jk_backend=self.jk_backend)


class ProfileContract(unittest.TestCase):
    def test_no_override_preserves_historical_grid_defaults(self):
        for method, expected in [('GKS', -1), ('GHF', -1), ('RKS', 4096), ('UKS', 4096)]:
            self.assertEqual(required('facade_options')(configuration(method))['grid_block_size'], expected)

    def test_explicit_auto_and_fixed_five_are_effective_input_only(self):
        original = configuration()
        self.assertIn('grid_block_size', inspect.signature(profile.effective_config).parameters)
        cfg = profile.effective_config(original, fixed_cycles=5, grid_block_size=0)
        self.assertEqual((cfg['max_cycle'], cfg['conv_tol'], cfg['grid_block_size']), (5, 0., 0))
        options = required('facade_options')(cfg, grid_block_size=0)
        self.assertEqual((options['max_cycle'], options['conv_tol'], options['grid_block_size']), (5, 0., 0))
        self.assertEqual((original['max_cycle'], original['conv_tol']), (200, 1e-10))
        self.assertNotIn('grid_block_size', original)
        tree = ast.parse(PROFILE.read_text())
        gates = [kw.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                 for kw in node.keywords if kw.arg == 'scientific_pass']
        self.assertTrue(any('args.fixed_cycles is None' in ast.unparse(gate) for gate in gates))

    def test_ri_forwards_metric_threshold_with_legacy_only_dfj_default(self):
        cfg, mf = configuration(), Facade()
        self.assertIs(required('configure_facade')(mf, cfg, 'df'), mf)
        self.assertEqual(mf._config().df_tol, 1e-12)
        self.assertEqual(mf._config().direct_scf_tol, 1e-12)
        self.assertFalse(mf._config().only_dfj)

    def test_old_config_without_new_thresholds_retains_facade_defaults(self):
        cfg, mf = configuration(), Facade()
        del cfg['df_tol'], cfg['direct_scf_tol']
        required('configure_facade')(mf, cfg, 'df')
        self.assertEqual(mf._config().df_tol, 1e-10)
        self.assertEqual(mf._config().direct_scf_tol, 0.)

    def test_metric_threshold_ignored_by_facade_is_rejected(self):
        class Broken(Facade):
            def _config(self):
                result = super()._config()
                result.df_tol = 1e-10
                return result
        with self.assertRaises(NotImplementedError):
            required('configure_facade')(Broken(), configuration(), 'df')

    def test_missing_explicit_only_dfj_capability_is_rejected(self):
        cfg = dict(configuration(), only_dfj=True)
        with self.assertRaises(NotImplementedError):
            required('configure_facade')(Facade(), cfg, 'df')

    def test_direct_uses_registered_screening_threshold(self):
        mf = Facade()
        required('configure_facade')(mf, configuration(representation='direct'), 'direct')
        self.assertEqual((mf._config().direct_scf_tol, mf._config().direct_scf_workers), (1e-12, 64))

    def test_rks_eager_atomic_history_is_explicitly_skipped(self):
        shared = SimpleNamespace(state=SimpleNamespace(fock=SimpleNamespace(shape=(2, 3, 3))), history=object())
        status = required('history_capture_status')('RKS', 880, shared,
            {'ancestor_stages': ['inputs.build_rks_integral_inputs', 'solver.run_uks_from_integrals']})
        self.assertEqual(status['status'], 'skipped')
        self.assertIn('RKS', status['reason'])
        self.assertIn('atomic', status['reason'])

    def test_unrestricted_history_requires_molecular_shape_and_origin(self):
        good = SimpleNamespace(state=SimpleNamespace(fock=SimpleNamespace(shape=(2, 7, 7))), history=object())
        metadata = {'ancestor_stages': ['solver.run_uks_from_integrals']}
        self.assertEqual(required('history_capture_status')('UKS', 7, good, metadata)['status'], 'ok')
        self.assertEqual(required('history_capture_status')('UKS', 880, good, metadata)['status'], 'skipped')
        atomic = {'ancestor_stages': ['inputs.build_uks_integral_inputs', 'solver.run_uks_from_integrals']}
        self.assertEqual(required('history_capture_status')('UKS', 7, good, atomic)['status'], 'skipped')
        self.assertEqual(required('history_capture_status')('UKS', 7, good, {})['status'], 'skipped')

    def test_generalized_history_uses_spinor_fock_shape(self):
        good = SimpleNamespace(state=SimpleNamespace(fock=SimpleNamespace(shape=(14, 14))), history=object())
        self.assertEqual(required('history_capture_status')('GHF', 7, good,
            {'ancestor_stages': ['solver.run_gks_from_integrals']})['status'], 'ok')
        self.assertEqual(required('history_capture_status')('GKS', 8, good,
            {'ancestor_stages': ['solver.run_gks_from_integrals']})['status'], 'skipped')


if __name__ == '__main__':
    unittest.main()
