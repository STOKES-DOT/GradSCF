"""Metadata-only control-variant tests; no PySCF/JAX or numerical work."""
import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ADAPTER = Path(__file__).with_name('reference_fresh_jk.py')


def adapter():
    assert ADAPTER.exists(), 'Fresh-density benchmark adapter is missing'
    spec = importlib.util.spec_from_file_location('fresh_jk_contract', ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMF:
    direct_scf = True
    def __init__(self):
        self._opt, self.calls = object(), []
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        self.calls.append((mol, dm, dm_last, vhf_last, hermi))
        return 'original-result'


class FreshJKContract(unittest.TestCase):
    def test_only_incremental_arguments_change_and_worker_options_survive(self):
        module, mf = adapter(), FakeMF()
        original, optimizer = mf.get_veff, mf._opt
        molecule, density, history, finite = object(), object(), [], lambda current: True
        def run(current, checks, *, max_cycle, continuation, finite_state):
            self.assertIs(current, mf)
            self.assertIs(checks, history)
            self.assertEqual((max_cycle, continuation), (200, True))
            self.assertIs(finite_state, finite)
            self.assertEqual(list(inspect.signature(current.get_veff).parameters),
                ['mol', 'dm', 'dm_last', 'vhf_last', 'hermi'])
            self.assertEqual(current.get_veff(molecule, density, 'old-D', 'old-V', 2), 'original-result')
            self.assertTrue(current.direct_scf)
            self.assertIs(current._opt, optimizer)
            return {'reference_accepted': True}
        info = module.fresh_reference(run, mf, history, max_cycle=200, continuation=True, finite_state=finite)
        self.assertEqual(mf.calls, [(molecule, density, 0, 0, 2)])
        self.assertEqual(mf.get_veff, original)
        self.assertNotIn('get_veff', vars(mf))
        self.assertEqual(info['reference_jk_policy'], 'fresh_density_keep_optimizer_v1')

    def test_error_restores_class_and_owned_bound_methods(self):
        module = adapter()
        for owned in (False, True):
            with self.subTest(owned=owned):
                mf = FakeMF()
                if owned: mf.get_veff = mf.get_veff
                original, optimizer = mf.get_veff, mf._opt
                def broken(current, *args, **kwargs):
                    current.get_veff(dm='current', dm_last='old', vhf_last='old')
                    raise RuntimeError('worker failed')
                with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                    module.fresh_reference(broken, mf, [], max_cycle=200, finite_state=lambda current: True)
                self.assertEqual(mf.get_veff, original)
                self.assertEqual('get_veff' in vars(mf), owned)
                self.assertTrue(mf.direct_scf)
                self.assertIs(mf._opt, optimizer)
        mf = FakeMF()
        mf.direct_scf = False
        with self.assertRaises(ValueError):
            module.fresh_reference(lambda *a, **k: self.fail('Optimizer guard was bypassed'), mf, [], max_cycle=200, finite_state=lambda current: True)

    def test_runtime_adds_adapter_metadata_without_relabelling_worker_or_old_outputs(self):
        module = adapter()
        with tempfile.TemporaryDirectory() as temporary:
            cfg, output = Path(temporary)/'config.json', Path(temporary)/'new-output'
            cfg.write_text(json.dumps(dict(engine='pyscf', method='RHF', representation='direct')))
            results = []
            def run(*args, **kwargs): return {'reference_accepted': True}
            worker = SimpleNamespace(run_reference=run)
            def main():
                output.mkdir()
                results.append(worker.run_reference(FakeMF(), [], max_cycle=200, continuation=True, finite_state=lambda mf: True))
                (output/'runtime.json').write_text(json.dumps(dict(probe_sha256='original-worker-5a', source_snapshot=os.environ['GRADSCF_SOURCE_SNAPSHOT'])))
                return 0
            worker.main = main
            argv = ['reference_fresh_jk.py', str(cfg), str(output), '--reference-continuation']
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, {'GRADSCF_SOURCE_SNAPSHOT':'parent'}):
                self.assertEqual(module.main(worker), 0)
                self.assertEqual(os.environ['GRADSCF_SOURCE_SNAPSHOT'], 'parent')
            self.assertIs(worker.run_reference, run)
            runtime = json.loads((output/'runtime.json').read_text())
            self.assertEqual(runtime['probe_sha256'], 'original-worker-5a')
            self.assertEqual(runtime['reference_jk_policy'], module.POLICY)
            self.assertIn(module.POLICY, runtime['source_snapshot'])
            self.assertEqual(len(runtime['adapter_sha256']), 64)
            self.assertEqual(results[0]['reference_jk_policy'], module.POLICY)
            before = (output/'runtime.json').read_bytes()
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit): module.main(worker)
            self.assertEqual((output/'runtime.json').read_bytes(), before)


    def test_original_hermi_default_and_other_arguments_survive_all_call_styles(self):
        module = adapter()
        for default in (0, 1):
            for style in ('omitted', 'positional', 'keyword'):
                with self.subTest(default=default, style=style):
                    class MF(FakeMF):
                        def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=default, *, marker='untouched'):
                            self.calls.append((mol, dm, dm_last, vhf_last, hermi, marker))
                            return 'original-result'
                    mf = MF()
                    def run(current, *args, **kwargs):
                        self.assertEqual(inspect.signature(current.get_veff).parameters['hermi'].default, default)
                        if style == 'omitted': current.get_veff(dm='D', dm_last='old', vhf_last='old')
                        elif style == 'positional': current.get_veff('M', 'D', 'old', 'old', 2, marker='custom')
                        else: current.get_veff(mol='M', dm='D', dm_last='old', vhf_last='old', hermi=2, marker='custom')
                        return {}
                    info = module.fresh_reference(run, mf, [], max_cycle=200, finite_state=lambda current: True)
                    expected = (None, 'D', 0, 0, default, 'untouched') if style == 'omitted' else ('M', 'D', 0, 0, 2, 'custom')
                    self.assertEqual(mf.calls, [expected])
                    self.assertEqual(info['reference_get_veff_defaults']['hermi'], default)
                    self.assertEqual(info['reference_get_veff_defaults']['marker'], 'untouched')

    def test_six_direct_method_parsers_record_actual_method_and_defaults(self):
        module = adapter()
        for method in ('RHF', 'UHF', 'GHF', 'RKS', 'UKS', 'GKS'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                config, output = Path(temporary)/'config.json', Path(temporary)/'output'
                config.write_text(json.dumps(dict(engine='pyscf', method=method, representation='direct')))
                worker = SimpleNamespace(run_reference=lambda *args, **kwargs: {'reference_accepted':True})
                results = []
                def main():
                    output.mkdir()
                    results.append(worker.run_reference(FakeMF(), [], max_cycle=200, finite_state=lambda current: True))
                    (output/'runtime.json').write_text(json.dumps(dict(probe_sha256='worker-original')))
                    return 0
                worker.main = main
                with patch.object(sys, 'argv', ['adapter', str(config), str(output)]):
                    self.assertEqual(module.main(worker), 0)
                runtime = json.loads((output/'runtime.json').read_text())
                self.assertEqual(results[0]['reference_jk_method'], method)
                self.assertEqual(runtime['reference_jk_method'], method)
                self.assertEqual(runtime['reference_get_veff_defaults']['hermi'], 1)
                self.assertEqual(runtime['probe_sha256'], 'worker-original')

    def test_non_direct_or_unknown_method_is_rejected_without_worker_execution(self):
        module = adapter()
        for engine, method, representation in [('gradscf','RHF','direct'),('pyscf','ROHF','direct'),('pyscf','UKS','df')]:
            with self.subTest(engine=engine, method=method, representation=representation), tempfile.TemporaryDirectory() as temporary:
                config, output = Path(temporary)/'config.json', Path(temporary)/'output'
                config.write_text(json.dumps(dict(engine=engine, method=method, representation=representation)))
                worker = SimpleNamespace(run_reference=object(), main=lambda: self.fail('Invalid variant reached worker'))
                with patch.object(sys, 'argv', ['adapter', str(config), str(output)]), self.assertRaises(SystemExit):
                    module.main(worker)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
