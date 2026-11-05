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


if __name__ == '__main__':
    unittest.main()
