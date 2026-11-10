"""Metadata-only execution-order overlay tests; python -S, no scheduler or engines."""
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
PLAN = Path('reports/scf_100/20261105-40a569bf/plan-v2')
SCRIPT = Path(__file__).with_name('continue_matrix.py')


def fingerprints(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class MatrixOrder(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/'campaign'
        self.plan = self.root/'baseline'/PLAN
        shutil.copytree(ROOT/PLAN, self.plan)
        self.order = json.loads((self.plan/'run_plan.json').read_text())['run_order']
        self.cfg = {case: json.loads((self.plan/'configs'/(case+'.json')).read_text()) for case in self.order}
        for directory in ('baseline/src', 'candidate-1a575d18/src'):
            (self.root/directory).mkdir(parents=True, exist_ok=True)
        for path in ('harness-reference-v1/benchmarks/probes/scf_100/worker.py',
                     'harness-reference-v1/benchmarks/probes/response_grid3/guard.py'):
            self.record(self.root/path, '# metadata fixture; never executed')
        self.record(self.root.parent/'gradscf-libcint-nnao-20261026/env-compare/bin/python', 'never executed')

    def record(self, path, text='{}'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def dry_run(self, direct_first=False, extra=()):
        args = [sys.executable, '-S', str(SCRIPT), '--root', str(self.root), '--job', 'metadata', '--dry-run']
        result = subprocess.run(args+(['--direct-first'] if direct_first else [])+list(extra),
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines()]

    def test_direct_first_is_stable_and_preserves_all_sixty_hashes(self):
        source_before, copied_before = fingerprints(ROOT/PLAN), fingerprints(self.plan)
        original = [r['case_id'] for r in self.dry_run()]
        reordered = [r['case_id'] for r in self.dry_run(True)]
        direct = [case for case in self.order if self.cfg[case]['representation']=='direct']
        ri = [case for case in self.order if self.cfg[case]['representation']=='df']
        self.assertEqual((len(direct), len(ri), len(reordered)), (30, 30, 60))
        self.assertEqual(original, self.order)
        self.assertEqual(reordered, direct+ri)
        self.assertEqual(set(original), set(reordered))
        manifest = json.loads((self.plan/'input_manifest.json').read_text())['config_file_sha256']
        self.assertEqual(len(manifest), 60)
        self.assertEqual({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (self.plan/'configs').glob('*.json')}, manifest)
        self.assertEqual(fingerprints(ROOT/PLAN), source_before)
        self.assertEqual(fingerprints(self.plan), copied_before)

    def test_existing_attempts_and_registered_caps_survive_both_orders(self):
        skipped = set(self.order[:3])
        output = self.root/'results/matrix-v1'
        self.record(output/(self.order[0]+'.process.json'), '{"status":"error"}')
        self.record(output/(self.order[1]+'.log'), 'orphan attempt retained')
        (output/self.order[2]).mkdir(parents=True)
        for case, cfg in self.cfg.items():
            if (cfg['method'], cfg['xc']) != ('RKS', 'pbe'):
                continue
            tag = 'candidate-grid-cache-v3' if cfg['engine']=='gradscf' else 'reference-certified-v1'
            path = (self.root/'results'/tag/(case+'.process.json') if cfg['representation']=='df'
                    else self.root/'results/no-ri-pair'/(tag+'-'+cfg['engine']+'.process.json'))
            self.record(path, '{"status":"ok"}')
            skipped.add(case)
        before = fingerprints(self.root)
        for enabled in (False, True):
            rows = self.dry_run(enabled)
            self.assertEqual({r['case_id'] for r in rows if r['status']=='skip_previous_attempt'}, skipped)
            for row in rows:
                self.assertEqual(row['cap_seconds'], self.cfg[row['case_id']]['process_limit_seconds'])
                self.assertEqual(row['status'], 'skip_previous_attempt' if row['case_id'] in skipped else 'would_run')
                if row['case_id'] in skipped:
                    self.assertTrue(Path(row['prior_record']).exists())
        self.assertEqual(fingerprints(self.root), before)


    def test_optimized_order_selects_thirty_grad_rows_with_stable_six_first(self):
        source = self.root/'candidate-optimized-v1'
        self.record(source/'src/gradscf/solver.py', '# new source fixture')
        rows = self.dry_run(extra=('--source-tree', source.name, '--output-prefix',
                                   'optimized-v1', '--optimized-order'))
        grad = [case for case in self.order if self.cfg[case]['engine']=='gradscf']
        keys = (('RHF', 'hf'), ('RKS', 'pbe'), ('UHF', 'hf'),
                ('GHF', 'hf'), ('UKS', 'pbe'), ('GKS', 'pbe'))
        first = [next(case for case in grad if self.cfg[case]['representation']=='direct'
                      and (self.cfg[case]['method'], self.cfg[case]['xc'])==key) for key in keys]
        self.assertEqual(len(rows), 30)
        self.assertEqual([row['case_id'] for row in rows], first+[case for case in grad if case not in first])
        for row in rows:
            self.assertEqual(row['source_tree'], str(source))
            self.assertEqual(row['output_prefix'], 'optimized-v1')
            self.assertEqual(row['cap_seconds'], self.cfg[row['case_id']]['process_limit_seconds'])

    def test_optimized_dry_run_isolates_prefix_without_writing_or_changing_configs(self):
        source = self.root/'candidate-optimized-v1'
        self.record(source/'src/gradscf/solver.py', '# new source fixture')
        grad = [case for case in self.order if self.cfg[case]['engine']=='gradscf']
        for case in grad:
            self.record(self.root/'results/matrix-v1'/(case+'.process.json'))
            self.record(self.root/'results/candidate-grid-cache-v3'/(case+'.process.json'))
        self.record(self.root/'results/no-ri-pair/candidate-grid-cache-v3-gradscf.process.json')
        own = self.root/'results/optimized-v1'/(grad[-1]+'.log')
        self.record(own, 'only this prefix counts')
        before = fingerprints(self.root)
        paths_before = {str(p.relative_to(self.root)) for p in self.root.rglob('*')}
        source_plan_before = fingerprints(ROOT/PLAN)
        rows = self.dry_run(extra=('--source-tree', str(source), '--output-prefix',
                                   'optimized-v1', '--optimized-order'))
        self.assertEqual({r['case_id'] for r in rows if r['status']=='skip_previous_attempt'}, {grad[-1]})
        self.assertEqual({r['prior_record'] for r in rows if r['prior_record']}, {str(own)})
        self.assertTrue(all(r['status']=='would_run' for r in rows if r['case_id']!=grad[-1]))
        manifest = json.loads((self.plan/'input_manifest.json').read_text())['config_file_sha256']
        actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.plan/'configs').glob('*.json')}
        self.assertEqual(len(actual), 60)
        self.assertEqual(actual, manifest)
        self.assertEqual(fingerprints(self.root), before)
        self.assertEqual({str(p.relative_to(self.root)) for p in self.root.rglob('*')}, paths_before)
        self.assertEqual(fingerprints(ROOT/PLAN), source_plan_before)

    def test_source_manifest_hashes_python_and_native_shared_library_changes(self):
        source = self.root/'candidate-optimized-v1'
        python = source/'src/gradscf/solver.py'
        native = source/'src/gradscf/_native/libintegrals.so'
        self.record(python, '# source version one')
        self.record(native, 'native fixture version one')
        source_manifest = runpy.run_path(str(SCRIPT))['source_manifest']
        first = source_manifest({'gradscf': source})
        files = first['sources']['gradscf']['files_sha256']
        self.assertEqual(set(files), {'src/gradscf/solver.py', 'src/gradscf/_native/libintegrals.so'})
        self.assertEqual(files['src/gradscf/_native/libintegrals.so'], hashlib.sha256(native.read_bytes()).hexdigest())
        self.assertEqual(source_manifest({'gradscf': source}), first)
        self.record(python, '# source version two')
        second = source_manifest({'gradscf': source})
        self.assertNotEqual(first, second)
        self.assertNotEqual(first['sources']['gradscf']['tree_sha256'], second['sources']['gradscf']['tree_sha256'])
        self.record(native, 'native fixture version two')
        third = source_manifest({'gradscf': source})
        self.assertNotEqual(second, third)
        self.assertNotEqual(second['sources']['gradscf']['tree_sha256'], third['sources']['gradscf']['tree_sha256'])


if __name__ == '__main__':
    unittest.main()
