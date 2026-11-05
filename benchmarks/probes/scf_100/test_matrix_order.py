"""Metadata-only execution-order overlay tests; python -S, no scheduler or engines."""
import hashlib
import json
from pathlib import Path
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

    def dry_run(self, direct_first=False):
        args = [sys.executable, '-S', str(SCRIPT), '--root', str(self.root), '--job', 'metadata', '--dry-run']
        result = subprocess.run(args+(['--direct-first'] if direct_first else []),
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


if __name__ == '__main__':
    unittest.main()
