"""Benchmark-only PyRHF/direct control: fresh density J/K, original optimizer.
Use the unchanged worker CLI: CONFIG NEW_OUTPUT --reference-continuation.
"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys

POLICY = 'fresh_density_keep_optimizer_v1'


def fresh_reference(run, mf, history, *, max_cycle, continuation=False, finite_state):
    if not mf.direct_scf:
        raise ValueError('Fresh-density control requires direct_scf=True; retain the optimizer')
    original, owned = mf.get_veff, 'get_veff' in vars(mf)

    def get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        return original(mol=mol, dm=dm, dm_last=0, vhf_last=0, hermi=hermi)

    mf.get_veff = get_veff
    try:
        info = run(mf, history, max_cycle=max_cycle, continuation=continuation, finite_state=finite_state)
        return dict(info, reference_jk_policy=POLICY)
    finally:
        if owned: mf.get_veff = original
        else: del mf.get_veff


def main(worker=None):
    if worker is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from benchmarks.probes.scf_100 import worker
    if any(arg in ('-h', '--help') for arg in sys.argv[1:]):
        return worker.main()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    args, _ = parser.parse_known_args()
    cfg = json.loads(args.config.read_text())
    if (cfg['engine'], cfg['method'], cfg['representation']) != ('pyscf', 'RHF', 'direct'):
        parser.error('This explicit control variant is only for PySCF RHF/direct')
    if args.output.exists(): parser.error('Use a new output directory; original attempts stay immutable')
    runtime_path = args.output.absolute()/'runtime.json'
    original, snapshot = worker.run_reference, os.environ.get('GRADSCF_SOURCE_SNAPSHOT')
    worker.run_reference = partial(fresh_reference, original)
    os.environ['GRADSCF_SOURCE_SNAPSHOT'] = POLICY
    try:
        return worker.main()
    finally:
        worker.run_reference = original
        if snapshot is None: os.environ.pop('GRADSCF_SOURCE_SNAPSHOT', None)
        else: os.environ['GRADSCF_SOURCE_SNAPSHOT'] = snapshot
        if runtime_path.exists():
            runtime = json.loads(runtime_path.read_text())
            runtime.update(reference_jk_policy=POLICY, adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            runtime_path.write_text(json.dumps(runtime, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
