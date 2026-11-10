"""Benchmark-only six-method PySCF/direct control: fresh density J/K, original optimizer.
Use the unchanged worker CLI: CONFIG NEW_OUTPUT --reference-continuation.
"""
import argparse
from functools import partial, wraps
import inspect
import hashlib
import json
import os
from pathlib import Path
import sys

POLICY = 'fresh_density_keep_optimizer_v1'
METHODS = ('RHF', 'UHF', 'GHF', 'RKS', 'UKS', 'GKS')


def fresh_reference(run, mf, history, *, max_cycle, continuation=False, finite_state, method=None, metadata=None):
    if not mf.direct_scf:
        raise ValueError('Fresh-density control requires direct_scf=True; retain the optimizer')
    original, owned = mf.get_veff, 'get_veff' in vars(mf)

    signature = inspect.signature(original)
    details = dict(reference_jk_policy=POLICY, reference_adapter_scope_version=2,
        reference_jk_method=method or type(mf).__name__,
        reference_get_veff_defaults={name:p.default for name,p in signature.parameters.items()
                                    if p.default is not inspect.Parameter.empty})
    if metadata is not None: metadata.update(details)

    @wraps(original)
    def get_veff(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        bound.arguments.update(dm_last=0, vhf_last=0)
        return original(*bound.args, **bound.kwargs)

    mf.get_veff = get_veff
    try:
        info = run(mf, history, max_cycle=max_cycle, continuation=continuation, finite_state=finite_state)
        return dict(info, **details)
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
    if cfg['engine'] != 'pyscf' or cfg['method'] not in METHODS or cfg['representation'] != 'direct':
        parser.error('This explicit control requires PySCF RHF/UHF/GHF/RKS/UKS/GKS direct')
    if args.output.exists(): parser.error('Use a new output directory; original attempts stay immutable')
    runtime_path = args.output.absolute()/'runtime.json'
    original, snapshot = worker.run_reference, os.environ.get('GRADSCF_SOURCE_SNAPSHOT')
    details = {}
    worker.run_reference = partial(fresh_reference, original, method=cfg['method'], metadata=details)
    os.environ['GRADSCF_SOURCE_SNAPSHOT'] = POLICY+'.six_methods_v2'
    try:
        return worker.main()
    finally:
        worker.run_reference = original
        if snapshot is None: os.environ.pop('GRADSCF_SOURCE_SNAPSHOT', None)
        else: os.environ['GRADSCF_SOURCE_SNAPSHOT'] = snapshot
        if runtime_path.exists():
            runtime = json.loads(runtime_path.read_text())
            runtime.update(details, reference_jk_policy=POLICY, reference_jk_method=cfg['method'],
                           adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            runtime_path.write_text(json.dumps(runtime, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
