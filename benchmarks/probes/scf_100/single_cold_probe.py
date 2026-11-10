"""One clean cold call for a single-factor diagnostic, not the final repeat gate.

Use the unchanged worker's CONFIG OUTPUT arguments. Validate the registered
configuration before limiting only its repetition count. The physical inputs,
SCF gates and per-call budget are unchanged; the original requested count and
this timing-only overlay remain explicit in the saved input and probe receipt.
"""
import hashlib
import json
from pathlib import Path
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from benchmarks.probes.scf_100 import worker

    output = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
    if output is not None and (output.exists() or output.is_symlink()):
        raise FileExistsError('Use a new probe output; original records stay immutable')
    original = worker.validate_config
    details = {}

    def validate(cfg):
        original(cfg)
        details['registered_repeats'] = cfg['repeats']
        cfg['registered_repeats'] = cfg['repeats']
        cfg['repeats'] = 1
        cfg['execution_overlay'] = 'single_cold_probe; final repeat acceptance still requires registered count'

    worker.validate_config = validate
    try:
        return worker.main()
    finally:
        worker.validate_config = original
        if output is not None and output.is_dir():
            receipt = dict(scope='single cold diagnostic only', production_kernel_instrumented=False,
                effective_repeats_requested=1, final_repeat_acceptance=False, **details,
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                worker_sha256=hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest())
            path = output/'single_cold_probe.json'
            with path.open('x') as stream:
                stream.write(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
