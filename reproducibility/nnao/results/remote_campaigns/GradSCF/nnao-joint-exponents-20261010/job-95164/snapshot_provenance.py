"""Audit the isolated exponent experiment without altering the original checkout."""
import argparse
import hashlib
import json
from pathlib import Path


BASELINE = '87f14784cbc038349f65f311c4b664809925932abd2f0cccea834c5400102b30'
NATIVE = 'b0e65d075c537a3e56e499d7e8df85b8b493ed0e71b6f86ab82cda196a426fd4'
CHECKPOINT = 'ba93fb40efca3268962c7501f82627e188ed9509b6a73bd27a2df2170a3495de'
CHANGED = {'src/gradscf/model/nnao/basis.py',
           'src/gradscf/integrals/density_fitting.py',
           'tools/optimize_methane_nnao.py'}
ADDED = {'tools/train_nnao_joint.py'}
PILOT_IDS = {'W4-11/ch4', 'W4-11/nh3', 'W4-11/h2o', 'W4-11/hf'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    package = root / 'src/gradscf'
    return {str(p.relative_to(root)): sha(p) for p in sorted(package.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.json', '.dat'}}


def scientific_hash(files):
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        if name.startswith('src/gradscf/') and Path(name).suffix in {'.py', '.json', '.dat'}:
            digest.update(name.removeprefix('src/gradscf/').encode())
            digest.update(value.encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--baseline-manifest', type=Path, required=True)
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.baseline_manifest.read_text())
    if manifest['scientific_source_sha256'] != BASELINE:
        raise ValueError('Unexpected original source baseline.')
    files = manifest['files']
    for name, expected in files.items():
        if sha(args.original / name) != expected:
            raise ValueError(f'Original training checkout changed: {name}')
    if scientific_hash(inventory(args.original)) != BASELINE:
        raise ValueError('Original source has unexpected added or removed files.')
    changed = []
    for name, expected in files.items():
        actual = sha(args.snapshot / name)
        if actual != expected:
            if name not in CHANGED:
                raise ValueError(f'Unexpected experiment delta: {name}')
            changed.append(name)
    for name in ADDED:
        if not (args.snapshot / name).is_file():
            raise ValueError(f'Missing experiment driver: {name}')
    actual_files = inventory(args.snapshot)
    old_package = {n for n in files if n.startswith('src/gradscf/')
                   and Path(n).suffix in {'.py', '.json', '.dat'}}
    if set(actual_files) != old_package:
        raise ValueError('Unexpected experiment package additions or removals.')
    for name in CHANGED | ADDED:
        actual_files[name] = sha(args.snapshot / name)
    native_relative = 'src/gradscf/integrals/_native/lib/libgradscf_integrals.so'
    if sha(args.snapshot / native_relative) != NATIVE or sha(args.original / native_relative) != NATIVE:
        raise ValueError('Native integral library changed.')
    if sha(args.checkpoint) != CHECKPOINT:
        raise ValueError('Frozen initialization checkpoint changed.')
    rows = [json.loads(line) for line in args.structures.read_text().splitlines() if line]
    if len(rows) != 4 or {r['structure_id'] for r in rows} != PILOT_IDS:
        raise ValueError('The optimization pilot must contain the four declared training structures.')
    if any(r['charge'] != 0 or r['spin'] != 0 for r in rows):
        raise ValueError('This first pilot uses neutral closed-shell RHF only.')
    report = dict(original_scientific_source_sha256=BASELINE,
                  scientific_source_sha256=scientific_hash(actual_files),
                  changed_files=sorted(changed), added_files=sorted(ADDED),
                  files=actual_files, native_library_sha256=NATIVE,
                  initialization_checkpoint_sha256=CHECKPOINT,
                  structures_sha256=sha(args.structures),
                  structure_ids=[r['structure_id'] for r in rows],
                  baseline_files_verified=len(files),
                  original_source_unchanged=True)
    if args.verify:
        if json.loads(args.report.read_text()) != report:
            raise ValueError('Experiment snapshot changed after sealing.')
    else:
        if args.report.exists():
            raise FileExistsError('Preserve the already sealed experiment manifest.')
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, sort_keys=True))


if __name__ == '__main__':
    main()
