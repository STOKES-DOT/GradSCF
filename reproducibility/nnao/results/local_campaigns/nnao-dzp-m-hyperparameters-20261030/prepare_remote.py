"""Extract matched input provenance and cache metadata; no electronic calculations."""
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).absolute().parent
PARENT = Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021')
CAMPAIGN = PARENT/'artifacts/nnao-family500-20261021'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    plan = json.loads((HERE/'execution-plan.json').read_text())
    checkpoint = CAMPAIGN/'runs/nnao_dzp_m/train/last.npz'
    if digest(checkpoint) != plan['source_checkpoint_sha256']:
        raise RuntimeError('The source checkpoint changed; do not silently choose a different pool.')
    with np.load(checkpoint, allow_pickle=False) as raw:
        identity = json.loads(str(raw['metadata']))['identity']
    if identity['scientific_source_sha256'] != plan['scientific_source_sha256']:
        raise RuntimeError('Checkpoint scientific source differs.')
    selected = json.loads((HERE/'proposed50.json').read_text())['structures']
    originals = {r['structure_id']: r for r in identity['references']['structures']}
    caches = {}
    for row in selected:
        identifier = row['structure_id']
        for field in ('symbols', 'coords_angstrom', 'charge', 'spin'):
            if row[field] != originals[identifier][field]:
                raise RuntimeError(f'{identifier}: selection changes {field}')
        source = identity['caches'][identifier]
        folder = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        path = CAMPAIGN/'cache/m'/folder/'integrals.npz'
        manifest = json.loads(path.with_suffix('.json').read_text())
        if manifest != {k: v for k, v in source.items() if k != 'contracted_nao'}:
            # Older manifests can already contain contracted_nao.
            if {k: v for k, v in manifest.items() if k != 'contracted_nao'} != {
                    k: v for k, v in source.items() if k != 'contracted_nao'}:
                raise RuntimeError(f'{identifier}: cache manifest differs from checkpoint')
        if not path.is_file():
            raise FileNotFoundError(path)
        caches[identifier] = {k: source[k] for k in ('signature', 'cache_sha256',
            'bytes', 'pool_nao', 'contracted_nao', 'df_shape')}
    data = dict(identity['references'], structures=selected,
        description='Matched 50-system hidden L1/L2 capacity comparison; actual SCF acceptance at epoch zero.')
    write(HERE/'inputs/references.json', data)
    write(HERE/'inputs/candidate.json', identity['candidate'])
    plan.update(cache_root=str(CAMPAIGN/'cache/m'), caches=caches,
        input_sha256={name: digest(HERE/name) for name in
            ['inputs/references.json', 'inputs/candidate.json', 'run_hidden_l.py']})
    write(HERE/'execution-plan.json', plan)
    write(HERE/'prepared.json', dict(status='prepared_not_numerically_evaluated',
        structures=len(selected), cache_bytes=sum(c['bytes'] for c in caches.values()),
        max_contracted_nao=max(c['contracted_nao'] for c in caches.values()),
        plan_sha256=digest(HERE/'execution-plan.json')))
    print((HERE/'prepared.json').read_text())


if __name__ == '__main__':
    main()
