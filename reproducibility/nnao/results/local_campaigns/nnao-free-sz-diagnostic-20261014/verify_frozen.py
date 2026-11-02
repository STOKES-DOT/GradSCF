"""Independent GPU4PySCF evaluation of the selected free-contraction bases."""
import importlib.util
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
PARENT=OUT.parent/'nnao-relative200-20261014'

def main():
    definition=json.loads((OUT/'configuration.json').read_text())
    rows=json.loads((OUT/'structures.json').read_text())
    results=json.loads((OUT/'summary.json').read_text())
    successful={r['structure_id']:r for r in results['results']
                if r['status'] in ['complete','partial'] and 'best_energy_hartree' in r}
    frozen=[]
    for path in sorted(OUT.glob('*-best-basis.json')):
        row=json.loads(path.read_text())
        assert row['structure_id'] in successful
        row.update(basis='NNAO-free-SZ',atom_labels=[f'{s}{i}' for i,s in enumerate(row['symbols'])],
            native_energy_hartree=row['energy_hartree'],
            checkpoint_profile='epoch629-pool-per-structure-free-SZ',checkpoint_epoch=629,
            checkpoint_sha256=definition['checkpoint_sha256'],
            native_source_sha256=definition['scientific_source_sha256'],
            note='Reference energy from projected native primitive integrals; independent contracted-basis check.',
            model_training_structure_count=200)
        frozen.append(row)
    (OUT/'verify-structures.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (OUT/'verify-frozen.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in frozen))
    spec=importlib.util.spec_from_file_location('gpu4_reference',PARENT/'gpu4_reference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.CUSTOM_BASES=(*module.CUSTOM_BASES,'NNAO-free-SZ')
    return module.main(['--structures',str(OUT/'verify-structures.jsonl'),
        '--frozen-bases',str(OUT/'verify-frozen.jsonl'),'--output',str(OUT/'gpu4-verification'),
        '--bases','NNAO-free-SZ'])

if __name__=='__main__':
    raise SystemExit(main())
