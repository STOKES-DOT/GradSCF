"""Native independent STO-6G+P monomer references; no dimer-derived basis."""
import json
import os
from pathlib import Path
import sys
import time

import jax

sys.path.insert(0,str(Path.cwd()/'tests/comparisons'))
from nnao_core.native import make_inputs
from nnao_reactivity.rescue import solve_native
from gradscf.model.nnao.runtime import atomic_json,scientific_source_hash,repository_root,file_hash


def main():
    root=Path('artifacts/nnao-dimer-reference-correction-20261027')
    data=json.loads((root/'isolated-inputs.json').read_text())
    assert scientific_source_hash(repository_root())==data['source_sha256']
    assert jax.config.jax_enable_x64
    path=root/'isolated-results.json'
    report=json.loads(path.read_text()) if path.exists() else dict(status='running',results=[],
        method=data['method'],auxbasis=data['auxbasis'],input_sha256=file_hash(root/'isolated-inputs.json'),
        source_sha256=data['source_sha256'],job_id=os.environ.get('SLURM_JOB_ID'),
        backend=jax.default_backend(),finite_differences=False)
    done={(r['dataset'],r['group'],r['fragment']) for r in report['results']}
    for job in data['jobs']:
        key=job['dataset'],job['group'],job['fragment']
        if key in done:continue
        start=time.perf_counter();row=job['row'];shells=job['atom_shells']
        atomic_json(root/'progress.json',dict(completed=len(done),total=24,dataset=key[0],group=key[1],fragment=key[2]))
        physical=make_inputs(row,shells,auxbasis=data['auxbasis'])
        state=solve_native(physical['overlap'],physical['hcore'],physical['factors'],
            row['nelectron'],row['spin'],physical['enuc'],row=row,atom_shells=shells)
        result=dict(dataset=key[0],group=key[1],fragment=key[2],structure_id=row['structure_id'],
            basis='STO-6G+P',energy_hartree=float(state['energy']),nao=len(physical['overlap']),
            orbital_residual=state['orbital_residual'],stability=state['stability'],
            seconds=time.perf_counter()-start,status='passed')
        report['results'].append(result);done.add(key);atomic_json(path,report)
        print(json.dumps(result),flush=True);jax.clear_caches()
    report.update(status='complete',passed=len(done));assert len(done)==24
    atomic_json(path,report);atomic_json(root/'progress.json',dict(status='complete',completed=24,total=24))


if __name__=='__main__':main()
