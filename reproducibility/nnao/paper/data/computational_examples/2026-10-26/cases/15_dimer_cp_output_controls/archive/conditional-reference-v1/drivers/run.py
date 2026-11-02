"""Run the missing five-energy controls with the frozen native HF backend."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import jax

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nnao_core.native import counterpoise
from nnao_reactivity.rescue import solve_native
from nnao_dimer_controls.protocols import fingerprint
from gradscf.model.nnao.runtime import atomic_json,scientific_source_hash,repository_root,file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-cases',type=int)
    args=parser.parse_args()
    data=json.loads(args.inputs.read_text())
    source=scientific_source_hash(repository_root())
    assert source==data['frozen_scientific_source_sha256'],source
    assert jax.config.jax_enable_x64
    args.output.mkdir(parents=True,exist_ok=True)
    identity=dict(input_sha256=file_hash(args.inputs),source_sha256=source,
        driver_sha256=file_hash(Path(__file__)),protocols_sha256=file_hash(Path(__file__).with_name('protocols.py')),
        method=data['method'],auxbasis=data['auxbasis'],dtype=data['dtype'],
        checkpoint_sha256=data['checkpoint_sha256'],checkpoint_update=data['checkpoint_update'])
    config=args.output/'configuration.json'
    if config.exists():assert json.loads(config.read_text())==identity,'Changed resume identity.'
    else:atomic_json(config,identity)
    path=args.output/'results.json'
    report=json.loads(path.read_text()) if path.exists() else dict(status='running',requested_cases=204,
        results=[],reused_cases=len(data['reused_records']),input_sha256=identity['input_sha256'])
    completed={r['key']:r for r in report['results']}
    assert not any(r['status']!='passed' for r in completed.values()),'Review failed records before a retry.'
    started=time.perf_counter();count=0
    for case in data['jobs']:
        if case['key'] in completed:continue
        atomic_json(args.output/'progress.json',dict(status='running',completed=len(completed),total=204,
            identifier=case['identifier'],basis=case['basis'],policy=case['policy']))
        tick=time.perf_counter()
        record={k:case[k] for k in ('key','identifier','dataset','group','label','scale','basis','policy',
                                     'anchor_identifier','source_basis_key','reference_kcal_mol')}
        record.update(task='curve',host=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),
                      backend=jax.default_backend(),device=jax.devices()[0].device_kind)
        print('BEGIN',json.dumps(record),flush=True)
        try:
            answer=counterpoise(case['row'],case['atom_shells'],case['fragment_indices'],
                                auxbasis=data['auxbasis'],solver=solve_native)
            assert len(answer['scf_diagnostics'])==5
            assert all(r['stability']['stable'] for r in answer['scf_diagnostics'])
            record.update(answer,status='passed')
        except Exception as error:
            record.update(status='failed',error=str(error),traceback=traceback.format_exc())
        record['seconds']=time.perf_counter()-tick
        report['results'].append(record);completed[case['key']]=record
        report.update(passed=sum(r['status']=='passed' for r in report['results']),
                      failed=sum(r['status']!='passed' for r in report['results']),
                      current_process_seconds=time.perf_counter()-started)
        atomic_json(path,report);print('END',json.dumps(record),flush=True)
        jax.clear_caches();count+=1
        if case['key'] in data['priority_keys'] and record['status']!='passed':
            atomic_json(args.output/'gate.json',dict(status='failed',key=case['key']))
            raise RuntimeError('Dimer short numerical gate failed; retained data remain unchanged.')
        if all(key in completed and completed[key]['status']=='passed' for key in data['priority_keys']):
            atomic_json(args.output/'gate.json',dict(status='passed',keys=data['priority_keys'],
                same_five_energy_protocol=True,finite_differences=False))
        if args.max_cases and count>=args.max_cases:break
    report['status']=('complete' if len(completed)==204 and not report['failed'] else
                      'complete_with_failures' if len(completed)==204 else 'partial')
    atomic_json(path,report)
    atomic_json(args.output/'progress.json',dict(status=report['status'],completed=len(completed),
                                               total=204,passed=report['passed'],failed=report['failed']))
    print('FINISHED',report['status'],report['passed'],'/',len(completed),flush=True)


if __name__=='__main__':main()
