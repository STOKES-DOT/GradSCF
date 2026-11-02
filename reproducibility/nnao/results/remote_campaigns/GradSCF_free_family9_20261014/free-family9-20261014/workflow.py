"""Single-allocation family campaign; each numerical member has a fresh process."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
PARENT=ROOT.parent/'nnao-relative200-20261014'
OLD=ROOT.parent/'nnao-free-sz-diagnostic-20261014'

def save(name,value):
    p=ROOT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def verify(tier,zeta):
    out=ROOT/'results'/f'{tier}-{zeta}'
    target=out/'gpu4-verification/comparison.json'
    if target.exists():
        return 0 if json.loads(target.read_text())['status']=='passed' else 1
    spec=importlib.util.spec_from_file_location('gpu4_reference',PARENT/'gpu4_reference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    label=f'NNAO-free-{tier}-{zeta}';module.CUSTOM_BASES=(*module.CUSTOM_BASES,label)
    rows=json.loads((OLD/'structures.json').read_text())
    structure_file=ROOT/'inputs/structures.jsonl'
    structure_file.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return module.main(['--structures',str(structure_file),'--frozen-bases',str(out/'frozen-bases.jsonl'),
        '--output',str(out/'gpu4-verification'),'--bases',label])

def main():
    from gradscf.model.nnao.runtime import scientific_source_hash,verify_source_imports
    expected=json.loads((ROOT/'provenance.json').read_text())['scientific_source_sha256']
    assert scientific_source_hash(REPO)==expected
    save('imported-sources.json',verify_source_imports(REPO))
    s_gate=json.loads((ROOT/'inputs/s-exponent-gradient-check.json').read_text())
    s_summary=json.loads((ROOT/'inputs/s-calibration-summary.json').read_text())
    assert s_gate['passed'] and s_summary['status']=='complete'
    snapshot=json.loads((PARENT/'snapshot.json').read_text())
    save('inputs/l-candidate.json',snapshot['candidate'])
    campaign=dict(status='running',source_sha256=expected,started_at=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),
        members=[],order=['s-1','s-2','s-3','l-1','l-2','l-3','m-calibration','m-1','m-2','m-3'])
    save('campaign.json',campaign)
    def run(args,platform='cuda,cpu'):
        env=dict(os.environ,JAX_PLATFORMS=platform,JAX_ENABLE_X64='1')
        return subprocess.run([sys.executable,'-u',*map(str,args)],env=env).returncode
    def member(tier,zeta):
        stage=f'{tier}-{zeta}';save('phase.json',dict(member=stage,phase='starting'))
        started=time.perf_counter()
        code=run([ROOT/'run_family.py','--tier',tier,'--zeta',zeta])
        summary=ROOT/'results'/stage/'summary.json'
        outcome=json.loads(summary.read_text())['status'] if summary.exists() else 'no_summary'
        check=run([__file__,'verify',tier,zeta]) if code==0 and summary.exists() else None
        campaign['members'].append(dict(member=stage,process_exit_code=code,status=outcome,
            verification_exit_code=check,elapsed_seconds=time.perf_counter()-started))
        save('campaign.json',campaign)
        run([ROOT/'aggregate.py'],'cpu')
    for tier in ['s','l']:
        for zeta in [1,2,3]:member(tier,zeta)
    calibration=ROOT/'m-calibration'
    save('phase.json',dict(phase='m-pool-calibration'))
    complete=calibration/'summary.json'
    if not complete.exists() or json.loads(complete.read_text())['status']!='complete':
        code=run(['-m','gradscf.model.nnao.workflows.pool_calibration',
            '--references',ROOT/'inputs/m-pilot.json','--output-dir',calibration,
            '--calibration-steps',20,'--contraction-steps',300,'--starts',3,'--seed',0,'--scf-platform','cuda'])
        if code:
            campaign.update(status='m_calibration_failed',calibration_exit_code=code)
            save('campaign.json',campaign);run([ROOT/'aggregate.py'],'cpu');return code
    code=run(['-m','gradscf.model.nnao.workflows.basis_grid','select',
        '--pilot-dir',calibration,'--references',ROOT/'inputs/m-pilot.json','--output-dir',ROOT/'m-selection'],'cpu')
    if code:
        campaign.update(status='m_selection_failed',selection_exit_code=code);save('campaign.json',campaign);return code
    candidate=json.loads((ROOT/'m-selection/candidate.json').read_text())
    save('inputs/m-candidate.json',candidate)
    for zeta in [1,2,3]:member('m',zeta)
    campaign['status']='complete' if all(x['status']=='complete' and x['verification_exit_code']==0
        for x in campaign['members']) and len(campaign['members'])==9 else 'complete_with_failures'
    campaign['elapsed_seconds']=time.time()-campaign['started_at']
    save('campaign.json',campaign);save('phase.json',dict(phase='complete',status=campaign['status']))
    run([ROOT/'aggregate.py'],'cpu')
    return 0 if campaign['status']=='complete' else 1

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='verify':raise SystemExit(verify(sys.argv[2],int(sys.argv[3])))
    raise SystemExit(main())
