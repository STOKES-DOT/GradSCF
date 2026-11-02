"""Lightweight Slurm release controller. Never modifies another task's jobs.

Jobs are submitted held. Release requires a passing export/preflight gate,
two consecutive headroom observations, <=3 account GPUs and <=64 account CPUs
after accounting for running jobs and already released pending requests.
"""
from pathlib import Path
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

# Slurm cli_filter validates logical PWD, not the resolved backing-store path.
ROOT=Path('/home/user_yuanjiao/private/homefile/NNAO_basis_accuracy_20261029')
GPU_CAP=3
CPU_CAP=64
RESOURCES={'preflight':(1,16),'small':(1,16),'tz':(1,16),'qz':(1,32)}


def write(path,data):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(path)


def command(args):
    r=subprocess.run(args,cwd=ROOT,env=dict(os.environ,PWD=str(ROOT)),capture_output=True,text=True)
    if r.returncode:raise RuntimeError(str(args)+': '+r.stderr)
    return r.stdout


def parse_gpu_count(description):
    """Use canonical Slurm TRES; a GPU node can run CPU-only jobs."""
    fields=dict(t.split('=',1) for t in description.split() if '=' in t)
    value=fields.get('AllocTRES',fields.get('TRES'))
    if value is None:raise ValueError('No authoritative TRES field in Slurm response')
    generic=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',value)
    if generic:return int(generic.group(1))
    return sum(map(int,re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)',value)))


def queue():
    output=command(['squeue','--noheader','--user','user_yuanjiao','--format','%i|%T|%C|%b|%r'])
    result={}
    for line in output.splitlines():
        job,state,cores,gres,reason=line.strip().split('|',4)
        description=command(['scontrol','show','job',job,'--oneliner'])
        result[job]=dict(state=state,cpu=int(cores),gpu=parse_gpu_count(description),reason=reason)
    return result


def submit(name):
    output=command(['sbatch','--parsable','--hold','--chdir='+str(ROOT),str(ROOT/(name+'.slurm'))])
    job=output.strip().splitlines()[-1].split(';')[0]
    assert job.isdigit()
    return job


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--direct',action='store_true');args=parser.parse_args()
    direct=args.direct
    state_path=ROOT/'scheduler.json'
    if state_path.exists():state=json.loads(state_path.read_text())
    else:
        names=['small','tz','qz'] if direct else ['preflight']
        state={'jobs':{},'released':[],'gpu_cap':GPU_CAP,'cpu_cap':CPU_CAP,
               'execution_mode':'direct' if direct else 'validated'}
        for name in names:
            state['jobs'][name]=submit(name);write(state_path,state);time.sleep(3)
        write(state_path,state)
    assert (state.get('execution_mode')=='direct')==direct,'Controller mode differs from scheduler state'
    free_observations={};start=time.monotonic()
    while time.monotonic()-start < 5*86400:
        try:
            jobs=queue()
            prep=(ROOT/'export-gate.json').exists() and json.loads((ROOT/'export-gate.json').read_text())['status']=='passed'
            runtime_file=ROOT/('runtime/toolkit-validation-gpu.json' if direct else 'runtime/ready.json')
            runtime=runtime_file.exists() and json.loads(runtime_file.read_text())['status']=='passed'
            prep=prep and runtime
            gate=prep if direct else ((ROOT/'preflight.json').exists() and json.loads((ROOT/'preflight.json').read_text())['status']=='passed')
            pref=state['jobs'].get('preflight')
            if not direct and pref in state['released'] and pref not in jobs and not gate:
                state.update(status='preflight_failed',message='Inspect logs; formal jobs were not released')
                write(state_path,state);return
            if not direct and gate and len(state['jobs'])==1:
                for name in ['small','tz','qz']:
                    state['jobs'][name]=submit(name);write(state_path,state);time.sleep(3)
            owned=set(state['jobs'].values())
            used_gpu=used_cpu=0
            for jid,job in jobs.items():
                active=job['state'] in ['RUNNING','CONFIGURING','COMPLETING']
                reserved=job['state']=='PENDING' and (
                    jid in state['released'] or (jid not in owned and job['reason'] not in
                        ['Dependency','DependencyNeverSatisfied','JobHeldUser','BeginTime']))
                if active or reserved:used_gpu+=job['gpu'];used_cpu+=job['cpu']
            state.update(status='running',checked_at=datetime.datetime.now().astimezone().isoformat(),
                account_gpus_running_or_reserved=used_gpu,account_cpus_running_or_reserved=used_cpu,
                export_gate_passed=prep,preflight_passed=None if direct else gate,
                preflight_policy='skipped_by_explicit_user_instruction' if direct else 'required',
                queue={j:jobs.get(j) for j in owned})
            for name,jid in state['jobs'].items():
                if jid in state['released'] or jid not in jobs:continue
                ready=prep if name=='preflight' else gate
                g,c=RESOURCES[name]
                enough=ready and used_gpu+g<=GPU_CAP and used_cpu+c<=CPU_CAP
                free_observations[name]=free_observations.get(name,0)+1 if enough else 0
                if free_observations[name]>=2:
                    command(['scontrol','release',jid]);state['released'].append(jid)
                    used_gpu+=g;used_cpu+=c;write(state_path,state)
            if len(state['jobs'])==(3 if direct else 4) and all(j in state['released'] and j not in jobs for j in state['jobs'].values()):
                statuses={n:json.loads((ROOT/f'lane-{n}.json').read_text())['status']
                    if (ROOT/f'lane-{n}.json').exists() else 'missing' for n in ['small','tz','qz']}
                state.update(status='complete' if all(s=='passed' for s in statuses.values()) else 'needs_repair',lane_status=statuses)
                write(state_path,state)
                if state['status']=='complete':
                    command([sys.executable,str(ROOT/'aggregate.py')])
                    state['analysis']='complete';write(state_path,state)
                return
            write(state_path,state)
        except Exception as error:
            state.update(last_controller_error=str(error),error_time=datetime.datetime.now().isoformat())
            write(state_path,state)
        time.sleep(60)
    state.update(status='controller_deadline_reached');write(state_path,state)


if __name__=='__main__':main()
