"""Serial bounded processes with separate solve/process deadline scopes."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('td_probe_parent_controller',ROOT/'benchmarks/probes/tddft_perf/run.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
rss_group=parent.rss_group
from status import summarize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('matrix',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--seconds',type=int,default=7200)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+args.seconds
    for cfg in json.loads(args.matrix.read_text()):
        if time.monotonic()>=deadline:break
        folder=args.output/(cfg['case_id']+'-'+cfg['variant'])
        folder.mkdir(parents=True,exist_ok=False)
        path=folder/'input.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
        env=dict(os.environ)
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','GRADSCF_NATIVE_THREADS']:
            env[key]=str(cfg.get('threads',1))
        env.update(JAX_PLATFORMS='cuda,cpu' if cfg['device']=='gpu' else 'cpu',
                   JAX_ENABLE_X64='true',JAX_ENABLE_COMPILATION_CACHE='false',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONDONTWRITEBYTECODE='1')
        affinity=sorted(os.sched_getaffinity(0))[:cfg.get('threads',1)]
        def bind():os.sched_setaffinity(0,affinity)
        entry=ROOT/cfg['entry'] if cfg.get('entry') else Path(__file__).with_name('worker.py')
        command=[sys.executable,str(entry),str(path),str(folder)]
        started=time.monotonic();stop=min(deadline,started+cfg.get('process_budget_seconds',1200))
        peak=clean_peak=0;timeline=[];status='ok';scope=None
        with (folder/'stdout.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,
                                   preexec_fn=bind,start_new_session=True)
            while child.poll() is None:
                now=time.monotonic();rss=rss_group(child.pid);peak=max(peak,rss)
                if not (folder/'clean_phase_done.json').exists():clean_peak=max(clean_peak,rss)
                timeline.append([now-started,rss])
                if now>stop:status='timeout';scope='whole_repeated_process'
                active=folder/'active_stage.json'
                if active.exists():
                    try:
                        stage=json.loads(active.read_text())
                        if stage['stage']=='response_solve' and now-stage['started_monotonic']>stage['solve_budget_seconds']:
                            status='timeout';scope='single_response_solve'
                    except (json.JSONDecodeError,KeyError):pass
                if rss>cfg.get('rss_limit_bytes',64*1024**3):status='oom_guard';scope='process_rss_guard'
                if status!='ok':
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
                time.sleep(.05)
            code=child.wait()
        process=dict(status=status if status!='ok' else ('ok' if code==0 else 'error'),returncode=code,
            wall_seconds=time.monotonic()-started,timeout_scope=scope,
            process_budget_seconds=cfg.get('process_budget_seconds',1200),solve_budget_seconds=cfg.get('solve_budget_seconds',600),
            sampled_rss_peak_bytes=peak,sampled_clean_rss_peak_bytes=clean_peak,rss_sampling_seconds=.05,
            command=command,affinity=affinity)
        records=[]
        if (folder/'measurements.jsonl').exists():records=[json.loads(s) for s in (folder/'measurements.jsonl').read_text().splitlines()]
        cert=next((x for x in reversed(records) if x['stage']=='certification'),{})
        process.update(summarize(process,records,residuals=cert.get('residual_per_root'),tolerance=cfg['td_tol'],
                       supersedes_run=cfg.get('supersedes_run'),residual_source='post-clean original operator' if cert else None))
        (folder/'process.json').write_text(json.dumps(process,indent=2)+'\n')
        (folder/'rss.json').write_text(json.dumps(timeline)+'\n')
        print(folder.name,process['process_status'],process['scientific_pass'],round(process['wall_seconds'],3),flush=True)


if __name__=='__main__':main()
