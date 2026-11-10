"""Serial, resumable execution of a frozen matrix in an existing allocation.

This login-side controller does no electronic-structure work. Every worker is
an srun step with its own process/RSS guard. It never submits another allocation.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def finish_attempt(output, case, cfg, allocation, source, returncode=None):
    """Rebuild derived run metadata from immutable worker/guard records."""
    process_path = output/(case+".process.json")
    process = json.loads(process_path.read_text()) if process_path.exists() else {}
    runtime_path = output/case/"runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    summary_path = output/case/"summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    records_path = output/case/"measurements.jsonl"
    records = [] if not records_path.exists() else [json.loads(s) for s in records_path.read_text().splitlines()]
    timeout = output/case/"timeout.json"
    code = process.get("returncode") if returncode is None else returncode
    if timeout.exists() or process.get("status") == "timeout":
        status = "timeout"
    elif process.get("status") == "rss_guard":
        status = "oom"
    elif not process:
        status = "interrupted_or_running"
    elif process.get("status") != "ok" or code != 0:
        status = "error"
    else:
        status = summary.get("status","error")
    success = bool(process.get("status") == "ok" and code == 0)
    complete = bool(success and status == "ok" and len(records) == cfg["repeats"])
    result = dict(case_id=case,status=status,process_status=process.get("status","missing"),
        process_returncode=process.get("returncode"),process_wall_seconds=process.get("wall_seconds"),
        process_rss_peak_bytes=process.get("sampled_group_rss_peak_bytes"),
        attempt_finished=bool(process),run_complete=complete,
        executed_repeats=len(records),requested_repeats=cfg["repeats"],
        timeout_scope=json.loads(timeout.read_text()) if timeout.exists() else "whole worker",
        accepted_for_comparison=bool(complete and all(r.get("scientific_pass") for r in records)),
        last_completed_scf=summary,allocation=str(runtime.get("job",allocation)),
        source_snapshot=str(source.parent/runtime["source_snapshot"]) if runtime.get("source_snapshot") else str(source),
        derived_under_allocation=allocation)
    (output/(case+".run.json")).write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("allocation")
    parser.add_argument("--output-name", default="final-matrix")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--case-list", type=Path, help="Explicit subset; original input configurations stay unchanged")
    parser.add_argument("--keep-allocation", action="store_true", help="Leave release to a following phase")
    args = parser.parse_args()
    root, source, plan_path = (p.resolve() for p in (args.campaign,args.source,args.plan))
    plan = json.loads((plan_path/"run_plan.json").read_text())
    order = plan["run_order"] if args.case_list is None else json.loads(args.case_list.read_text())
    if not set(order).issubset(plan["run_order"]):
        parser.error("Case list contains configurations outside the frozen plan")
    python = root.parent/"gradscf-libcint-nnao-20261026/env-compare/bin/python"
    worker = (args.worker.resolve() if args.worker is not None else
              source/"benchmarks/probes/scf_solver/runner.py")
    output = root/"results"/args.output_name
    output.mkdir(parents=True,exist_ok=True)
    # The site's CLI filter checks logical PWD under ~/private/homefile;
    # resolving the storage symlink produces an /opt path it will reject.
    submission_root = args.campaign.expanduser().absolute()
    os.chdir(submission_root)
    os.environ["PWD"] = str(submission_root)
    completed = 0
    deferred = []
    uncertain_attempts = []
    normal_completion = False
    try:
        for case in order:
            if args.limit is not None and completed >= args.limit:
                break
            process_path = output/(case+".process.json")
            config_path = plan_path/"configs"/(case+".json")
            cfg = json.loads(config_path.read_text())
            if process_path.exists():
                finish_attempt(output,case,cfg,args.allocation,source)
                continue  # Recover metadata; never overwrite a raw attempt.
            if (output/(case+".log")).exists():
                finish_attempt(output,case,cfg,args.allocation,source)
                uncertain_attempts.append(case)
                continue  # Inspect the original step before any retry.
            end = subprocess.check_output(["squeue","-h","-j",args.allocation,"-o","%e"],text=True).strip()
            if not end:
                raise RuntimeError("Allocation is no longer active")
            remaining = int(datetime.datetime.fromisoformat(end).timestamp()-time.time())
            if remaining < cfg["process_limit_seconds"]+90:
                deferred.append(case)
                continue
            env = dict(os.environ,PWD=str(submission_root),PYTHONPATH=str(source/"src")+":"+str(source),
                JAX_PLATFORMS="cpu",JAX_ENABLE_X64="1",JAX_ENABLE_COMPILATION_CACHE="false",
                OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONUNBUFFERED="1",
                GRADSCF_NATIVE_THREADS=str(cfg["threads"]),GRADSCF_SOURCE_SNAPSHOT=source.name)
            command = ["srun","--jobid="+args.allocation,"--exact","--gres=none","--mem=240G",
                "-N1","-n1","-c"+str(cfg["threads"]),"--chdir="+str(source),str(python),
                "benchmarks/probes/response_grid3/guard.py","--seconds",str(cfg["process_limit_seconds"]),
                "--rss-bytes",str(plan["rss_guard_bytes"]),str(process_path),"--",str(python),
                str(worker),str(config_path),str(output/case)]
            with (output/(case+".log")).open("x") as log:
                returncode = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            result = finish_attempt(output,case,cfg,args.allocation,source,returncode)
            if returncode != 0 and not process_path.exists():
                uncertain_attempts.append(case)
                print(json.dumps(dict(case_id=case,status="unconfirmed_launch",
                    returncode=returncode,action="stop and retain allocation")),flush=True)
                break
            completed += 1
            print(json.dumps(dict(completed_this_session=completed,case_id=case,status=result["status"])),flush=True)
        (output/"deferred.json").write_text(json.dumps(deferred,indent=2)+"\n")
        (output/"uncertain_attempts.json").write_text(json.dumps(uncertain_attempts,indent=2)+"\n")
        normal_completion = True
    finally:
        if normal_completion and not uncertain_attempts and not args.keep_allocation:
            (root/("release."+args.allocation)).touch()


if __name__ == "__main__":
    main()
