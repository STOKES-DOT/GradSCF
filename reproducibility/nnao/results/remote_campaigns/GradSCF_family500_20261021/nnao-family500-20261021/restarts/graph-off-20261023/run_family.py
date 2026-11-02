"""Run the frozen campaign with CUDA command buffers disabled before JAX starts."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import traceback


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    args = parser.parse_args()
    flags = shlex.split(os.environ.get("XLA_FLAGS", ""))
    selections = [flag for flag in flags if flag.startswith("--xla_gpu_enable_command_buffer=")]
    if not selections or selections[-1] != "--xla_gpu_enable_command_buffer=":
        raise RuntimeError("CUDA command buffers must be disabled before importing JAX.")
    plan = json.loads((ROOT / "plan.json").read_text())
    if args.family not in plan["families"]:
        raise ValueError("Unknown family: " + args.family)
    job = os.environ["SLURM_JOB_ID"]
    attempt = os.environ.get("SLURM_RESTART_COUNT", "0")
    record = dict(family=args.family, slurm_job_id=job, restart_count=int(attempt),
                  started_at=datetime.now(timezone.utc).isoformat(),
                  xla_flags=os.environ.get("XLA_FLAGS"),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  cpu_limit=os.environ.get("SLURM_CPUS_PER_TASK"),
                  expected_start=plan["families"][args.family],
                  scientific_source_sha256=plan["scientific_source_sha256"],
                  finite_differences=False, extra_evaluations=False, status="starting")
    destination = ROOT / "execution" / (args.family + "-" + job + "-r" + attempt + ".json")
    write(destination, record)
    try:
        spec = importlib.util.spec_from_file_location("frozen_nnao_campaign", CAMPAIGN / "campaign.py")
        campaign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(campaign)
        campaign.verify()
        record["status"] = "running"
        write(destination, record)
        campaign.train(args.family)
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        record["ended_at"] = datetime.now(timezone.utc).isoformat()
        write(destination, record)


if __name__ == "__main__":
    main()
