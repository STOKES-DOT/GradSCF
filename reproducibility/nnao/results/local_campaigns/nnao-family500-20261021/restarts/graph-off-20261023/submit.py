"""Submit three serial, fail-closed lanes; never submit the restart twice."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
LOGICAL_REPOSITORY = Path("/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021")


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    result_path = HERE / "submitted.json"
    assert not result_path.exists(), "Review the existing submission before resubmitting."
    plan = json.loads((HERE / "plan.json").read_text())
    assert json.loads((HERE / "preflight.json").read_text())["status"] == "passed"
    queued = subprocess.check_output(["squeue", "-u", "user_yuanjiao", "-h", "-o", "%j"], text=True)
    assert not any(name.startswith("nnao_") or name.startswith("nnao-") for name in queued.splitlines()), queued
    os.chdir(LOGICAL_REPOSITORY)
    os.environ["PWD"] = str(LOGICAL_REPOSITORY)
    record = dict(created_at=datetime.now(timezone.utc).isoformat(), status="submitting",
                  max_simultaneous_gpus=3, graph_capture=False, dependency_policy="afterok",
                  lanes={lane: [] for lane in plan["lanes"]})
    write(result_path, record)
    try:
        for lane, families in plan["lanes"].items():
            previous = None
            for family in families:
                script = HERE / (family + ".slurm")
                args = ["sbatch", "--parsable"]
                if previous is not None:
                    args.append(f"--dependency=afterok:{previous}")
                args.append(str(script))
                submitted = subprocess.run(args, text=True, capture_output=True)
                if submitted.returncode:
                    raise RuntimeError(submitted.stdout + submitted.stderr)
                job = int(submitted.stdout.strip().splitlines()[-1].split(";")[0])
                row = dict(family=family, job_id=job,
                           dependency=f"afterok:{previous}" if previous is not None else None,
                           script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                           seed_update=plan["families"][family]["update_step"],
                           resources=plan["resource_policy"][lane])
                record["lanes"][lane].append(row)
                write(result_path, record)
                print(json.dumps(row), flush=True)
                previous = job
                time.sleep(2.5)
        record["status"] = "submitted"
    except Exception as error:
        record.update(status="submission_failed", error=str(error))
        raise
    finally:
        write(result_path, record)


if __name__ == "__main__":
    main()
