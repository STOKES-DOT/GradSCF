"""A serial thirty-minute diagnostic inside one existing Slurm allocation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Slurm allocation required")
    root, output = args.campaign.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = root/"candidate-roundoff-20261104"
    probes = Path(__file__).resolve().parent
    deadline = time.monotonic()+1740  # Leave60s for final Slurm cleanup.
    records = []
    manifest = dict(started_at=datetime.datetime.now().astimezone().isoformat(),
        allocation=os.environ["SLURM_JOB_ID"], source=str(source), python=sys.executable,
        budget_seconds=1740, probes_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in probes.glob("*.py")}, interpretation="Serial independent processes; fixed5 is a nonconverged diagnostic, not a full SCF benchmark. No production edits.")
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    env = dict(os.environ, PYTHONPATH=str(source/"src")+":"+str(source),
        JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1", JAX_ENABLE_COMPILATION_CACHE="false",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        GRADSCF_SOURCE_SNAPSHOT=source.name, PYTHONUNBUFFERED="1")

    def run(name, config, mode, cap, minimum, extra=()):
        remaining = deadline-time.monotonic()
        if remaining < minimum+20:
            row = dict(phase=name, status="skipped_budget", remaining_seconds=remaining)
        else:
            budget = min(cap, remaining-20)
            guard = output/(name+".process.json")
            command = ["srun", "--exact", "--gres=none", "--mem=240G", "-N1", "-n1", "-c64",
                "--chdir="+str(source), sys.executable, "benchmarks/probes/response_grid3/guard.py",
                "--seconds", str(int(budget)+10), "--rss-bytes", str(200*1024**3), str(guard), "--",
                sys.executable, str(probes/"profile_stages.py"), str(config), str(output/name),
                "--mode", mode, "--budget-seconds", str(budget), *extra]
            tick = time.monotonic()
            with (output/(name+".launcher.log")).open("x") as log:
                code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            row = dict(phase=name, returncode=code, process_wall_seconds=time.monotonic()-tick,
                       budget_seconds=budget, status="finished" if code==0 else "failed_or_timed_out")
        records.append(row)
        (output/"phases.json").write_text(json.dumps(records, indent=2)+"\n")
        print(json.dumps(row), flush=True)

    inputs = root/"plan-v2/configs"
    case = "p-benzoquinone-anion-def2-tzvp-{}-pbe0-gradscf.json"
    run("gks8-clean", inputs/case.format("gks"), "clean", 1000, 700)
    run("gks8-fixed5-diagnostic", inputs/case.format("gks"), "diagnostic", 600, 210,
        ("--fixed-cycles", "5", "--jax-trace"))
    run("uks8-ri-diagnostic", inputs/case.format("uks"), "diagnostic", 240, 160)
    cfg = json.loads((inputs/case.format("gks")).read_text())
    cfg.update(threads=64, case_id=cfg["case_id"]+"-profile-cpu64")
    cpu64 = output/"gks-cpu64-input.json"
    cpu64.write_text(json.dumps(cfg, indent=2)+"\n")
    run("gks64-clean", cpu64, "clean", 600, 150)
    manifest.update(finished_at=datetime.datetime.now().astimezone().isoformat(),
                    remaining_seconds=deadline-time.monotonic())
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
