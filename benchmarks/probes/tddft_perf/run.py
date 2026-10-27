"""Serial subprocess driver with an explicit wall-time and RSS guard."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def rss_group(pid):
    total = 0
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = path.read_text().split(") ", 1)[1].split()
            if int(stat[2]) == pid:
                total += int(stat[21])*os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            pass
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=int, default=7200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic()+args.seconds
    for cfg in json.loads(args.matrix.read_text()):
        if time.monotonic() >= deadline:
            break
        name = cfg["case_id"]+"-"+cfg["variant"]
        folder = args.output/"runs"/name
        folder.mkdir(parents=True, exist_ok=True)
        (folder/"input.json").write_text(json.dumps(cfg, indent=2)+"\n")
        affinity = sorted(os.sched_getaffinity(0))[:cfg["threads"]]
        def restrict():
            os.sched_setaffinity(0, affinity)
        env = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(cfg["threads"])
        env.update(JAX_PLATFORMS="cuda,cpu" if cfg["device"] == "gpu" else "cpu",
                   JAX_ENABLE_X64="true", JAX_ENABLE_COMPILATION_CACHE="false",
                   JAX_LOG_COMPILES="true" if cfg.get("profile") else "false",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, str(Path(__file__).with_name("worker.py")), str(folder/"input.json"), str(folder)]
        if cfg.get("entry"):
            command = [sys.executable, str(Path(__file__).with_name(cfg["entry"])), str(folder/"input.json"), str(folder)]
        peak, samples, state = 0, [], "ok"
        started = time.monotonic()
        with (folder/"stdout.log").open("w") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                     preexec_fn=restrict, start_new_session=True)
            while child.poll() is None:
                current = rss_group(child.pid)
                peak = max(peak, current)
                samples.append([time.monotonic()-started, current])
                if time.monotonic() > min(deadline, started+cfg.get("timeout", 600)):
                    state = "timeout"
                if current > cfg.get("rss_limit_bytes", 64*1024**3):
                    state = "oom_guard"
                if state != "ok":
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(.05)
            code = child.wait()
        result = dict(case_id=cfg["case_id"], variant=cfg["variant"], command=command,
                      status=state if state != "ok" else ("ok" if code == 0 else "error"),
                      returncode=code, wall_seconds=time.monotonic()-started,
                      cpu_peak_rss_bytes=peak, rss_sampling_seconds=.05, affinity=affinity,
                      notes="sum process-group Linux RSS sampled at 50ms; can miss short peaks")
        (folder/"process.json").write_text(json.dumps(result, indent=2)+"\n")
        (folder/"rss.json").write_text(json.dumps(samples)+"\n")
        with (args.output/"processes.jsonl").open("a") as stream:
            stream.write(json.dumps(result)+"\n")
        print(name, result["status"], round(result["wall_seconds"], 3), flush=True)


if __name__ == "__main__":
    main()
