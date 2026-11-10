"""Bound one single-process probe and retain child RSS even on errors/timeouts."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
from time import perf_counter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--seconds", type=int, default=6000)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command or args.seconds <= 0:
    parser.error("Provide a command after -- and a positive timeout")
started = perf_counter()
child = subprocess.Popen(command, start_new_session=True)
status = "ok"
try:
    code = child.wait(timeout=args.seconds)
except subprocess.TimeoutExpired:
    status = "timeout"
    os.killpg(child.pid, signal.SIGTERM)
    try:
        code = child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        code = child.wait()
result = dict(command=command, status=status if status != "ok" else ("ok" if code == 0 else "error"),
    returncode=code, wall_seconds=perf_counter()-started,
    peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
    memory_metric="Linux ru_maxrss of this wrapper's sole child; imports/compile/execution included; not summed concurrent RSS")
args.output.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result), flush=True)
raise SystemExit(124 if status == "timeout" else code)
