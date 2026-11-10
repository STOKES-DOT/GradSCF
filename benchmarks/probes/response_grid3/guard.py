"""One bounded process; reuse the existing process-group RSS sampler."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
from benchmarks.probes.tddft_perf.run import rss_group

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
parser.add_argument('--seconds', type=int, required=True)
parser.add_argument('--rss-bytes', type=int, required=True)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
assert command and args.seconds > 0 and args.rss_bytes > 0
assert not args.output.exists(), 'Raw process records are immutable.'
start, peak, status = time.monotonic(), 0, 'ok'
child = subprocess.Popen(command, start_new_session=True)
while child.poll() is None:
    peak = max(peak, rss_group(child.pid))
    if time.monotonic()-start > args.seconds:
        status = 'timeout'
    elif peak > args.rss_bytes:
        status = 'rss_guard'
    if status != 'ok':
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
        break
    time.sleep(.2)
code = child.wait()
record = dict(command=command, status=status if status != 'ok' else ('ok' if code == 0 else 'error'),
    returncode=code, wall_seconds=time.monotonic()-start, timeout_seconds=args.seconds,
    rss_limit_bytes=args.rss_bytes, sampled_group_rss_peak_bytes=peak, sample_seconds=.2,
    child_ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
args.output.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
raise SystemExit(124 if status == 'timeout' else 137 if status == 'rss_guard' else code)
