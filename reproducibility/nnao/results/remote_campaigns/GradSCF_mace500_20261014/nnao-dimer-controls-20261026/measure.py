"""Portable child-process timing/RSS capture without system time(1)."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command:parser.error('Missing child command.')
    start=time.perf_counter();timestamp=datetime.now(timezone.utc).isoformat()
    code=subprocess.run(command).returncode
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    report=dict(command=command,job_id=os.environ.get('SLURM_JOB_ID'),host=socket.gethostname(),
        started_at_utc=timestamp,elapsed_seconds=time.perf_counter()-start,exit_code=code,
        max_rss=usage.ru_maxrss,max_rss_units='KiB on Linux; bytes on macOS',
        user_cpu_seconds=usage.ru_utime,system_cpu_seconds=usage.ru_stime)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PROCESS_RESOURCES',json.dumps(report),flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
