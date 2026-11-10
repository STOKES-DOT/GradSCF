"""Refresh derived campaign summaries; no scientific work or job submission."""
import argparse
import datetime
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("allocation")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error("Summary interval must be at least30 seconds")
    root = args.campaign.resolve()
    harness = root/"harness/frozen-20261104"
    source = root/"candidate-roundoff-20261104"
    plan = root/"plan-v2"
    original = root/"results/final-matrix"
    overlay = root/"results/roundoff-retries-20261104"
    while True:
        print(datetime.datetime.now().astimezone().isoformat(), flush=True)
        try:
            for results in (original, overlay):
                subprocess.run([sys.executable, str(harness/"refresh_results.py"),
                    str(plan), str(results), str(source), args.allocation], check=True)
            subprocess.run([sys.executable, str(harness/"summarize.py"), str(plan), str(original),
                str(root/"results/summary-20261104"), "--overlay", str(overlay),
                "--timing-exclusions", str(root/"timing-exclusions-20261104.json")], check=True)
        except (subprocess.CalledProcessError, OSError) as error:
            print(f"Summary refresh failed: {error}; original records are preserved.", flush=True)
        active = subprocess.check_output(["squeue", "-h", "-j", args.allocation, "-o", "%T"], text=True).strip()
        if not active:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
