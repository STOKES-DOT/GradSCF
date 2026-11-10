"""Rebuild derived status after completed NFS writes become visible.

Only process records already present are consumed. Original scientific outputs
are never changed, and no calculation is launched or retried.
"""
import argparse
import json
from pathlib import Path

from campaign import finish_attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("allocation")
    args = parser.parse_args()
    plan = json.loads((args.plan/"run_plan.json").read_text())
    count = 0
    for case in plan["run_order"]:
        if not (args.results/(case+".process.json")).exists():
            continue
        cfg = json.loads((args.plan/"configs"/(case+".json")).read_text())
        finish_attempt(args.results, case, cfg, args.allocation, args.source)
        count += 1
    print(json.dumps(dict(refreshed_derived_records=count)))


if __name__ == "__main__":
    main()
