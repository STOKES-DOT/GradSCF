"""Compare paired workflow_reuse JSONL measurements, using only stdlib."""
import argparse
import json
from pathlib import Path
import statistics


NUMERICS = {
    "energy_hartree": 3e-9,
    "excitation_hartree": 3e-8,
    "mo_energy": 3e-8,
    "screening_energy": 3e-8,
    "qp_residual": 3e-8,
    "fixed_point_residual": 3e-8,
    "particle_number_error": 3e-8,
    "total_energy": 3e-8,
    "residual_norms": 3e-8,
}
FLAGS = ("qp_computed_mask", "converged_mask", "qp_branch_valid", "converged",
         "stable", "screening_valid", "response_valid", "scientific_pass", "capability_guard_pass")


def _error(left, right):
    if left is None or right is None:
        return 0. if left is right else float("inf")
    if isinstance(left, list) and isinstance(right, list):
        return max((_error(a, b) for a, b in zip(left, right)), default=0.) if len(left) == len(right) else float("inf")
    if isinstance(left, dict) and isinstance(right, dict):
        return max((_error(left[key], right[key]) for key in left), default=0.) if left.keys() == right.keys() else float("inf")
    return abs(left-right)


def _rows(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {(r["case"], r["repeat"], r["stage"]): r for r in records}
    if len(keyed) != len(records):
        raise ValueError("Repeated case/repeat/stage key; an error may follow a failed phase")
    return keyed


def compare(before, after):
    left, right = _rows(before), _rows(after)
    failures, parity, timings = [], [], []
    if not left or not right:
        failures.append("An input contains no measurement records")
    if left.keys() != right.keys():
        failures.append("Case/repeat/stage records differ")
    for label, rows in (("before", left), ("after", right)):
        for case, repeat in {(key[0], key[1]) for key in rows}:
            workflow = rows.get((case, repeat, "workflow"))
            if workflow is None or workflow.get("status") != "complete":
                failures.append(f"{label}: {case}/{repeat} has no completed workflow")
        for key, row in rows.items():
            if row.get("status") == "verified_unsupported":
                if row.get("capability_guard_pass") is not True:
                    failures.append(f"{label}: {key} did not verify its capability guard")
            elif key[2] != "workflow" and row.get("scientific_pass") is not True:
                failures.append(f"{label}: {key} did not pass its scientific gate")
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key], right[key]
        if a.get("status") == "error" or b.get("status") == "error":
            failures.append(f"{key}: phase failed")
        for field in ("suite", "molecule", "basis", "backend", "xc", "grid", "dtype") + FLAGS:
            if a.get(field) != b.get(field):
                failures.append(f"{key}: {field} differs")
        errors = {}
        for field, tolerance in NUMERICS.items():
            if field in a or field in b:
                difference = _error(a.get(field), b.get(field))
                errors[field] = difference if difference < float("inf") else None
                if difference > tolerance:
                    failures.append(f"{key}: {field} exceeds {tolerance:g} Ha")
        parity.append(dict(case=key[0], repeat=key[1], stage=key[2], absolute_errors=errors))
    phases = sorted({(key[0], key[2]) for key in left.keys() & right.keys()})
    for case, stage in phases:
        for state in ("first_case_call", "warm_fresh_objects"):
            keys = [key for key in left.keys() & right.keys() if (key[0], key[2]) == (case, stage)
                    and left[key].get("cache_state") == right[key].get("cache_state") == state]
            if not keys:
                continue
            first = statistics.median(left[key]["seconds"] for key in keys)
            second = statistics.median(right[key]["seconds"] for key in keys)
            eligible = all(left[key].get('scientific_pass') is True
                           and right[key].get('scientific_pass') is True for key in keys)
            timings.append(dict(case=case, stage=stage, cache_state=state,
                                before_seconds=first, after_seconds=second,
                                eligible_for_speed_ratio=eligible,
                                time_ratio_after_before=second/first if eligible and first > 0 else None,
                                speedup_before_after=first/second if eligible and second > 0 else None,
                                before_peak_rss_bytes=max(left[key]["peak_rss_bytes"] for key in keys),
                                after_peak_rss_bytes=max(right[key]["peak_rss_bytes"] for key in keys)))
    return dict(paired_numerical_pass=not failures, failures=failures, parity=parity, timings=timings,
                note="Timing ratios are descriptive; RSS is cumulative process peak. Capability guards are not response calculations.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh comparison output path")
    result = compare(args.before, args.after)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("paired_numerical_pass", "failures")}))
    return int(not result["paired_numerical_pass"])


if __name__ == "__main__":
    raise SystemExit(main())
