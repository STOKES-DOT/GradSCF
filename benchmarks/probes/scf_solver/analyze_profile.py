"""Summarize one or more saved profile directories; standard library only.

No calculations are rerun. Inclusive Python/CPU-thread durations are not wall
partitions. CSV rows must not be summed across nesting, threads, or trace files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import pstats


def interval_union(intervals):
    total, end = 0., -math.inf
    for left, right in sorted(intervals):
        total += max(0., right-max(left, end))
        end = max(end, right)
    return total


def read_jsonl(path, warnings):
    rows = []
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object")
                rows.append(row)
            except ValueError as error:
                warnings.append(f"{path.name}:{number}: {error}")
    return rows


def write_csv(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def profile_stats(path):
    stats = pstats.Stats(str(path))
    rows = [dict(filename=f, line=line, function=name, primitive_calls=cc,
                 total_calls=nc, self_seconds=tt, cumulative_seconds=ct)
            for (f, line, name), (cc, nc, tt, ct, _) in stats.stats.items()]
    return dict(file=path.name, total_self_seconds=stats.total_tt,
        backend_compile_and_load=[r for r in rows if r["function"] == "backend_compile_and_load"],
        top_self=sorted(rows, key=lambda r: r["self_seconds"], reverse=True)[:20],
        top_cumulative=sorted(rows, key=lambda r: r["cumulative_seconds"], reverse=True)[:20],
        note="Python calls in profiled thread; not unique compilations. Cumulative parent/child times overlap; no sum is formed.")


def trace_stats(path, root):
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    events = data.get("traceEvents") if isinstance(data, dict) else data
    if not isinstance(events, list):
        raise ValueError("Chrome traceEvents must be a list")
    groups, modules, names = {}, defaultdict(list), {}
    low, high, valid, rejected = math.inf, -math.inf, 0, 0
    phases = Counter()
    for event in events:
        if not isinstance(event, dict):
            rejected += 1
            continue
        args = event.get("args", {})
        args = args if isinstance(args, dict) else {}
        phases[event.get("ph", "missing")] += 1
        if event.get("ph") == "M" and event.get("name") == "thread_name":
            names[(event.get("pid"), event.get("tid"))] = args.get("name")
        if event.get("ph") != "X":
            continue
        try:
            start, duration = float(event["ts"]), float(event["dur"])
            finish = start+duration
            if not all(math.isfinite(v) for v in (start, duration, finish)) or duration < 0:
                raise ValueError("Invalid complete-event interval")
        except (KeyError, TypeError, ValueError):
            rejected += 1
            continue
        valid += 1
        low, high = min(low, start), max(high, finish)
        # Runtime identifiers distinguish these from compiler/module annotations.
        if not all(key in args for key in ("hlo_module", "hlo_op", "run_id", "device_ordinal")):
            continue
        if "pid" not in event or "tid" not in event:
            continue
        module, op = str(args["hlo_module"]), str(args["hlo_op"])
        key = (module, op)
        row = groups.setdefault(key, dict(hlo_module=module, hlo_op=op, calls=0,
            inclusive_cpu_thread_seconds=0., run_ids=set(), event_names=set()))
        row["calls"] += 1
        row["inclusive_cpu_thread_seconds"] += duration/1e6
        row["run_ids"].add(str(args["run_id"]))
        row["event_names"].add(str(event.get("name", "")))
        modules[(module, event.get("pid"), event.get("tid"))].append((start, finish))
    operations = []
    for row in groups.values():
        row["distinct_run_ids"] = len(row.pop("run_ids"))
        row["event_names"] = sorted(row["event_names"])
        operations.append(row)
    thread_modules = [dict(hlo_module=module, pid=pid, tid=tid,
        thread_name=names.get((pid, tid)), calls=len(spans),
        inclusive_cpu_thread_seconds=sum(b-a for a, b in spans)/1e6,
        same_thread_interval_union_seconds=interval_union(spans)/1e6,
        start_us=min(a for a, _ in spans), end_us=max(b for _, b in spans))
        for (module, pid, tid), spans in modules.items()]
    return dict(file=str(path.relative_to(root)), total_events=len(events), phases=dict(phases),
        valid_complete_events=valid, rejected_complete_events=rejected,
        complete_event_start_us=low if valid else None,
        complete_event_end_us=high if valid else None,
        complete_event_range_seconds=(high-low)/1e6 if valid else None,
        runtime_hlo_candidate_events=sum(r["calls"] for r in operations),
        operations=sorted(operations, key=lambda r: r["inclusive_cpu_thread_seconds"], reverse=True),
        modules_by_thread=thread_modules,
        note="Chrome ts/dur are microseconds, regardless of displayTimeUnit. HLO candidates require module/op/run/device args. Inclusive CPU thread time is not wall time; same-thread module union removes nesting but cannot be summed across threads/modules/files. Range includes all complete events, including waits; no physical attribution or percentages.")


def analyze(root):
    root = Path(root).resolve()
    warnings, analysis = [], {"directory": str(root)}
    for name in ("summary", "runtime", "timeout", "failure"):
        path = root/(name+".json")
        if path.exists():
            try:
                analysis[name] = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                warnings.append(f"{path.name}: {error}")
    events = read_jsonl(root/"events.jsonl", warnings)
    ended = [r for r in events if r.get("event") == "stage_end"]
    spans = [dict(r, start_seconds=r["elapsed_seconds"]-r["seconds"],
                  end_seconds=r["elapsed_seconds"])
             for r in ended if r.get("timing_kind") == "synchronized_inclusive" and not r.get("tracing")]
    top = [(r["start_seconds"], r["end_seconds"]) for r in spans if r.get("parent_id") is None]
    analysis["stages"] = dict(spans=spans, excluded_tracing_calls=len(ended)-len(spans),
        unmatched_begins=len({r["call_id"] for r in events if r.get("event") == "stage_begin"}
                             -{r["call_id"] for r in ended}),
        top_level_interval_union_seconds=interval_union(top),
        note="Synchronized inclusive spans include compilation, waits and synchronization. Nested spans are not additive; Python call counts are not SCF cycles.")
    write_csv(root/"stages.csv", spans, ["call_id", "parent_id", "depth", "stage", "status",
        "start_seconds", "end_seconds", "seconds", "timing_kind"])
    kernels = read_jsonl(root/"kernels.jsonl", warnings)
    analysis["kernels"] = kernels
    write_csv(root/"kernels.csv", [r for r in kernels if r.get("event") == "kernel_phase"],
        ["name", "phase", "sample", "seconds", "elapsed_seconds", "remaining_seconds"])
    analysis["cprofile"] = []
    for path in sorted(set(root.glob("*.prof")) | set(root.glob("*.pstats"))):
        try:
            analysis["cprofile"].append(profile_stats(path))
        except (OSError, ValueError, EOFError, TypeError) as error:
            warnings.append(f"{path.name}: {error}")
    analysis["traces"] = []
    hlo_rows, module_rows = [], []
    for path in sorted(p for p in root.rglob("*.json.gz") if "trace" in p.name):
        try:
            trace = trace_stats(path, root)
            analysis["traces"].append(trace)
            hlo_rows.extend(dict(file=trace["file"], **r) for r in trace["operations"])
            module_rows.extend(dict(file=trace["file"], **r) for r in trace["modules_by_thread"])
        except (OSError, ValueError, EOFError, TypeError) as error:
            warnings.append(f"{path.relative_to(root)}: {error}")
    write_csv(root/"trace_hlo.csv", hlo_rows, ["file", "hlo_module", "hlo_op", "calls",
        "inclusive_cpu_thread_seconds", "distinct_run_ids", "event_names"])
    write_csv(root/"trace_modules.csv", module_rows, ["file", "hlo_module", "pid", "tid", "thread_name",
        "calls", "inclusive_cpu_thread_seconds", "same_thread_interval_union_seconds", "start_us", "end_us"])
    analysis["warnings"] = warnings
    (root/"analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False)+"\n")
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    for directory in parser.parse_args().directories:
        result = analyze(directory)
        print(json.dumps(dict(directory=str(directory), trace_files=len(result["traces"]), warnings=result["warnings"])))
