"""Stdlib-only GW-family evidence summary; never imports numerical packages."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(rows, key, warm=True):
    values = [r[key] for r in rows if key in r and (r.get("repeat", 0) > 0 if warm else r.get("repeat") == 0)]
    return dict(n=len(values), samples=values, median=statistics.median(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None,
                dispersion="unknown" if len(values) < 2 else "observed_range")


def load(p):
    rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.is_file() else []
    manifest = p.with_suffix(".manifest.json")
    meta = json.loads(manifest.read_text()) if manifest.is_file() else {}
    meta.update(next((r for r in rows if r.get("stage") == "compile"), {}))
    samples = [r for r in rows if "repeat" in r]
    first = next(iter(samples), {})
    identity = dict(input_sha256=meta.get("input_sha256", first.get("input_sha256", meta.get("case_input_sha256"))), dtype=meta.get("dtype", first.get("dtype")),
                    scope=first.get("timing_scope", "molecular end_to_end"), method=first.get("method", first.get("kind", "g0w0")),
                    representation=first.get("integral_representation", "fixed_array" if "kind" in first else "ri"), device=first.get("device"),
                    threads=meta.get("threads", first.get("threads")), affinity=meta.get("affinity", first.get("logical_cpus")),
                    dimensions={k: first[k] for k in ("nao", "naux", "nmo", "nw") if k in first})
    if first.get("kind") == "scgw": identity["dimensions"].pop("nmo", None)
    if first.get("kind") == "charge": identity["dimensions"].pop("naux", None)
    memory = {k: r[k] for r in rows for k in ("cpu_peak_rss_bytes", "process_peak_rss_bytes", "xla_temporary_bytes", "xla_output_bytes") if k in r}
    return dict(rows=rows, samples=samples, identity=identity, memory=memory, sha256=sha(p) if p.is_file() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parents[3]/"reports/gw_family_memory/20261030-ea646bf")
    out = parser.parse_args().report_dir.resolve()
    raw, verification, traces = out/"raw", {}, []
    exits = {p.stem: int(p.read_text().strip()) for p in sorted(raw.glob("*.exit"))}
    for p in sorted(raw.glob("*.xml")):
        root = ET.parse(p).getroot()
        v = {k: sum(int(s.get(k, 0)) for s in root.iter("testsuite")) for k in ("tests", "failures", "errors", "skipped")}
        v.update(exit_code=exits.get(p.stem), seconds=sum(float(s.get("time", 0)) for s in root.iter("testsuite")), sha256=sha(p))
        v["status"] = "failed" if v["failures"] or v["errors"] or v["exit_code"] not in (None, 0) else "passed" if v["exit_code"] == 0 and not v["skipped"] else "incomplete_or_skipped"
        verification[p.name] = v
        for t in root.iter("testcase"):
            properties = {q.get("name"): q.get("value") for q in t.iter("property")}
            if properties:
                traces.append(dict(source=p.name, test=t.get("name"), properties=properties, source_status=v["status"], test_status="failed" if t.find("failure") is not None or t.find("error") is not None else "passed", backend_compile_count=None))
    def oracle(*names):
        return next((n for n in names if n in verification), None)
    gates = dict(qsgw=[oracle("final-qsstream.xml", "final-test_qsgw_streaming.xml", "accepted-test_qsgw_streaming.xml")],
                 scgw=[oracle("final-screentile.xml", "final-test_scgw_working_arrays.xml", "accepted-test_scgw_working_arrays.xml"), oracle("accepted-test_gw_matsubara.xml")],
                 charge=[oracle("final-scgw-response.xml", "accepted-test_gw_scgw_response.xml")],
                 evgw=[oracle("final-evprep.xml", "final-test_evgw_preparation.xml", "accepted-test_evgw_preparation.xml"), oracle("accepted-test_gw_evgw.xml")],
                 g0w0=[oracle("conventional-final.xml", "final-conventional.xml", "final-test_conventional.xml", "accepted-test_conventional.xml")])
    pairs = [(k+"-cpu1", "baseline-"+k+"-cpu1", "candidate-"+k+"-cpu1", k) for k in ("qsgw", "scgw", "charge")]
    pairs += [("ch2o-"+k, "baseline-ch2o-"+k+"-cpu32", "candidate-ch2o-"+k+"-cpu32", "evgw") for k in ("evgw0", "evgw")]
    pairs += [("phenanthrene", "phenanthrene-g0w0-ri-cpu32", "candidate-phenanthrene-g0w0", "g0w0"), ("water-conventional", "baseline-water-conventional", "candidate-water-conventional", "g0w0")]
    comparisons = []
    for label, bname, cname, gate in pairs:
        b, c = load(raw/(bname+".jsonl")), load(raw/(cname+".jsonl"))
        reason, delta, residual = [], None, None
        if not b["samples"] or not c["samples"]: reason.append("missing_measurements")
        if exits.get(bname) != 0 or exits.get(cname) != 0: reason.append("missing_or_nonzero_exit")
        if b["identity"] != c["identity"]: reason.append("unmatched_input_scope_device_threads_or_dimensions")
        if any(b["identity"][k] is None for k in ("input_sha256", "dtype", "device", "threads", "affinity")): reason.append("incomplete_pairing_metadata")
        if any(n is None or verification[n]["status"] != "passed" for n in gates[gate]): reason.append("oracle_not_passed")
        samples = b["samples"]+c["samples"]
        if samples and "qp_energy_hartree" in samples[0]:
            delta = max((abs(x-y) for u, v in itertools.product(b["samples"], c["samples"]) for x, y in zip(u["qp_energy_hartree"], v["qp_energy_hartree"])), default=None)
            residual = max(abs(x) for r in samples for x in r["qp_residual_hartree"])
            if delta is None or delta > 1e-8 or residual >= (1e-8 if gate == "evgw" else 1e-10): reason.append("energy_or_residual_gate")
            if not all(r.get("scientific_pass") and all(r["qp_branch_valid"]) and all(math.isfinite(z) and z > 0 for z in r["qp_weight"]) and all(math.isfinite(x) for k in ("qp_energy_hartree", "qp_residual_hartree") for x in r[k]) for r in samples): reason.append("root_certificate_gate")
            if len({tuple(r["qp_candidate_count"]) for r in samples}) != 1 or len({len(r["qp_energy_hartree"]) for r in samples}) != 1: reason.append("changed_candidate_counts_or_root_coverage")
        elif samples and not all(r.get("finite") for r in samples): reason.append("nonfinite_kernel")
        for metric in (("wall_seconds", "gw_seconds") if gate in ("evgw", "g0w0") else ("wall_seconds",)):
            bs, cs = stats(b["samples"], metric), stats(c["samples"], metric)
            ratio = cs["median"]/bs["median"] if not reason and bs["n"] and cs["n"] else None
            comparisons.append(dict(case=label, metric=metric, timing_scope=b["identity"]["scope"], baseline_file=bname+".jsonl", candidate_file=cname+".jsonl", status="paired" if ratio is not None else "unpaired", reason=reason,
                                    after_over_before=ratio, faster=ratio < 1 if ratio is not None else None, baseline=bs, candidate=cs, baseline_first=stats(b["samples"], metric, False), candidate_first=stats(c["samples"], metric, False),
                                    max_qp_change_hartree=delta, max_root_residual_hartree=residual, oracle_files=gates[gate], identity=b["identity"], candidate_identity=c["identity"], baseline_memory=b["memory"], candidate_memory=c["memory"]))
    def write(name, obj):
        (out/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    write("verification.json", dict(note="Each XML is an execution; repeated/overlapping gates are not unique test totals.", suites=verification, process_exit_codes=exits))
    preferred = {k: oracle(*names) for k, names in dict(evprep=("final-evprep.xml", "final-test_evgw_preparation.xml"), qsstream=("final-qsstream.xml", "final-test_qsgw_streaming.xml")).items()}
    write("compile_counts.json", dict(metric="Python tracing counts from JUnit properties, not backend compile counts", backend_compile_count=None, preferred_sources=preferred, preferred_records=[t for t in traces if t["source"] in preferred.values()], records=traces))
    with (out/"measurements.jsonl").open("w") as stream:
        for p in sorted(raw.glob("*.jsonl")):
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if line.strip(): stream.write(json.dumps(dict(source="raw/"+p.name, line=i, sha256=sha(p), measurement=json.loads(line)), allow_nan=False)+"\n")
    with (out/"before_after.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in comparisons)
    timing = [f'| {r["case"]}/{r["metric"]} | {r["baseline"]["median"]} (n={r["baseline"]["n"]}) | {r["candidate"]["median"]} (n={r["candidate"]["n"]}) | {r["after_over_before"]} | {", ".join(r["reason"]) or "paired"} |' for r in comparisons]
    checks = [f'| {n} | {v["tests"]} | {v["failures"]} | {v["errors"]} | {v["skipped"]} | {v["exit_code"]} | {v["status"]} |' for n, v in verification.items()]
    trace_lines = [f'- `{t["source"]}`: '+", ".join(f'{k}={v}' for k, v in t["properties"].items()) for t in traces if t["source"] in preferred.values()]
    report = f'''# GW-family memory/execution evidence (draft)

Baseline `ea646bf`, branch `perf/gw-family-memory-20261030`. Only this campaign's raw evidence is summarized. Missing/failed/unmatched runs have no ratio.

## Methods and performance

CPU matrix probes time synchronized fixed-input qsGW self-energy, scGW screening factorization, or chemical-potential kernels, not SCF or complete workflows. Molecular rows include SCF/end-to-end and separate GW times. First execution is separate from warm median/range. qsGW/screening lowering/compilation is outside execution timing; chemical-potential first execution includes internal JIT and its separate lower/compile times are unknown (`null`). Its warm calls restart from the same unconverged chemical-potential guess. CSV retains every warm sample, range and count; n=1 dispersion is unknown. Pairing requires equal input hash, scope, dtype, device, thread environment/affinity and dimensions, zero process exits and this campaign's oracle gates. Matrix norms/finite flags are auxiliary; scientific validity depends on those oracle tests. QP pairs also require <=1e-8 Ha energy change, original residual limits, branches and positive finite Z. Ratio is **after/before**; <1 means faster.

The principal molecular evGW0/evGW workload is CH2O/Cartesian def2-SVP (40 AOs, 8 occupied orbitals), `weigend`, full G/W, six QPs (zero-based 5–10), eta=0.001 Ha and 32 frequency nodes; geometry comes from the repository TDDFT fixture. CPU32 uses the recorded affinity and OpenMP/BLAS/native thread settings, not a claim of 32 physical cores.

| Case/metric (seconds) | Baseline median | Candidate median | After/before | Pairing state |
|---|---:|---:|---:|---|
{chr(10).join(timing)}

Phenanthrene baseline: actual 260 Cartesian AOs, 974 auxiliary functions, 47 occupied orbitals, full G/W and six QPs; first/warm total 199.773/191.863 s, process ru_maxrss 8,509,132,800 bytes. One warm sample does not establish dispersion. This is angular phenanthrene (C14H10). RSS is whole-process high water, distinct from XLA temporary estimates and GPU allocator/pool counters retained in raw data. No prior-campaign speed ratio is reused.

## Verification and compilation

| XML execution | Tests | Failures | Errors | Skipped | Exit | State |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(checks)}

These are separate, potentially overlapping executions, not an additive unique-test total. Missing final SCGW/NNAO or performance artifacts remain pending. `compile_counts.json` preserves cold/new-object/updated-input Python **tracing** properties (including failed attempts); backend compilation counts are unknown (`null`). Final evprep/qsstream properties are preferred evidence when present; timings do not infer backend counts.

{chr(10).join(trace_lines) or 'Final tracing properties are pending retrieval.'}

The initial combined process exited 139 after 30 completed tests. The same old water qsGW case passed in an isolated CPU1 process before the stable-JIT change. A separate cache regression reproduced tracing 4→2; that does not establish the segfault's root cause. Original failure logs/exits remain under raw/.

## Capability boundaries

Explicit `GW.conventional` uses true four-center Coulomb data/actions without AO-pair spectral diagonalization. Restricted G0W0/evGW0/evGW conventional paths are covered by the present oracle tests; existing factor/BSE entry points still have consumers. Conventional qsGW/scGW/U-GW, cache-free conventional screening and conventional BSE export are not implemented. scGW measurements cover screening factorization/chemical kernels; full Pi/W/time output caching and complete-workflow acceleration are not established. NNAO evidence is a synthetic H2 radial RI→implicit SCF→QP training fixture, not MACE training, forces or full RI nuclear/auxiliary-metric AD.

## Provenance and regeneration

`measurements.jsonl` preserves every current raw JSONL record with source, line and SHA256. `before_after.csv` links paired files, identities, actual samples, separate RSS/XLA observations and oracle filenames. `verification.json` lists JUnit failure/skip counts and all process exits, including failures without XML. `compile_counts.json` reads raw JUnit properties (final-evprep/final-qsstream and original filenames). Raw source/manifests/logs are preserved and are not submitted by this script.

Run `python benchmarks/probes/g0w0_memory/family_analysis.py` after retrieving remaining raw files. This script uses only the standard library and launches no calculations. This draft does not close pending source/float32-promotion, SCGW/NNAO or performance acceptance.
'''
    (out/"report.md").write_text(report)
    print(f"Wrote five summaries: {len(verification)} XML executions, {len(comparisons)} timing rows; backend_compile_count=null")


if __name__ == "__main__":
    main()
