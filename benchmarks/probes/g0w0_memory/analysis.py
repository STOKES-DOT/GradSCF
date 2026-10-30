"""Summarize saved G0W0 evidence using only the Python standard library.

This does not import GradSCF/JAX or launch calculations. Run again after adding
accepted-candidate-*.jsonl and accepted-*-narrow.xml/.exit to raw/results.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


HARTREE_TO_EV = 27.211386245988
QP_GATE_HARTREE = 1e-8
ROOT_GATE_HARTREE = 1e-10
EXTERNAL_GATE_EV = 1e-3
DEFAULT_REPORT = (Path(__file__).resolve().parents[3]
                  / "reports/g0w0_memory/20261030-a10b865")


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    values = list(values)
    if not values:
        return None
    return dict(n=len(values), samples=values, median=statistics.median(values),
                min=min(values), max=max(values),
                dispersion="unknown_single_sample" if len(values) == 1 else "observed_range")


def max_difference(before, after, field):
    """Compare all completed repeats; a missing field is unknown, never zero."""
    left = [row[field] for row in before if row.get(field) is not None]
    right = [row[field] for row in after if row.get(field) is not None]
    if not left or not right:
        return None
    differences = []
    for a, b in itertools.product(left, right):
        if isinstance(a, list):
            if len(a) != len(b):
                raise ValueError(f"Different {field} dimensions")
            differences.extend(abs(x-y) for x, y in zip(a, b))
        else:
            differences.append(abs(a-b))
    return max(differences)


def scientific_comparison(before, after, molecular):
    if not molecular:
        fields = ("value_real", "value_imag", "parameter_gradient")
        result = {"max_abs_"+field+"_difference": max_difference(before, after, field)
                  for field in fields}
        result["sigma_gate_atol_hartree"] = 1e-10
        result["sigma_gate_rtol"] = 1e-9
        result["pass"] = all(row.get("status") == "ok" for row in before+after) and all(
            delta is None or delta <= 1e-10 + 1e-9*max(
                abs(row[field]) for row in before if row.get(field) is not None)
            for field in fields
            for delta in [result["max_abs_"+field+"_difference"]])
        return result
    rows = before + after
    residual = max(abs(v) for row in rows for v in row["qp_residual_hartree"])
    energy_delta = max_difference(before, after, "qp_energy_hartree")
    candidates = {tuple(row["qp_candidate_count"]) for row in rows}
    branches = {tuple(row["qp_branch_valid"]) for row in rows}
    weights = [v for row in rows for v in row["qp_weight"]]
    nroots = {len(row["qp_energy_hartree"]) for row in rows}
    all_branches_valid = all(all(flags) for flags in branches)
    passed = (all(row.get("scientific_pass") and row["status"] == "ok" for row in rows)
              and energy_delta <= QP_GATE_HARTREE and residual < ROOT_GATE_HARTREE
              and len(candidates) == len(branches) == len(nroots) == 1
              and all_branches_valid
              and all(math.isfinite(z) and z > 0 for z in weights))
    return dict({"pass": passed}, qp_gate_hartree=QP_GATE_HARTREE,
                max_abs_qp_difference_hartree=energy_delta,
                max_abs_qp_residual_hartree=residual,
                max_abs_residual_difference_hartree=max_difference(
                    before, after, "qp_residual_hartree"),
                max_abs_z_difference=max_difference(before, after, "qp_weight"),
                min_z=min(weights), candidate_counts_identical=len(candidates) == 1,
                branch_flags_identical=len(branches) == 1, all_branches_valid=all_branches_valid,
                nroots=next(iter(nroots)),
                coverage_evidence="run.py scientific_pass includes converged/computed masks; masks not serialized separately")


def xml_summary(path):
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    result = {field: sum(int(s.get(field, 0)) for s in suites)
              for field in ("tests", "failures", "errors", "skipped")}
    result["seconds"] = sum(float(s.get("time", 0)) for s in suites)
    result["exit_code"] = (path.with_suffix(".exit").read_text().strip()
                           if path.with_suffix(".exit").is_file() else None)
    return result


def build_comparison(raw, name, baseline, candidate_names, molecular=True):
    candidate = next((raw/n for n in candidate_names if (raw/n).is_file()), None)
    if candidate is None:
        raise FileNotFoundError(candidate_names)
    baseline = raw/baseline
    before_all, after_all = read_rows(baseline), read_rows(candidate)
    stage = "end_to_end" if molecular else "warm_execution"
    before = [r for r in before_all if r.get("stage") == stage]
    after = [r for r in after_all if r.get("stage") == stage]
    warm_before = [r for r in before if r.get("repeat", 0) > 0]
    warm_after = [r for r in after if r.get("repeat", 0) > 0]
    bmanifest = read_json(baseline.with_suffix(".manifest.json"))
    cmanifest = read_json(candidate.with_suffix(".manifest.json"))
    for key in ("dtype", "nw", "affinity", "threads"):
        if bmanifest[key] != cmanifest[key]:
            raise ValueError(f"{name}: unmatched {key}")
    if bmanifest.get("case_input_sha256") != cmanifest.get("case_input_sha256"):
        raise ValueError(f"{name}: unmatched input file")
    for key in (("nao", "naux", "nw") if molecular else ("naux", "nmo", "nocc", "nvir")):
        if len({r[key] for r in before+after}) != 1:
            raise ValueError(f"{name}: unmatched {key}")
    metrics = {}
    for key in (("wall_seconds", "gw_seconds", "scf_seconds", "transfer_seconds")
                if molecular else ("wall_seconds",)):
        b = summary(r[key] for r in warm_before if key in r)
        c = summary(r[key] for r in warm_after if key in r)
        if b and c:
            metrics[key] = dict(before=b, after=c, before_over_after=b["median"]/c["median"])
    rss_before = next((r["cpu_peak_rss_bytes"] for r in before_all
                       if r.get("stage") == "process_peak"), None)
    rss_after = next((r["cpu_peak_rss_bytes"] for r in after_all
                      if r.get("stage") == "process_peak"), None)
    rss_notes = "Linux ru_maxrss, whole fresh process including imports/preparation/compilation/all repeats"
    if name == "benzene-def2-svp-cpu64":
        rss_before = read_json(raw/"parent-benzene-stop.json")["peak_rss_before_stop_bytes"]
        rss_notes = ("Before: Linux VmHWM at intentional budget stop after cold+one complete warm; "
                     "may include interrupted repeat start. After: whole-process Linux ru_maxrss. "
                     "Different observation windows; not per-iteration active peaks; ratio unset.")
    return dict(id=name, molecular=molecular, baseline_file=baseline.name,
                candidate_file=candidate.name, baseline_sha256=digest(baseline),
                candidate_sha256=digest(candidate),
                baseline_manifest_sha256=digest(baseline.with_suffix(".manifest.json")),
                candidate_manifest_sha256=digest(candidate.with_suffix(".manifest.json")),
                case_input_sha256=bmanifest.get("case_input_sha256"), dimensions={
                    k: before[0][k] for k in ("nao", "naux", "nw", "nmo", "nocc", "nvir") if k in before[0]},
                dtype=bmanifest["dtype"], cpu_logical=len(bmanifest["affinity"]),
                cpu_physical=len({(r["physical_package"], r["core_id"])
                                  for r in bmanifest.get("cpu_topology", [])}) or None,
                threads=bmanifest["threads"], metrics=metrics,
                before_first=before_all[0], after_first=after_all[0],
                process_rss_before_bytes=rss_before, process_rss_after_bytes=rss_after,
                process_rss_notes=rss_notes, gpu_before=bmanifest.get("gpu_memory"),
                gpu_after=cmanifest.get("gpu_memory"),
                xla_compile_before=next((r for r in before_all if r.get("stage") == "compile"), None),
                xla_compile_after=next((r for r in after_all if r.get("stage") == "compile"), None),
                science=scientific_comparison(before, after, molecular))


def build_references(raw):
    grad = read_rows(raw/"reference-candidate-water-cpu1.jsonl")
    grad = [r for r in grad if r.get("stage") == "end_to_end"]
    output = dict(hartree_to_ev=HARTREE_TO_EV, qp_gate_ev=EXTERNAL_GATE_EV,
                  gradscf_source="reference-candidate-water-cpu1.jsonl",
                  case="cases/water-reference.json", time_ratio=None,
                  time_ratio_reason="Different frequency algorithms/Sigma-grid work and upstream root diagnostics",
                  shared_input_diagnostic=read_json(raw/"pyscf-alignment-matched.json"),
                  molgw_source_provenance=read_json(raw/"molgw-provenance.json"),
                  references=[])
    for name, definition in (
            ("pyscf-water-reference", "Initial exact bare exchange; definition mismatch, excluded"),
            ("pyscf-water-reference-ri-exchange", "Matched RI bare exchange: explicit vhf_df=True"),
            ("molgw-water-reference", "Initial missing libiomp5 runtime; failed, excluded"),
            ("molgw-water-reference-runtime", "Missing DEF2-SVP assets; failed, excluded"),
            ("molgw-water-reference-common-basis", "Matched Cartesian orbital/auxiliary basis DATA; valid")):
        path = raw/name/"measurements.jsonl"
        rows = read_rows(path)
        good = [r for r in rows if r["status"] == "ok" and "qp_energy_hartree" in r]
        delta = max_difference(grad, good, "qp_energy_hartree")
        delta_ev = None if delta is None else delta*HARTREE_TO_EV
        valid = bool(good) and "pyscf-water-reference" != name
        output["references"].append(dict(
            id=name, source=str(path.relative_to(raw)), sha256=digest(path),
            definition=definition, successful_repeats=len(good),
            measured_returncodes=sorted({r["returncode"] for r in rows if "returncode" in r}),
            max_abs_qp_difference_hartree=delta, max_abs_qp_difference_ev=delta_ev,
            numerical_gate_pass=delta_ev is not None and delta_ev <= EXTERNAL_GATE_EV,
            accepted_definition=valid, time_ratio=None,
            warm_wall_seconds=summary(r["wall_seconds"] for r in good if r["repeat"] > 0),
            qp_energy_hartree=good[0]["qp_energy_hartree"] if good else None,
            root_residual_hartree=None,
            residual_scope="Upstream public result does not expose the same six-candidate residual/branch certificate",
            memory=read_json(raw/name/"memory.json")))
    return output


def fmt(stat):
    if stat["n"] == 1:
        return f'{stat["median"]:.6g} (n=1; dispersion unknown)'
    return f'{stat["median"]:.6g} [{stat["min"]:.6g}, {stat["max"]:.6g}] (n={stat["n"]})'


def write_report(report, comparisons, references, evidence, acceptance):
    by_id = {c["id"]: c for c in comparisons}
    def performance_row(c, metric, label):
        m = c["metrics"][metric]
        return f'| {label} | {fmt(m["before"])} | {fmt(m["after"])} | {m["before_over_after"]:.4g} |'
    performance = []
    labels = (("water-def2-svp-cpu1", "Water/def2-SVP CPU1"),
              ("water-def2-tzvp-cpu64", "Water/def2-TZVP CPU64"),
              ("benzene-def2-svp-cpu64", "Benzene/def2-SVP CPU64"),
              ("water-def2-svp-gpu", "Water/def2-SVP GPU"))
    for name, label in labels:
        for metric, stage in (("gw_seconds", "GW"), ("wall_seconds", "end-to-end")):
            performance.append(performance_row(by_id[name], metric, label+" "+stage))
    micro = []
    for name in ("residue128-cpu1-projected", "residue512-cpu1", "residue-ad128-cpu1",
                 "residue128-gpu-serial", "residue128-gpu-tile4", "residue128-gpu-tile8"):
        c = by_id[name]
        micro.append(performance_row(c, "wall_seconds", name))
    science = []
    for name, label in labels:
        s = by_id[name]["science"]
        science.append(f'| {label} | {s["max_abs_qp_difference_hartree"]:.3g} | '
                       f'{s["max_abs_qp_residual_hartree"]:.3g} | '
                       f'{s["max_abs_z_difference"]:.3g} | {s["min_z"]:.8g} | '
                       f'{"pass" if s["pass"] else "FAIL"} |')
    rss = []
    for name, label in labels:
        c = by_id[name]
        rss.append(f'| {label} | {c["process_rss_before_bytes"]:,} | '
                   f'{c["process_rss_after_bytes"]:,} |')
    xla = []
    for name in ("residue128-cpu1-projected", "residue512-cpu1", "residue-ad128-cpu1"):
        c = by_id[name]
        xla.append(f'| {name} | {c["xla_compile_before"]["xla_temp_bytes"]:,} | '
                   f'{c["xla_compile_after"]["xla_temp_bytes"]:,} | '
                   f'{c["process_rss_before_bytes"]:,} | {c["process_rss_after_bytes"]:,} |')
    refs = []
    for r in references["references"]:
        if r["accepted_definition"]:
            refs.append(f'| {r["id"]} | {r["max_abs_qp_difference_ev"]:.6g} | '
                        f'{fmt(r["warm_wall_seconds"])} | '
                        f'{"pass" if r["numerical_gate_pass"] else "FAIL"} |')
    sources = []
    for c in comparisons:
        sources.append(f'- `{c["id"]}`: `raw/results/{c["baseline_file"]}` and '
                       f'`raw/results/{c["candidate_file"]}`; adjacent `.manifest.json` files.')
    gate_lines = []
    for name in ("final-narrow.xml", "final-source-narrow.xml", "projected.xml", "geometry-compile.xml",
                 "ugw-ad.xml", "gmres-hvp.xml", "integration6.xml", "nnao-qp.xml", "final-gpu-array.xml"):
        e = evidence[name]
        gate_lines.append(f'- `{name}`: {e["tests"]-e["failures"]-e["errors"]-e["skipped"]} passed, '
                          f'{e["failures"]} failed, {e["errors"]} errors, {e["skipped"]} skipped; '
                          f'{e["seconds"]:.3f} s.')
    resumed_lines = []
    for name, e in evidence.items():
        if name.startswith(("accepted-", "resume-", "resumed-")):
            resumed_lines.append(f'- `{name}`: {e["tests"]-e["failures"]-e["errors"]-e["skipped"]} passed, '
                                 f'{e["failures"]} failed, {e["errors"]} errors, {e["skipped"]} skipped; '
                                 f'{e["seconds"]:.3f} s, exit {e["exit_code"]}.')
    if acceptance:
        accepted_pass = (acceptance["tests"] >= 35 and acceptance["failures"] == 0
                         and acceptance["errors"] == 0 and acceptance["skipped"] == 0
                         and acceptance["exit_code"] == "0")
        status = (f'The resumed corrected-source narrow gate is **{"passed" if accepted_pass else "not accepted"}**: '
                  f'`{acceptance["file"]}`, {acceptance["tests"]} tests, '
                  f'{acceptance["failures"]} failures, {acceptance["errors"]} errors, '
                  f'{acceptance["skipped"]} skipped, exit {acceptance["exit_code"]}; '
                  f'{acceptance["seconds"]:.3f} s.')
    else:
        status = ('The resumed corrected-source narrow gate is **pending**: no accepted narrow XML/exit pair '
                  'has been retrieved. Historical green gates below do not close this final gate.')
    gpu = by_id["water-def2-svp-gpu"]
    gpu_metrics = []
    for key in ("bytes_in_use", "peak_bytes_in_use", "pool_bytes", "peak_pool_bytes"):
        b = (gpu["gpu_before"] or [{}])[0].get("metrics", {}).get(key)
        a = (gpu["gpu_after"] or [{}])[0].get("metrics", {}).get(key)
        gpu_metrics.append(f'- `{key}`: {b} before, {a} after (bytes; null means unavailable).')
    text = f'''# Molecular G0W0 optimization: measured results

This report analyzes saved remote evidence from 2026-10-30. The baseline is
`a10b8655b99a07dfd53557deda4d6481718534a0`; the implementation/recovery commits
are `a799f0e`, `c79d488`, `be50190` and `28af7c8`. No merge or push is implied.
{status}

## Methods

All molecular probes use Cartesian Gaussian orbitals, RHF, true RI with the
`weigend` auxiliary basis, float64, full G and W orbital spaces, finite eta,
and the original nonlinear CD/QP equations. The response entry is explicitly
pinned to `resolvent_expansion=False`. Water/def2-SVP has 25 AOs, 85 auxiliary
dimensions, four requested QPs (zero-based orbitals 3–6), 32 imaginary nodes,
and eta=0.001 Ha. Water/def2-TZVP has **48 AOs**, 85 auxiliary dimensions,
six QPs, 64 nodes and eta=0.001 Ha. Benzene/def2-SVP has 120 AOs, 438
auxiliary dimensions, six QPs (18–23), 64 nodes, 21 occupied/99 virtual
orbitals and 2079 screening transitions. These distinct workloads do not
define a universal scaling exponent or an aggregate speed average.

SCF energy convergence is 1e-12 Ha and requested QP residuals must be below
1e-10 Ha, with convergence/computed coverage, valid branches and finite
positive Z. Exact rearrangements use atol=1e-10 Ha, rtol=1e-9; same-branch QP
changes must be at most 1e-8 Ha. Candidate counts, branches, residuals and Z
are compared as well as QP energies. No orbital window, node count, seed,
dtype or scientific tolerance is relaxed for a performance comparison.

Each variant starts in a fresh process with the compilation cache disabled.
Repeat 0 includes first-use compilation; warm repeats reuse compiled programs,
rebuild RI/SCF from a new facade, and never reuse a saved QP result. Timers
synchronize the whole JAX result. End-to-end includes molecular/native RI
preparation, SCF, explicit transfer (where applicable) and GW. Separate SCF,
transfer and GW timers are retained in the machine-readable records. The
per-calculation OS alarm is 6000 s; voluntary resource stops are separate.
Tables give warm medians, observed [min,max] and actual sample counts.

CPU hardware is Intel Xeon Gold 6230 @ 2.10 GHz; CPU64 is 64 logical CPUs
covering 32 physical package/core pairs, native threads=64 and
OpenMP/OpenBLAS/MKL threads=1. CPU1 uses one logical CPU and native thread.
GPU response uses `cuda:0` in the V100 allocation, with native RI/SCF on CPU
and 445200 estimated physical input-transfer bytes per water repeat.
JAX/jaxlib 0.10.0, NumPy 2.4.4, SciPy 1.17.1 and PySCF 2.13.0 were recorded.
The isolated native library SHA256 is
`c4756801f7cad8aa826760fb5d761ba7ea6f345606c17402401cc8f4ac6b384b`.
`environment-final.json` checked 27 source files without mismatches; recovery
comparison matched those files except the declared test-only budget change.

The retained molecular path applies target/G/W selection before AO2MO, using
`gradscf.ao2mo` ordered Qia/Qpm/Qmp vertices. Imaginary W uses the same ordered
B[m,p] vertex on both sides; residues retain separate left/right vertices.
Bare exchange retains all occupied reference orbitals. Each fixed imaginary
dielectric factor serves RHS blocks and both UGW frames. Projected residues
solve `[I-Pi(z)]x=Pi(z)r` and immediately contract `l^T x`, preserving finite
eta, static subtraction, half-pole weights, six candidate seeds, frequency
slopes and branch certificates. CPU sequential callbacks skip inactive poles;
checkpointing recomputes local dense intermediates in reverse AD. Existing
periodic q0/off-diagonal matrix contracts are retained.

Known residents, promoted factors, primal/tangent weighted factors, complex
solves, RHS/frequency batches and caller reservations use the common decimal-MB
budget. Small GPU batches retain the dense path when their entire working set
fits. Optional matrix-free screening uses the existing checked shared complex
GMRES; real factors act separately on real/imaginary vectors. Independent
`GW(mf).density_fit(auxbasis)`/`UGW(mf).density_fit(auxbasis)` prepares fixed-
geometry libcint RI without rerunning SCF and reconstructs F-h-J from the actual
SCF Coulomb provider when fits differ. Cache ownership/capacity checks precede
preparation and transfer. Full packed 3c supply and selected-vertex caches must
still fit; there is no out-of-core provider. Production native, SCF, CC and
periodic numerical kernels were not changed.

## Results

### Paired molecular timings

All values are seconds. Ratios are before median / after median, descriptive
for each stated case only.

| Workload/stage | Before median [range] | After median [range] | Ratio |
|---|---:|---:|---:|
{chr(10).join(performance)}

The water CPU1 pair is the **final** baseline/candidate series, replacing the
earlier projected-only snapshot (3.9762→0.5650 s end-to-end) in headline results.
Small CPU water already had an RPA fast path in the default facade; these
pinned direct-CD timings do not establish a default-facade speedup. Benzene's
2079-transition space naturally uses the direct large-system route.

Benzene baseline cold+one warm completed. The declared cost policy deliberately
stopped it after one complete warm; single-sample dispersion is unknown.
The candidate has three complete warm samples. The CPU8 benzene pilot was
aborted for resource replanning after 14m31s without a first QP set; the saved
343 s RSS observation was earlier than the stop. This is not a timeout, a
completed solve, or a usable timing ratio.

The GPU water GW medians are {gpu['metrics']['gw_seconds']['before']['median']*1000:.3f}→{gpu['metrics']['gw_seconds']['after']['median']*1000:.3f} ms: **no reliable GPU speed gain is
claimed**. End-to-end variation also contains CPU preparation/SCF. Cold
observations and every warm sample are in `before_after.csv` and
`measurements.jsonl`; they are not mixed into warm medians.

### Scientific gates

Maxima below cover all completed repeats in each paired dataset; candidate
counts and branch flags are identical, and all serialized scientific gates pass.
QP residuals are the nonlinear self-energy root residuals. Molecular records
do not serialize separate Sigma components, so no unmeasured component error
is assigned. Synthetic Sigma products and VJPs are checked separately below.

| Case | Max QP change (Ha) | Max root residual (Ha) | Max Z change | Min Z | Gate |
|---|---:|---:|---:|---:|---|
{chr(10).join(science)}

### Memory measurements and kernel experiments

These are **actual Linux process RSS high-water observations**, in bytes,
including imports, preparation, compilation, allocator retention and repeats.
They are not the sum of live GW tensors or per-iteration active peaks.

| Molecular process | Before high water (bytes) | After high water (bytes) |
|---|---:|---:|
{chr(10).join(rss)}

For benzene, before is VmHWM at the voluntary budget stop, possibly including
the start of an interrupted optional repeat; after is complete-process
ru_maxrss. The observation windows differ, so no per-iteration memory ratio
is inferred from 69647646720 versus {by_id['benzene-def2-svp-cpu64']['process_rss_after_bytes']} bytes.

Synthetic residue probes use seeded (31) arrays, 32 MOs, 8 occupied/24 virtual
orbitals, eta=0.01 Ha and omega=0.86 Ha. They measure isolated Sigma products;
`residue-ad128` measures value plus a scalar-parameter VJP. Exact synthetic
Sigma real/imaginary and parameter-gradient differences pass the registered
atol=1e-10/rtol=1e-9 gate. This is kernel evidence, not a molecular total.

| Synthetic execution | Before median [range] (s) | Experiment median [range] (s) | Ratio |
|---|---:|---:|---:|
{chr(10).join(micro)}

| CPU synthetic probe | Before XLA temporary (bytes) | After XLA temporary (bytes) | Before process RSS (bytes) | After process RSS (bytes) |
|---|---:|---:|---:|---:|
{chr(10).join(xla)}

XLA `temp_size_in_bytes` is a compiler estimate for one compiled executable,
not a measured process peak. The smaller serial/projected and tile4/tile8 GPU
kernels were slower than the dense small-GPU kernel and were not retained for
budget-fitting batches. The production budget-dependent dense choice was
retained; kernel memory reductions are not used to claim a GPU total speedup.

Water GPU allocator counters (per-run manifest) are separate from CPU RSS:

{chr(10).join(gpu_metrics)}

Bytes-in-use counters describe allocator activity; pool counters describe
reserved capacity. The final snapshot is not called an active peak, and GPU
allocator counters are not a whole-process GPU-memory measurement.

### External numerical references

The independent reference case is water/def2-SVP/`weigend`, Cartesian, full G/W,
RHF, the same four QPs, eta=1e-5 Ha, 128 GradSCF/PySCF imaginary nodes and
1e-12 Ha SCF convergence. This tighter quadrature/eta case is separate from
the 32-node timing case. The external gate is 1e-3 eV; conversion is
1 Ha={HARTREE_TO_EV} eV. Both references generate their own integrals, SCF
orbitals and QPs from common geometry/basis DATA.

| Reference | Max independent QP error (eV) | Warm total seconds, median [range] | Numerical gate |
|---|---:|---:|---|
{chr(10).join(refs)}

The initial PySCF default exact bare exchange differed by up to 2.631 meV
(definition mismatch). Shared-input diagnostics matched S/h, DF Gram matrices
and mean-field energies, then explicit PySCF `vhf_df=True` reduced shared-input
QP disagreement to {references['shared_input_diagnostic']['matched_ri_exchange_error_ev']:.6g} eV.
That diagnostic is distinct from the independent results in the table. No
production kernel or scientific tolerance was changed for reference agreement.

Original MOLGW revision `b831818d7a845c36f295d036dc8ceef59daf9991` (3.4)
matched all 95 archived source files. Its unchanged executable SHA256 is
`bedee25b4d99844d9a59cb834a9a509dc0d276b48103aaa5a71da0ab105075bf`.
Initial runs failed because libiomp5 was missing and then DEF2-SVP assets were
absent. The valid run supplied the existing runtime only to the subprocess and
exported the common Cartesian orbital/auxiliary basis DATA into its input
directory. It used analytic poles and a 40001-point Sigma grid (step 0.00005 Ha),
fresh subprocess directories and no restart state or reference source patches.
These launch/input failures remain in the evidence and are excluded numerically.

All cross-software `time_ratio` fields remain **null**. MOLGW's analytic poles/
full Sigma grid and upstream root diagnostics differ from GradSCF's CD and
six-seed certification; PySCF also exposes different root diagnostics. Passing
the numerical gate does not equate the timed work or supply missing upstream
residual/branch certificates. Warm upstream totals are descriptive only.

### Derivatives and acceptance scope

{status}

The preceding final-source gate failed only
`test_dense_frequency_batch_charges_weighted_transition_factors`: its 60 MB
fixture could not hold newly declared primal/tangent weighted scratch.
Changing that **test budget to 100 MB** leaves all numerical tolerances intact.
The full traceback is retained, not replaced by static reasoning.
Historical remote XML evidence is listed individually (overlapping suites must
not be added into a distinct-test total):

{chr(10).join(gate_lines)}

Resumed-source evidence (separate runs, also overlapping):

{chr(10).join(resumed_lines) if resumed_lines else 'Pending retrieval.'}

The synthetic four-parameter H2 radial fixture rebuilds native contracted RI,
S/h, overlap transformation, density and orbitals, then differentiates an
implicit SCF solution and nonlinear QP loss. At fixed 0.74 Angstrom it checks
two central-FD steps (2e-4, 1e-4; gradient atol=3e-7, rtol=3e-5), same-shape
parameter rebinding/cache invalidation and one Adam update on CPU. It certifies
neither trained MACE weights, basis accuracy, forces, nor full GPU NNAO training.
Its auxiliary metric is fixed for radial-parameter AD and rejects a geometry
change. Existing full-integral H2 nuclear response/compile tests remain;
full RI nuclear/auxiliary-metric/whitening geometry AD was not added or
approximated by freezing dependencies. The current report makes no claim of
CH4/naphthalene completion, full-suite acceptance or universal system scaling.

## Provenance and reproduction

From the repository root, regenerate without importing numerical packages:

```sh
python benchmarks/probes/g0w0_memory/analysis.py --report-dir reports/g0w0_memory/20261030-a10b865
```

`analysis.py` automatically prefers accepted candidate benzene/GPU files when
present. Its outputs are this report, `before_after.csv`, normalized
`measurements.jsonl` (every raw row plus relative source path, line and SHA256),
and `reference_comparison.json`. CSV stores every warm sample, median/range,
cold observation, science fields, separate RSS/XLA/GPU metrics and exact raw-file
file hashes. All earlier experiments remain available in normalized raw rows;
they are not silently substituted into final pairs.

Paired/experimental sources relative to this report directory:

{chr(10).join(sources)}

External sources: `raw/results/reference-candidate-water-cpu1.jsonl`,
`raw/results/pyscf-water-reference-ri-exchange/measurements.jsonl`,
`raw/results/molgw-water-reference-common-basis/measurements.jsonl` and their
memory/input/stdout records; diagnostics `pyscf-alignment-matched.json`,
`molgw-provenance.json` and `molgw-water-reference-common-basis/basis-data.json`.
`raw/results/molgw-water-reference*` and `pyscf-water-reference` retain initial
failures/mismatched definitions. Resource-stop provenance is
`large_repeat_policy.json`, `parent-benzene-stop.json` and
`cpu8-benzene-aborted.json`; the 14m31s stop is recorded in `PLAN.md` whereas
the aborted JSON's RSS snapshot reports the earlier 343 s observation.
Environment/source provenance is `environment-final.json`, adjacent per-run
manifests, `recovery_byte_comparison.json` and `accepted_source_hashes.json`.
Tests use raw `.xml`, `.log` and available `.exit` files under `raw/results`.
No local JAX/chemistry calculations were run to produce these summaries.
'''
    (report/"report.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--acceptance-xml", type=Path)
    args = parser.parse_args()
    report, raw = args.report_dir.resolve(), args.report_dir.resolve()/"raw/results"
    specs = (
        ("water-def2-svp-cpu1", "final-baseline-water-cpu1.jsonl", ["final-candidate-water-cpu1.jsonl"], True),
        ("water-def2-tzvp-cpu64", "baseline-water-def2-tzvp-cpu64-flat.jsonl", ["candidate-water-def2-tzvp-cpu64-flat.jsonl"], True),
        ("benzene-def2-svp-cpu64", "baseline-benzene-def2-svp-cpu64-flat.jsonl", ["accepted-candidate-benzene-def2-svp-cpu64-flat.jsonl", "candidate-benzene-def2-svp-cpu64-flat.jsonl"], True),
        ("water-def2-svp-gpu", "final-baseline-water-gpu.jsonl", ["accepted-candidate-water-gpu.jsonl", "final-candidate-water-gpu.jsonl"], True),
        ("residue128-cpu1-projected", "baseline-residue128-cpu1.jsonl", ["projected-residue128-cpu1.jsonl"], False),
        ("residue512-cpu1", "baseline-residue512-cpu1.jsonl", ["candidate-residue512-cpu1.jsonl"], False),
        ("residue-ad128-cpu1", "baseline-residue-ad128-cpu1.jsonl", ["candidate-residue-ad128-cpu1.jsonl"], False),
        ("residue128-gpu-serial", "baseline-residue128-gpu.jsonl", ["candidate-residue128-gpu.jsonl"], False),
        ("residue128-gpu-tile4", "baseline-residue128-gpu.jsonl", ["gpu-tile4-residue128-gpu.jsonl"], False),
        ("residue128-gpu-tile8", "baseline-residue128-gpu.jsonl", ["gpu-tile8-residue128-gpu.jsonl"], False))
    comparisons = [build_comparison(raw, *spec) for spec in specs]
    selected = {c[k] for c in comparisons for k in ("baseline_file", "candidate_file")}
    with (report/"measurements.jsonl").open("w") as stream:
        for path in sorted(raw.rglob("*.jsonl")):
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                row = dict(record_type="raw_measurement", source_file=str(path.relative_to(report)),
                           source_line=line_number, source_sha256=digest(path),
                           selected_for_comparison=path.name in selected, measurement=json.loads(line))
                stream.write(json.dumps(row, allow_nan=False, sort_keys=True)+"\n")
    csv_rows = []
    for c in comparisons:
        common = {k: c[k] for k in ("id", "baseline_file", "candidate_file", "baseline_sha256", "candidate_sha256", "baseline_manifest_sha256", "candidate_manifest_sha256", "case_input_sha256", "dtype", "cpu_logical", "cpu_physical")}
        for k in ("dimensions", "threads", "science", "before_first", "after_first", "gpu_before", "gpu_after"):
            common[k+"_json"] = json.dumps(c[k], sort_keys=True)
        for metric, values in c["metrics"].items():
            row = dict(common, metric=metric, unit="seconds", ratio_before_over_after=values["before_over_after"], notes="Warm samples only; repeat zero retained separately")
            for variant, key in (("before", "baseline"), ("after", "candidate")):
                row.update({key+"_"+k: json.dumps(v) if k == "samples" else v for k, v in values[variant].items()})
            csv_rows.append(row)
        csv_rows.append(dict(common, metric="process_rss_high_water", unit="bytes",
                             baseline_median=c["process_rss_before_bytes"], candidate_median=c["process_rss_after_bytes"],
                             ratio_before_over_after=None, notes=c["process_rss_notes"]))
        if c["xla_compile_before"] and c["xla_compile_after"]:
            for metric in ("xla_temp_bytes", "xla_argument_bytes", "xla_output_bytes", "wall_seconds", "lower_seconds"):
                csv_rows.append(dict(common, metric="compile_"+metric,
                                     unit="bytes" if metric.startswith("xla_") else "seconds",
                                     baseline_median=c["xla_compile_before"][metric],
                                     candidate_median=c["xla_compile_after"][metric],
                                     ratio_before_over_after=None, notes="Single compiler observation; not process RSS"))
    fields = list(dict.fromkeys(k for row in csv_rows for k in row))
    with (report/"before_after.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    references = build_references(raw)
    evidence = {path.name: xml_summary(path) for path in sorted(raw.glob("*.xml"))}
    acceptance_path = args.acceptance_xml
    if acceptance_path is None:
        paths = sorted(path for path in raw.glob("*.xml")
                       if path.name.startswith(("accepted-", "resume-", "resumed-"))
                       and ("narrow" in path.name or "gate" in path.name))
        acceptance_path = paths[-1] if paths else None
    acceptance = None if acceptance_path is None else dict(
        xml_summary(acceptance_path), file=str(acceptance_path.relative_to(report)))
    references["test_evidence"] = evidence
    references["corrected_final_gate"] = acceptance
    references["comparison_scientific_gates"] = {c["id"]: c["science"] for c in comparisons}
    (report/"reference_comparison.json").write_text(json.dumps(references, indent=2, allow_nan=False)+"\n")
    write_report(report, comparisons, references, evidence, acceptance)
    print(f"Analyzed {len(comparisons)} comparisons; wrote report.md, before_after.csv, measurements.jsonl, reference_comparison.json")


if __name__ == "__main__":
    main()
