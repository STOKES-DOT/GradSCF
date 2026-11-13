# Handoff

Implementation worktree: `/Users/jiaoyuan/.codex/worktrees/bd0a/GradSCF`.
Branch: `perf/shared-scf-response-20261110`; parent c5c8b28e. Original workspace
and its unrelated dirty files are preserved. No merge or push was performed.

Production commits: 87ab1bb9 (restricted full TD XC reuse/Fock diagonal),
37ba5352 (common dynamic physical grid-field JIT), 7dd87dfa (safe shell-direct
response program reuse). Only four production Python files changed, 79 added /
30 removed lines; no SCF iteration kernel, native library, GW/BSE physics,
convergence criterion or numerical response cache was replaced. Tests463212dd
and f24cfbe2 isolate instrumentation and certify coordinate AD.

The existing six HF/KS facade/solver/provider architecture is reused. Six-mode
direct/RI independent-reference gates and direct/dense gates pass. GHF/GKS
spinor TD response remains unavailable. Known linear XC full-TD uses one shared
X+Y action; opaque/clipped callbacks retain separate X/Y actions. Generic raw
factors and A/B exchange semantics remain intact. Small automatic exact ERI
caches preserve eager admission; ambiguous/custom providers keep their route.

Final recorded test identities: 254 pass, 11 fail identically on untouched
parent, zero skip. Seven failures concern host NumPy/JAX storage expectations;
four concern old exact-cache/admitted-budget and unsupported-basis-AD
expectations. They are separate from one mixed JAX compiler segfault (-11),
whose cause remains unresolved. Isolated groups25/14/17/20 all pass. New36
regressions pass on GPU. Two explicit shared-direct coordinate-AD cases pass:
PBE/globalNN full TDDFT FD errors9.85e-9/5.34e-9 Ha/Bohr. External NN three
training modes, force fixtures and NNAO basis fixtures pass; no full dataset
training campaign or large Hessian was run.

Clean same-node 5090 response benchmarks use Cartesian def2-SVP/PBE0/grid3,
all orbitals, five full-TDDFT roots, float64. Warm medians, parent->candidate:

| Case | Response seconds | Entire SCF->spectrum seconds |
|---|---:|---:|
| Water direct | 3.286->0.187 | 3.530->0.434 |
| Benzene adjacent direct | 25.853->16.971 | 34.834->25.866 |
| Benzene RI | 2.230->1.231 | 6.755->5.644 |

Each process has one first call and two warm fresh-object calls. The original
separated benzene pair showed no end-to-end benefit due to SCF-phase drift;
its unfavorable first/warm measurements remain in the report. Adjacent direct
SCF was stable8.874/8.786 s. Do not multiply independent speedups. Direct
benzene GPU allocator cumulative peak increases0.810->1.639 GB; RSS drops,
so no uniform memory reduction is claimed. First calls and memory definitions
are in report.md/timings.csv/process_memory.csv. Fresh public benzene response
compile events are3 initially and0 on its next call. Program caching never
returns an old density, geometry, screening or NN parameter value.

H2 public workflows cover four molecular GW modes and the supported three QP
GW modes->TDA/full BSE. Water/PBE0/def2-SVP/grid3 G0W0->both BSE was paired;
25QP roots are valid, max QP residual2.58e-9 Ha and BSE residual1.61e-10 Ha.
Downstream physical values are unchanged. scGW has no validated QP continuation
for BSE and its rejection is a guard, not a successful spectrum. No benzene
GW/BSE or new GW kernel speedup was established.

All scientific tests were remote within one job119123 (NV5090/16 logical
CPU/128 GiB); main workflows use CPU native/SCF preparation then GPU response,
not GPU-native libcint. Controller ended2026-11-10 14:38:11+08, squeue empty;
sacct is unavailable because accounting is disabled. No local scientific run.
590 source Python hashes for each remote export match the respective source.

Evidence: report.md, manifest.json, response_parity.json, coordinate_ad.json,
acceptance.json, baseline_failures.json, compile_counts.json; raw/ holds every
JSONL/XML/log/process record and actual queue script, indexed by
raw_inventory.json. Raw large arrays/build outputs are excluded. Commands:
`benchmarks/probes/shared_execution/README.md`, plus the downstream probe README.
Metadata-only reproduction:

```sh
python3 benchmarks/probes/shared_execution/summarize.py reports/shared_execution/20261110-c5c8b28e
```
