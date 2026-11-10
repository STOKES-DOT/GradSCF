# Call-local AO reuse: accepted RKS result

| GradSCF variant | Cold facade (s) | Warm median (s) | Warm sample SD (s) | Whole-worker peak RSS (decimal GB) |
|---|---:|---:|---:|---:|
| Parent `40a569bf` | 792.491512 | 750.493826 | 0.257322 | 70.485 |
| Exact-buffer call-local AO reuse v3 | 577.914945 | 535.544680 | 1.355704 | 71.205 |

The complete 880-AO beta-carotene RKS/PBE/RI-J pair passed all four registered
calls. With preparation included, the **warm median decreased by 28.64%**;
whole-worker peak RSS increased slightly, by 719,446,016 bytes (1.02%).

Both use 23 SCF cycles per call, energy -1554.7036403036154 Ha and final MO
gradient norm 3.9285116124375864e-8 Ha, with zero recorded parent/candidate
differences. The warm ratio is candidate/parent = 0.713590; cold time decreased
by 27.08%. [Parent](baseline_rks_pbe_ri.json) and
[candidate](candidate_rks_pbe_ri.json) retain samples and raw hashes. RSS spans
all four repeats with 0.2-s process-group sampling, not an isolated warm peak.
No overall memory reduction is claimed.

## What changed

The old planner could reject one-shot full-grid primitive scratch even when
retained AO planes would fit. The streamed provider then reevaluated Gaussian
values on each SCF pass. The production patch charges live inputs, forward
state matrices and known native/RI workspace; prepares Gaussian blocks once;
and retains exact grid-leading value/derivative planes with bounded gathers.
Stable JIT plus disjoint full-block/tail writes avoids a redundant full output
copy during preparation.

The derived reducer belongs to one solve. Raw providers, reference objects and
global caches are not mutated; new inputs rebuild it. Explicit streaming,
inadequate budgets, unknown operators/devices and external/global callbacks
retain their old paths. No functional, quadrature, orbital cutoff, integral
engine, physical stop-gradient or relaxed convergence gate was introduced.

The tested RKS/grid files match `1a575d18`; the timing snapshot's unused UKS
caller has the earlier single exchange-workspace reservation. The compatibility
run uses a complete `1a575d18` archive. Exact lineage is in
[ao_cache_candidate_source.json](ao_cache_candidate_source.json).

## Evidence and limits

The fixed-five diagnostic located streamed execution: initial Fock about
27.4 s, five-cycle solver 159.7 s, and input preparation 173.5 s including
105.9 s for RI factors. These synchronized inclusive diagnostic spans include
overhead; nested children are not additive or formal clean benchmark phases.

An earlier shared-input XC prototype measured warm action 25.1830 -> 12.0135 s
with zero recorded E/V differences. Its 106.7303 s cache preparation is
separately charged. Factor capacity was reserved but factors were not physically
held. This is operator evidence for the same-input experiment, not a complete
SCF result or a timing measurement of the later exact-buffer snapshot.

Execution and workspace RED records were retained. The budgeted candidate and
existing regressions passed 96 cases; the exact-buffer refactor subsequently
passed 13 cache/budget/AD cases in 10.50 s, including JVP/VJP, dense HVP,
two-step FD, changing inputs, ownership and fallback checks. These groups
overlap. [TDD_AMENDMENTS.md](TDD_AMENDMENTS.md) preserves earlier failures and
fixture corrections without changing numerical thresholds.

The additional eight-case NNAO/native-geometry compatibility group passed
without skips (94.350 s test time, 100.9668 s guarded wall time, peak RSS
2,646,773,760 bytes), using the full `1a575d18` archive and unchanged native
library. It covers UKS/GKS implicit/unrolled geometry, methane direct/RI NNAO,
native coefficient/exponent products and a small two-geometry MACE update.
Together with earlier three-mode neural and force-loss passes, this closes
T4.2 within its previously supported scope. It does not add auxiliary-geometry
AD or establish a trained-model accuracy claim; the full-size matrix is separate.

The [certified PySCF reference](reference_certified.json) has warm median
211.686044 s and passes four calls, but its atomic guess, RI representation and
bounded final-check continuation differ. The original single-kernel failure
and metadata observer that built unnecessary factors remain recorded in
[reference_original_single_kernel.json](reference_original_single_kernel.json).
No cross-engine implementation speedup is inferred from these different workflows.

The no-RI parent RKS/PBE run started at 2026-11-05 15:39:32 +08:00; candidate
and reference runs remain queued behind it. The full 30-configuration matrix
per engine is unfinished. This accepted RI-J result is not a universal,
all-method or GPU speedup claim.
