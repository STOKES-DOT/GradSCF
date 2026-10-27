# TD response performance probe

Baseline: `33a7cab3473a123f027877e9e7e51a3d86f75fd8`, clean tracked
source. The original checkout and its unrelated untracked files are retained.

- [x] Freeze source, inputs, dirty/index fingerprints and resource limits.
  - [x] Create `probe/tddft-perf-20261027` in a separate managed worktree.
  - [x] Read lwcluster 0.8.0 manual and obtain one A800, 64 CPUs, 512 GiB.
  - [x] Build libcint in the isolated remote source tree.
  - [x] Run water/PBE smoke, then an independent clean timing baseline.
- [x] Establish the small representative baseline.
  - [x] Water/PBE TDA and full TDDFT, independent PySCF comparison.
  - [x] Benzene/PBE0/RI scale point and OH/PBE unrestricted case.
  - [x] Bounded CH2O/PBE0 fallback for stage and exact-rearrangement tests.
  - [x] CPU1, representative CPU8, existing GPU path and memory estimates.
- [x] Diagnose and test at most three local exact rearrangements.
  - [x] Separate preparation, lowering, compilation and warm actions.
  - [x] Packed public J/K, TDA demand-driven factors, full same-source sharing.
  - [x] Full geometry AD, NNAO coefficients/log exponents and neural response.
  - [x] Offline NTO/K-action accuracy and ISDF/THC storage feasibility.
- [x] Analyze evidence and deliver.
  - [x] Classify H1–H7 with measured versus source-only evidence.
  - [x] Preserve every failure, timeout, input, repetition and resource metric.
  - [x] Write report, recommendations, reproduction commands and summaries.
  - [x] Recheck original workspace fingerprints; do not merge or push.

Formal validation is bounded by 7200 s from the first clean baseline input;
individual processes default to 600 s and a sampled 64 GiB RSS guard. The
8-hour allocation limit is imposed by n005; it is not the test budget.
No production numerical kernel is changed by this campaign. Diagnostic runs
and experimental monkeypatches are separate from clean baseline timings.

Completed as a bounded diagnostic campaign. Unsupported, nonconverged, timeout and provenance-limited results are recorded in report.md; this does not certify all method capabilities.
