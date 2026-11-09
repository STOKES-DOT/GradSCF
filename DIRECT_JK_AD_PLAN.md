# Direct J/K parameter response — fixed TDD scope

Baseline:96ad38e5, after the completed XC locality round. Continue the existing
isolated worktree; preserve all unrelated dirty files and the XC evidence.
The user requested geometry/basis gradients without losing forward efficiency,
and explicitly requested initial testing before implementation.

## Contracts

- libcint shell-direct forward remains the reference algorithm. No AO ERI or
  AO integral Jacobian/cotangent is introduced into the production direct path.
- Preserve nonsymmetric/batched/complex density, J-only/K-only/joint selection,
  Cartesian/spherical AOs, static omega and existing zero-cutoff coordinate AD.
- Keep normalization, floating centers, exponents and raw contraction
  coefficients live. Distinguish raw and normalized coefficient derivatives.
- A positive screening mask is an explicit approximation: within an unchanged
  mask its indicator derivative is zero. Preserve the same primal selection in
  push/pull/mixed products, independently of tangent/cotangent density. It is
  not a global physical derivative error bound; test against unscreened truth too.
- Keep native worker/workspace quotas and trace-local preparation. Forward
  timing must be measured separately; never substitute unscreened work or a
  dense integral fallback to claim the direct path is complete.
- Existing public adapters and NNAO S/T/V limitations are separate from J/K.
  Do not claim full force training or an unimplemented nuc_grad facade merely
  because an integral product passes.

## TODO

### DJK-01: baseline and explicit RED
- [x] 01a Verify current zero-cutoff geometry/density and preparation tests.
- [x] 01b Reproduce positive-cutoff geometry and raw coefficient/log-exponent
  JVP/VJP failures, plus prepared/provider entrypoints.
- [x] 01c Lock small FD/transpose and full-ERI oracle controls; mixed basis
  references use FD of supported first derivatives where higher AD is absent.
- [x] 01d Record same-source native/library provenance and bounded forward timings.

### DJK-02: shared direct derivative products
- [x] 02a Reuse existing canonical coefficient/exponent shell evaluators for
  contracted JVP/VJP, retaining zero-coefficient and normalization response.
- [x] 02b Extend geometry products with the same forward quartet predicate,
  including its reference, J/K selection and static omega.
- [x] 02c Preserve geometry second response and add required parameter-mixed
  products with explicitly documented supported derivative orders.
- [x] 02d Retain bounded thread-local work/reductions and prepare once per state;
  avoid stale parameter caches and unbounded per-worker matrix allocation.

### DJK-03: consumer integration
- [x] 03a Migrate get_jk, bound/projected providers and prepare_jk/cache selection.
- [x] 03b Remove only obsolete traced-geometry guards after kernel acceptance;
  verify supported HF/KS relaxed geometry paths retain their requested policy.
- [x] 03c Verify actual NNAO energy training/coefficient/exponent chains; record
  any independent one-electron/mixed-force capability gaps without hiding them.
- [x] 03d Preserve six HF/KS forward modes and neural-XC default behavior using
  narrow existing regressions; no new general gradient facade/framework.

### DJK-04: numerical, efficiency and delivery
- [x] 04a CPU small JVP/VJP/FD/mixed gates; GPU stage-boundary checks where supported.
- [x] 04b Same-task forward cold/warm timing and derivative products, actual RSS,
  shell work and no-ERI assertions; representative larger forward anchor.
- [x] 04c Independent source review, fixed plan audit and concise scope/results.
- [x] 04d Focused local commits and the subsequently authorized local merge;
  no push and no extra numerical matrix.

## Validation/resources

Float64, coordinates Bohr, energies Hartree. Initial array JVP/VJP gates use
atol3e-10/rtol3e-9; two FD steps2e-4/1e-4 with atol3e-8/rtol3e-6 (parameter
units explicit). Relaxed gradients use1e-5Ha/Bohr and tighter existing fixtures.
Forward equivalence uses tight array regression and unchanged convergence.
Screening mask stability is checked before finite-difference acceptance.
All numerical tests/builds run on jy_gpu_2 Slurm: one NV5090,64 logical CPUs,
256GiB, serial heavy steps, per-process guard6000s and RSS200GiB,2h rounds.
Use isolated remote source/build outputs; never overwrite the completed XC
baseline. No local numerical tests or dependency installation.

Evidence2026-11-09: baseline42; initialRED10/2; partialmask/providerRED20;
screenedgeometryGREEN103/9diagnosticskips. Forwardrepeatold12.3122s versus
candidate12.5090s; densityreference11.8350s versus11.4566s. All J/K arrays
exactlymatch. Firstpair17.8%signal retained; oldbinaryperioddrift observed.

Final resource record: the final 64- and 32-CPU reservations remained pending
and were cancelled. Job 118589 used one NV5090, 8 logical CPUs (4 physical
cores), and 256 GiB. The A800/V100 fallback was authorized but not used.
Slurm denied the requested time extension. Baseline five / candidate three
warm samples were registered before candidate execution; inputs and per-call
work are unchanged. No whole-process wall-time ratio is reported.

## Final audit — 2026-11-09

- DJK-03b/03d: final CPU sweep was 107 passed plus one UHF None-ERI failure.
  Commit b9a38f7f fixes that interface; the exact failed case and 16 related
  direct/DF, six-facade and neural-XC training checks then passed. The initial
  failing receipt is retained; this is targeted revalidation, not a repeated
  108-case sweep. No unsupported derivative order is silently downgraded.
- DJK-04a: parameter group 86 passed, spherical partial-mask group 4 passed,
  final CUDA-stage group 6 passed, NNAO/synthetic basis training group 3 passed.
  These groups overlap. Geometry, raw coefficient and log-exponent contracts
  and complete derivative boundaries are in the acceptance ledger.
- DJK-04b: final 96-atom / 880-AO forward arrays match exactly. CPU8 warm
  medians are 141.3201 -> 132.0922 s (basis mask) and 136.1782 -> 117.1250 s
  (density-reference mask). Shared-node drift prevents a stable speedup claim;
  neither variant exceeds the predeclared 10% regression target. Eight
  fixed-density value/AD products pass their consistency gates. Diagnostic
  coverage is 35 passed plus one missing-environment skip, followed by the
  exact skipped test passing with diagnostics enabled. One/four workers each
  evaluate 55 quartets for the eight-density fixture.
- DJK-04c/04d: the source review's schedule-storage and inherited-budget issues
  were reproduced in three RED tests and fixed. Final remote provenance
  matches all 586 audited local/committed source files. Production commits:
  9c3c2180, d3d23408, b9a38f7f. The final receipts close this bounded plan;
  delivery records accompany the local merge requested by the user.

See reports/direct_jk_ad/20261109/{report.md,acceptance.json,manifest.json,
commands.md,results/}. The separate XC_LOCALITY_PLAN.md is already complete.
Older HFKS reference-recovery gates and large benchmark matrices remain separate.
