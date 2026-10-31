# Molecular GW-family and conventional-integral execution

Baseline: ea646bf; preserve release/v1.0.0 and the previous G0W0 branch.
User confirmed phenanthrene (angular C14H10), not linear anthracene.
Conventional means actual four-center Coulomb data/actions, no auxiliary RI
and no preliminary full AO-pair spectral factors. Exact packed/direct choices
may follow the budget. No periodic-method expansion or physics truncation.

- [x] GF-01: Freeze source, resources and new physical input.
  - [x] Record versions, SHA, source/library paths and original workspace state.
  - [x] Use one V100/NV5090, <=64 logical CPUs, 256 GiB; no duplicate allocation.
  - [x] Validate phenanthrene identity, charge/spin, Cartesian def2-SVP and raw geometry hash.
  - [x] Run existing small baseline gates; retain actual convergence/residuals.
- [x] GF-02: Generalize fixed-orbital preparation to evGW0/evGW.
  - [x] Prepare static ordered MO vertices/exchange once, not every energy iteration.
  - [x] Reuse W0 only when physically fixed; rebuild W for evGW without changing response.
  - [x] Propagate budget and dense/GMRES configuration with stable numerical entry points.
  - [x] Check full outer-root JVP/VJP and same-shape data/cache invalidation.
- [x] GF-03: Bound qsGW endpoint matrix work.
  - [x] Produce required self-energy rows/columns rather than all-endpoint full tensors.
  - [x] Keep all matrix elements, mode-A Hermitian average and actual residual certificate.
  - [x] Share fixed-frequency preparation while updating coefficients/density each iteration.
  - [x] Compare dense values, first response, input-change results and memory.
- [ ] GF-04: Bound scGW matrix/tail work and repeated preparation.
  - [ ] Measure and tile time/frequency screening contractions; retain the exact grid.
  - [ ] Reuse only spectral/grid data independent of current chemical-potential probe.
  - [x] Preserve density/charge, moments, reference subtraction and GM-energy definitions.
  - [x] Verify matrix values/outer implicit response, particle/residual gates and timing.
- [x] GF-05: Supply conventional screened actions without AO-pair diagonalization.
  - [x] Reuse ao2mo four-center selected blocks and native direct J products.
  - [x] Implement exact transition-space dielectric products using shared checked solvers.
  - [x] Budget optional stored transition integrals or direct actions, with no hidden RI.
  - [x] Compare against full small-system conventional results and array/geometry products.
- [x] GF-06: Integrate conventional consumers and public capability boundaries.
  - [x] Connect G0W0 and energy-updated/matrix workflows to declared actual representation.
  - [x] Retain direct SCF providers; do not silently materialize full ERIs or spin identity tensors.
  - [x] Preserve current output, derivative and failure contracts; explicit errors for unsupported combinations.
  - [x] Regress RI and conventional parity under controlled integral inputs.
- [ ] GF-07: Complete phenanthrene and bounded method comparisons.
  - [x] First run verified G0W0 RI on full G/W spaces, six frontier QPs and explicit precision.
  - [x] Run conventional counterparts within preflight budget; mark limits, never force OOM.
  - [x] Pair cold/warm, iteration/call counts and process/GPU memory without mixed workloads.
  - [ ] Compare aligned PySCF/MOLGW only where definitions/outputs match.
- [x] GF-08: Close implementation, evidence and resources.
  - [x] Recheck all TODO states, source/library fingerprints and scientific gates.
  - [x] Save raw records, comparison CSV, report, limits and rerun commands.
  - [x] Local reviewable commits; no automatic merge/push.
  - [x] Release only this project's allocation and confirm original workspace preservation.

## Budgets and acceptance

All numerical tests remote inside Slurm. Individual calculation cap 6000 s;
default validation round 7200 s. Repetition reductions must be explicit and
different cache/hardware/algorithm scopes are never divided into one ratio.
Float64; preserve grid, states, full G/W spaces, eta, six seeds and root policy.
Rearrangement products atol1e-10 Ha/rtol1e-9; same-branch QPs <=1e-8 Ha plus
actual root residual. External aligned QPs <=1e-3 eV. Existing nuclear
response <=1e-5 Ha/bohr. Record coupled outer/particle residuals separately.
No general full RI geometry/auxiliary-metric AD claim is added by preparation
reuse. Conventional AD must keep Coulomb metric/integral dependencies live.

## Historical midpoint (superseded by closeout below)

- Allocation 113525: n003, one V100, 32 logical CPUs (subset of authorized
  64), 256 GiB. Only one numerical step at a time.
- Phenanthrene RI baseline: full G/W, 260 Cartesian AOs, 974 auxiliary
  functions, 47 occupied orbitals, six frontier QPs. Both recorded executions
  passed QP residual/positive-weight/branch checks; first/warm total
  199.773/191.863 s. Process ru_maxrss 8,509,132,800 bytes.
- First narrow candidate: 18 passed; first qsGW streaming suite: 9 passed.
- Reviewed candidate: conventional 16 passed (includes full moving-native
  H2 QP JVP/VJP and reconverged geometry FD), evGW preparation/compile
  5 passed, qsGW streaming 11 passed, original qsGW 12 passed, original
  evGW 6 passed. Remaining outer/Matsubara response groups are in progress.
- Preserve initial exit 139 from a combined process (30 completed tests,
  compiler stack in water qsGW). The same water case passed in an isolated
  CPU1 process before the stable-JIT fix. Repeated tracing is separately
  reproduced/fixed; it is not a proven explanation for the native crash.
- Exact conventional screening currently covers restricted G0W0/evGW0/evGW.
  It stores V_OV and selected cross vertices, with native AO2MO or packed
  source. Cache-free conventional screening, conventional qsGW/scGW/U-GW
  and conventional BSE export are not implemented; no spectral fallback.
- scGW change currently bounds frequency-factorization workspace only.
  Full Pi/W/time outputs and their previous response contract remain.

## Iteration closeout

Production commits: ca87549 (qsGW), 12d2ad7 (Matsubara),
28475e6 (shared restricted preparation/conventional CD and ownership fixes).
Source bytes loaded remotely match these committed production files.

GF-04 local deliverables are complete: per-frequency LU checkpointing and
one shared compiled canonical chemical-potential search, preserving its
initial-hit fast path. Full Pi/W/time output streaming and a complete scGW
caller/response budget are NOT implemented; those original broad subitems
remain open rather than being marked complete. Full numerical/response gates
passed; performance claims are limited to measured kernels.

GF-07 measured deliverables: matched CH2O evGW0/evGW CPU32, CPU1 fixed-input
qsGW/screening/charge kernels, actual non-RI water G0W0, phenanthrene 260-AO
CPU/GPU G0W0 and 564-AO GPU G0W0. All G/W spaces are complete. Fresh external
PySCF/MOLGW or phenanthrene conventional calculations were NOT run, so these
optional original subitems remain open and receive no invented ratio.

Final gates: the original broad affected suite recorded 129 passes. The final
affected sequence recorded 74 passes, followed by conventional-final 20 and
facade-final 8 passes after their last budget patches. These counts overlap.
The fixed-input full-output comparison passed with qsGW max difference
1.39e-17 Ha and zero observed scGW-screen/charge differences.

Allocation 113525 was released and squeue confirmed it absent. Original
workspace HEAD 4d67479 and its empty tracked diffs/untracked path list remain.
No merge or push. Remaining combinations (qs/sc/U conventional, cache-free
conventional actions and conventional BSE export) are explicit scope boundaries.
