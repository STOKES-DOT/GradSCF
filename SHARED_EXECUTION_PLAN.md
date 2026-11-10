# Shared SCF and response execution — fixed TDD scope

Baseline c5c8b28e (includes new iKS work); do not revert to c2c12bcc.
Worktree bd0a, branch perf/shared-scf-response-20261110. All scientific
execution/builds remote jy_gpu_2 Slurm only; local stdlib inspection/edits.

## Contracts

Use existing common RHF/UHF/GHF and RKS/UKS/GKS solver/grid interfaces.
Retain dynamic geometry, basis, density and external/neural parameters; no
new stop_gradient or numeric cache keyed only by mutable object identity.
Keep full/packed/direct/true RI meanings, small direct packed-cache selection,
non-symmetric raw factors/densities, independent X/Y, clipped/opaque/global
XC semantics and existing fixed_density/explicit/implicit training modes.
Improve observed repeated work without changing physics, root space, grid or
convergence. No residual safeguard removal or full response matrix.

## TODO

### SRE-01 shared physical fields and six-method contracts
- [x] 01a Remote RED: stable same-shape grid field program with changed data.
- [x] 01b Implement shared dynamic field projection helpers in existing layers.
- [x] 01c Values/JVP/VJP, geometry and neural training regressions.
- [x] 01d RHF/UHF/GHF/RKS/UKS/GKS parity and relevant XC-family coverage.

### SRE-02 TDDFT numerical action and preparation
- [x] 02a RED linear/opaque XC reuse, unused density, noncanonical diagonal.
- [x] 02b Minimal math/dataflow fixes retaining generic callback contracts.
- [x] 02c RED/GREEN direct compile reuse with dynamic inputs and cache policy.
- [x] 02d TDA/full, unrestricted, arrays/model AD and error-contract regressions.

### SRE-03 downstream molecular workflow
- [x] 03a Verify current APIs/capabilities and freeze shared-input baseline.
- [x] 03b Test supported four GW forward modes and available BSE chains.
- [x] 03c DFT->TDDFT and DFT->GW->BSE water/benzene integration as budget allows.
- [x] 03d Preserve supported complete derivative paths; expose existing gaps.

### SRE-04 acceptance and delivery
- [x] 04a Same-hardware parent/candidate water and benzene timed comparison.
- [x] 04b Independent source/spec review; remove unjustified candidate changes.
- [x] 04c Report raw timings, compile counts, root residuals and actual memory.
- [x] 04d Focused local commits, final source hash and plan audit; no merge/push.

Float64; total energy1e-7Ha external / tighter same-code parity, TD energies
1e-9Ha, original operator residual1e-8Ha, supported geometry1e-5Ha/Bohr.
Array and model derivative tests retain existing tighter tolerances. No
post-result tolerance relaxation. Count instrumentation and clean timing are
separate. Numerical caches and program caches have independent dependencies.

Resources <=one NV5090/V100/A800 and64logicalCPU/256GiB. One shared allocation,
heavy cases serial, CPU8/GPU representative measurements paired identically.
Individual guards <=6000s; one validation round <=2h, with idle release. Root
alone owns sync/job scheduling; agents only edit assigned files and send tests.
Unsupported scGW-to-BSE/QP routes remain explicit; no invented continuation.

## Final evidence audit (2026-11-10)

SRE-01: RED/GREEN shared dynamic fields; six facades direct/true-RI versus
independent PySCF and direct/dense; executable XC categories and three neural
training modes pass their scoped regressions. No six duplicate solver kernels.
SRE-02: 36 new gates pass on GPU, matching CPU gates; explicit shared-direct
coordinate AD passes PBE/globalNN full TDDFT with complete relaxed-SCF FD.
SRE-03: three parent/candidate downstream comparison sets pass; four GW modes
forward and supported three QP GW modes -> TDA/full BSE, real water G0W0/BSE.
Matsubara scGW continuation and generalized spinor TD remain explicit guards.
Benzene GW/BSE was not run; benzene full TDDFT was paired in direct and RI.
SRE-04: independent source/spec review found no P1/P2; all final isolated
regressions and exact source hashes verified. 254 unique test identities pass;
11 fail identically on unchanged parent (7 host-type, 4 cache/old AD boundary).
Mixed-process compiler segfault remains unresolved and separately disclosed;
isolated groups25+14+17+20 and new coordinate2 pass. This is completion of the
bounded engineering plan, not a clean full-repository suite or new capabilities.
Original source fingerprints/status unchanged. Job119123 released14:38:11+08.
Local commits only; no merge/push.
