# Fixed TDD contract: molecular HF/KS execution

Contract frozen. Current implementation and execution status are recorded in
`TDD_STATUS.json`; this document defines acceptance rather than completion.

This contract replaces the unfinished 60-system campaign as the active
optimization target. Its original results remain immutable. Changes to this
contract require a dated reason and an explicit diff; failed tests must not be
removed or weakened to close a task.

## Frozen physical task

- Source baseline: `40a569bf9c9515e92e30d499f3784266ad2408a8`.
- Worktree: `/Users/jiaoyuan/.codex/worktrees/5fbb/GradSCF`.
- Branch: `perf/scf-100-grid3-20261105`.
- Molecule: the existing PubChem beta-carotene conformer, C40H56, 96 atoms,
  296 electrons, charge 0, spin 0. This is approximately 100 atoms; atoms are
  neither added nor removed. Geometry is copied from the registered catalog.
- Basis: Cartesian def2-SVP. KS grid level: 3. No orbital truncation, reduced
  grid, mixed precision, or density-fitting surrogate made from four-center
  ERIs is allowed.
- Methods: RHF, UHF, GHF; RKS, UKS, GKS with SVWN, PBE, PBE0, CAM-B3LYP.
- Two representations per method: conventional libcint direct J/K, and
  explicit auxiliary RI with weigend and metric threshold `1e-12`.
- RI means RI-JK for HF/global hybrids, RI-J for pure semilocal methods.
  The first RSH RI implementation is explicitly **RI-J + direct K**, including
  its required range-separated exchange. It must not be labelled RI-JK or
  compared to a different reference approximation.
- Initial guess: `atom`, unchanged within every parent/candidate pair. The
  GradSCF atomic-UHF superposition and PySCF spherical atomic guess are
  different algorithms; their setup cost and iteration counts are reported.
- SCF gates: energy change `1e-10 Ha`, density RMS `1e-8`, occupied-weighted
  MO gradient `1e-7 Ha`, maximum 200 cycles. Finite energy is insufficient.
- Direct screening: `1e-12` for the large forward benchmark. Complete geometry
  derivative fixtures use zero screening on both AD and finite-difference
  sides. This numerical setting is not an RI approximation.

The generated protocol, geometry and all case configurations are recorded in
the input manifest with SHA256 before their first scientific execution.
There are 30 method/representation cases per engine, not a new molecule sweep.

The canonical inputs are `plan-v2/`; the earlier `plan/` was an unexecuted
draft. Geometry SHA256 is
`4d4f50763390ba75438c1e28e9a3774e51648af173c2eaf8cf2ab697e975008d`.
The canonical protocol SHA256 is
`3ce5657527478ea592e46ab4d4a13d70e8f3503a3298ce7575d077269ac820a2`.
`TDD_LOCK.json` records exact input/test bytes. The public RI-J/direct-K
selection is `density_fit(auxbasis="weigend", only_dfj=True)`.

## Execution and measurement rules

- Reuse only the authorized allocation, or one sequential replacement:
  one V100/NV5090, at most 64 logical CPUs, 256 GiB. No scientific test runs
  locally or on the login node. Heavy runs are serial.
- CPU float64 is the common baseline. GPU is a separate optional measurement,
  and cannot replace an unfinished CPU correctness gate.
- Software planning budget: 180000 decimal MB; process-group RSS guard:
  200 GiB. A single scientific call is bounded by 6000 seconds and remaining
  allocation time. Step-launch waiting and calculation time are separate.
- Every large configuration has an independent process and one cold call.
  RKS/PBE and UKS/PBE0 additionally use three fresh-object warm calls for each
  representation. Inputs and initial-guess policy remain unchanged; no saved
  converged state is silently injected.
- Synchronize complete JAX output before stopping timers. Record facade time,
  independent preparation/action diagnostics, iterations, final residual,
  sampled group RSS, ru_maxrss, source/module/native paths and hashes,
  affinity, native/BLAS/JAX settings, dtype and actual representation.
- Process completion, scientific acceptance and timing eligibility are
  separate fields. Unsupported parent paths have no speedup denominator.
- Array/coordinate/model updates must invalidate physical data while allowing
  executable reuse. No new stop_gradient boundary or detached numeric cache.

## Ordered TODOs

### T1. Freeze and establish evidence

- [x] T1.1 Stop the old controller and scientific step; preserve results and
  resource ownership. Record the user-requested interruption separately.
- [ ] T1.2 Freeze the input manifest, 30-case matrix per engine, numerical gates
  and resource/time limits before scientific execution.
- [ ] T1.3 Run the unmodified baseline: small correctness first, then bounded
  large representatives. Preserve failures and capability rejections.
- [ ] T1.4 Record RED results for the narrow production changes below; retain
  existing passing regressions as additional acceptance gates.

### T2. Improve restricted/unrestricted consumers

- [ ] T2.1 Restrict occupied RI exchange intermediates to the actual occupied
  widths; preserve fractional-occupation and general-density contracts.
- [ ] T2.2 Reuse direct J/K through an explicit previous-state contract; handle
  fresh certification, mixed densities and RSH consistently.
- [ ] T2.3 Apply the existing total budget to actual contraction/worker
  workspace without allocating it all to one kernel or changing approximation.
- [ ] T2.4 Provide explicit RSH RI-J/direct-K assembly and matching reference
  settings; retain the conventional range-separated path.

### T3. Make generalized consumers fit the same physical task

- [ ] T3.1 Support bounded RI/direct generalized J/K with all four complex
  spin blocks. Do not substitute a UHF/UKS calculation.
- [ ] T3.2 Keep HF grid-free and support bounded GKS grid execution while
  preserving the existing collinear/noncollinear functional contracts.
- [ ] T3.3 Connect generalized RSH to the same full/SR/LR exchange semantics;
  expose any remaining unsupported mode explicitly.
- [ ] T3.4 Check mixed-spin densities, different occupied/virtual dimensions,
  same-shape input changes and independent J/K selection against an oracle.

### T4. Close numerical, AD and performance gates

- [ ] T4.1 Pass small value, complex-array JVP/VJP, transpose and two-step
  finite-difference checks before interpreting large-system timings.
- [ ] T4.2 Preserve supported geometry, neural-XC fixed-density/explicit/
  implicit, NNAO coefficient/exponent and existing force-loss regressions.
- [ ] T4.3 Execute the fixed large matrix with independent PySCF counterparts;
  compare only matched, converged tasks and separately time targeted probes.
- [ ] T4.4 Publish parent/candidate raw repetitions, memory, capability gaps,
  local commits and source provenance; verify the original workspace unchanged.

## Acceptance gates

| Object | Gate |
|---|---|
| Organization-only value regression | Energy difference <= 1e-10 Ha versus the same-input parent |
| Matched independent SCF | Energy difference <= 1e-7 Ha and all original SCF residual gates |
| Small J/K array values | abs <= 1e-10, rel <= 1e-10 against explicit contractions |
| Array JVP/VJP and transpose products | abs <= 1e-9, rel <= 1e-8, including complex nonsymmetric inputs where supported |
| Complete nuclear derivative | <= 1e-5 Ha/Bohr; match integral screening, basis and grid response |
| Model/basis derivatives | Retain existing fixture units and gates; new normalized parameter directions use two FD steps 1e-4 and 3e-5, abs <= 1e-6 and rel <= 3e-4 |
| Physical updates | Fresh and reused-program results agree; changed parameters produce changed physical results |
| Memory | No oversized four-center resident tensor in the large direct/RI paths; process guard is respected |
| Performance | At least three warm samples for named representatives; retain only a measured runtime/memory/correctness improvement with disclosed tradeoffs |

SCF-root differences must be diagnosed using matched inputs, occupations,
densities and cross-evaluation; a converged but energy-mismatched pair is not
a valid speed comparison. Failure of a newly supported capability is an open
task, not grounds to remove its case. No overall speedup or full AD claim is
inferred from a local kernel or from a forward-only result.
