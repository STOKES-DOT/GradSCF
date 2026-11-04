# Shared SCF solver implementation and acceptance

2026-11-02. Worktree: perf/bse-tddft-grid3-20261101, starting HEAD709e4ea.
This continues the measured phenanthrene SCF diagnosis. Existing TDDFT changes
and diagnostic artifacts are retained; the release worktree stays untouched.
The user explicitly authorized implementation of the following shared design.

## Boundary

RHF/RKS, UHF/UKS and GHF/GKS already share Fock-building families, but have
three separate numerical SCF loops. One array/PyTree solver will own the scan,
DIIS state, damping schedule, convergence checks, freezing and scalar history.
It will live in `gradscf.solvers.scf`, reuse `solvers.nonlinear.diis`, and import
no electronic-structure modules. Method modules retain Fock/XC/integral physics.
Thin physical adapters in `scf/iteration.py` will serve both built-in methods
and DifferentiableSCF, without a second implementation of the iteration.

The shared API is `solve_scf`, `SCFState`, `SCFConfig` and `SCFResult`. Physical
arrays and model parameters remain dynamic. Actual callbacks supply energy,
raw Fock, auxiliary J/K state and diagonalization. Retain scan reverse-mode,
fixed-cycle conv_tol=0, direct-J/K fresh-state verification, complex residual
packing and existing static per-family damping/DIIS cadence. No new stop_gradient.

Convergence uses the occupation-weighted MO commutator, including fractional
occupations and conjugate transposes. This preserves restricted factor2,
unrestricted joint-spin normalization and generalized spinor normalization.
Exact energy differences remain exact; opaque/neural callbacks are never
silently replaced by a quadratic or local-grid approximation.

## TODO

- [x] S1: Establish contracts and a failing test before production edits.
  - [x] Freeze source/native hashes and focused six-method/training baseline on Slurm.
  - [x] Test shared SCF state, auxiliary data, freeze/selection and fractional/complex residuals.
  - [x] Test explicit JVP/VJP/HVP and same-shape dynamic data updates.
  - [x] Record the expected RED result and keep this plan in scope.
- [x] S2: Migrate all actual numerical consumers.
  - [x] Implement the shared scan using the existing canonical DIIS.
  - [x] Migrate R/U/G physical adapters, so all six public methods use it.
  - [x] Migrate DifferentiableSCF; preserve fixed_density/explicit/implicit and legacy iterate-selection contracts.
  - [x] Remove redundant loop/DIIS/convergence implementations; update adjacent RO imports only.
- [x] S3: Correct measured precision problems without changing the physical task.
  - [x] Apply the common occupation-weighted stable MO residual, with narrow ill-conditioned tests.
  - [x] Probe orbital orthonormalization and energy-difference precision independently.
  - [x] Retain only a validated exact-arithmetic-equivalent improvement; preserve existing clipping boundaries.
  - [x] Rerun the original cation/TZVP/grid3/RI case with unchanged1e-12/1e-11/1e-10 gates.
- [ ] S4: Verify the six methods and training before reporting completion.
  - [x] Run R/U/G HF and KS numerical/complex/representation regression groups.
  - [x] Run existing three-mode neural/custom XC, fractional occupations and failure/optimizer-state tests.
  - [x] Run supported geometry/force-loss mixed derivatives and NNAO parameter/HVP/update fixtures.
  - [ ] Record source/import/native provenance, timings/RSS, compile behavior and limits; release resources.
- [ ] S5: Complete the user-requested broad closing matrix.
  - [x] Freeze 60 systems, 12 per basis, 5--150 atoms and 15 non-noble elements H--Ar.
  - [x] Register balanced SVWN/PBE/PBE0/CAM-B3LYP, grid3 and applicable six-method/PySCF pairs.
  - [ ] Run primary cases and representative cross-basis/XC cold+warm controls with explicit resource/capability statuses.
  - [ ] Report convergence, final residuals, energy differences and matched-task timings; no mixed-method average.

## Verification and recovery

All numerical tests run only under Slurm on jy_gpu_2. Resource ceiling: one
V100/NV5090,64 logical CPUs,256GiB; tests use CPU8 first, largest case CPU64.
Each large calculation is bounded to6000s. Reuse the existing verified Linux
libcint binary because this work changes no native source. Baseline archive
SHA256: `7ceb291faf5e30f647e42cb2338d0659054abd95e4dbe1ef2299ff35ae1c2e19`.

Focused gates include SCF autodiff, DIIS scaling/state, convergence policy,
direct fresh-Fock refresh, GHF/GKS complex invariance, external neural XC,
force training, and NNAO RI metric/parameter response. A failed group is
investigated before widening coverage. Numerical tolerances and outputs remain
explicit; a completed process is not a scientifically converged state.

Structural migration and precision changes are measured separately. The frozen
remote baseline remains runnable throughout; a failing candidate is corrected
in the isolated worktree without resetting release or deleting user data.
No merge or push is part of this task.

Existing unsupported capabilities stay explicit: spinor GKS neural/custom-XC
training, GKS DF/direct/RSH/MGGA and RI geometry tracing are not created by
sharing the numerical loop. Existing molecular RKS/UKS model training must
continue working. ROHF/ROKS retain their common-orbital physical algorithm.

## Progress evidence

Allocation114687, NV5090/64 logical CPUs/256GiB on n006. Frozen baseline83
checks passed46.37s. Initial shared implementation120 checks passed76.68s.
Final precision snapshot SHA256
`91fdbb6eefaac9048fda3132f504a4d85da9ab3e9d555b4f9f00c0a14b986516`:
99 core/complex/AD/representation/adjacent-RO tests passed84.36s;20 optional
atom-guess tests passed25.74s;11 neural/force tests passed143.76s;8 NNAO
parameter/metric/HVP tests passed64.26s;one small-MACE update/checkpoint
fixture passed20.25s. Counts are groups, not a claim that all are disjoint
from previous runs. GPU-only array tests were stopped during CUDA/NUMA init;
no GPU numerical acceptance follows. The earlier combined neural process
aborted during compilation; the required nodes passed in isolated groups.

New optional `init_guess='atom'` is native isolated-UHF density superposition,
not spherical SAD. It is bounded, immutable/cache-keyed and restricted to six
validated named all-electron basis families H--Ar; defaults and model/raw-basis
paths are unchanged. Its performance benefit is a separate registered ablation.

Closing plan-v2 registers858 independent configurations including cross-controls;
primary cases balance the four XC families15 systems each. The final matrix
uses conventional packed below128AO; larger R/U use explicit weigend RI;
RSH R/U use direct; GHF/GKS retain their supported packed representation.
Each reference uses the matching physical approximation. Unsupported or
resource-limited combinations remain visible and cannot enter speed ratios.

2026-11-04: recovered original strict cases. Neutral RKS:82cycles/493.779866s
baseline versus51cycles/256.654833s candidate, energy difference3.61e-11Ha;
single cold observations, not a warm/repeated speed claim. Cation candidate
passed69cycles while baseline failed at200cycles. S3 is complete.

Allocation115757 replaces the stopped Nov2 allocations with the same resource
unit on n006 for48h. Formal matrix started. It exposed an existing eager GHF/GKS
Hermiticity gate problem: Cl/TZVP hcore finite, scale146.9Ha, asymmetry1.10e-13Ha,
old fixed100eps tolerance2.22e-14Ha. Separate scaled-tolerance patch:23 GHF/GKS
tests pass16.53s (2 RED beforehand); all five affected initial cases pass
correction attempts. Original outputs retained. Remaining calculations use
candidate-roundoff-20261104, whose only production change from frozen precision
candidate is this validation. Numerical methods and convergence gates unchanged.

NFS visibility caused stale derived statuses, recovered from completed guard
records without rerunning calculations. Correction-overlay metadata:8 PASS0.11s.
Initial43 attempts conservatively excluded from timing ratios due to overlap
with small diagnostics, but their scientific results remain usable. Formal
matrix still running; S5 and final resource cleanup remain open.
