# Molecular G0W0 memory and execution plan

Approved scope: isolated molecular G0W0 optimization from a10b865, preserving
the original CD equation, finite eta, requested spaces, QP seeds/certification,
and existing AD. No automatic merge or push. Numerical tests run only inside
one remote Slurm allocation: V100/NV5090, 64 logical CPUs, 256 GiB, at most 48 h.
Each calculation is limited to 6000 s; default measurement round is 7200 s.

Each item has one sublevel and at most four tasks. Review this list after each
completed item. Source implementation, numerical validation and performance
acceptance are separate states; a running job is not a completed test.

- [x] GW-01: Freeze the baseline and obtain reproducible measurements.
  - [x] Create an independent worktree and record original index/status/patch hashes.
  - [x] Verify remote source, imports, native library and software/hardware provenance.
  - [x] Run small exact residue and independent water G0W0 baselines.
  - [x] Record cold/warm, stage timings, real RSS, dimensions and QP certification.
- [x] GW-02: Integrate the shared memory budget and explicit data lifetimes.
  - [x] Resolve max_memory from the reference or the existing 60000 MB default.
  - [x] Account for residents, complex screening/factors, RHS and QP batches.
  - [x] Plan static tiles and optional caches without changing physical spaces.
  - [x] Reject a minimum workspace that cannot fit; test budget-invariant results.
- [x] GW-03: Consume residue screening through projected solves.
  - [x] Test then implement eps*x=Pi*b and immediate left-vertex contraction.
  - [x] Skip inactive poles safely while preserving branch/static conventions.
  - [x] Verify dense parity, frequency slopes, array JVP/VJP and QP diagnostics.
  - [x] Keep periodic q0/off-diagonal consumers correct; measure actual memory.
- [x] GW-04: Apply target/G/W windows before AO2MO.
  - [x] Resolve distinct target, G and screening spaces with original MO indices.
  - [x] Reuse gradscf.ao2mo for Qia, Qpm and required exchange blocks.
  - [x] Avoid full-pair W and full exchange matrix for selected target outputs.
  - [x] Test reordered/empty spaces, nonsymmetric arrays and spin channels.
- [x] GW-05: Reuse fixed-frequency preparation within its proper lifetime.
  - [x] Reuse each imaginary-frequency dielectric factor across RHS blocks.
  - [x] Separate frequency-independent, fixed-quadrature and trial-frequency data.
  - [x] Budget the required projected cache and use block AD checkpointing/recomputation.
  - [x] Test compile reuse and fresh results after same-shape parameter changes.
- [x] GW-06: Supply explicit true RI and bounded large-auxiliary solves.
  - [x] Allow independent libcint 2c/3c preparation after a non-DF SCF reference.
  - [x] Keep conventional/spectral choices explicit; do not silently alter them.
  - [x] Bound RI preparation/AO2MO without independently whitening auxiliary tiles.
  - [x] Use shared complex GMRES with true residuals; retain dense comparison.
- [x] GW-07: Validate derivatives and existing downstream consumers.
  - [x] Check orbital/factor/model JVP/VJP, transpose identities and two FD steps.
  - [x] Preserve MolecularGW full small-molecule geometry/QP/Z response.
  - [x] Verify an existing NNAO fixture, update and numeric-cache invalidation.
  - [x] Regress UGW, BSE handoff and shared evGW code without new physics.
- [x] GW-08: Complete scale tests, MOLGW comparison and delivery.
  - [x] Expand after small gates: water/def2-SVP, benzene/def2-SVP and
    water/def2-TZVP. CH4/aug-cc-pVDZ and naphthalene were not selected in this bounded run.
  - [x] Compare projected-residue/final snapshots and final pairs; CPU1/CPU64/GPU
    remain separate. The unfinished CPU8 pilot is excluded.
  - [x] Use original MOLGW with matched definitions; separate interpolation/rank/root differences.
  - [x] Deliver raw records, tests, local commits, limitations and rerun commands.

## Numerical gates

Exact product rearrangements: atol 1e-10 Ha, rtol 1e-9. Certified same-branch
QP differences: <=1e-8 Ha plus the original root residual test; compare Sigma,
Z, candidates/counts and coverage. Matched external QPs: <=1e-3 eV (otherwise
excluded from speed ratios). Existing complete geometry response: <=1e-5
Ha/bohr, two FD steps. Parameter derivatives have fixture-specific scales
registered before running. Branch boundaries/singular slopes remain explicit.

## Capability boundary

True RI at fixed geometry/metric supports existing orbital basis parameter AD.
It does not certify full RI geometry response. Auxiliary-center/metric and
whitening geometry derivatives are not added implicitly in this project. Keep
the existing complete small-system geometry path working; never freeze these
dependencies or describe parameter partials as complete nuclear derivatives.

## Commit and recovery

Keep baseline harness, projected residue, memory planning, selected factors,
RI supply and final validation independently reviewable. Reject experiments
without a measured benefit or a clear correctness/memory/maintenance purpose.
Revert only the independent failing patch, preserving evidence and the original
worktree; no automatic merge/push and no destructive cleanup.

## Checkpoint: implementation and narrow acceptance

- GW-01 baseline source: 1910 files checked; unchanged native sources and
  copied isolated library SHA256 c4756801f7cad8aa826760fb5d761ba7ea6f345606c17402401cc8f4ac6b384b.
- GW-03/04/07: projected products, early ordered windows, selected UGW and
  native NNAO -> implicit SCF -> nonlinear QP parameter training validated.
  Current narrow gate: 35 passed; earlier shared periodic/resolvent gate: 74 passed;
  geometry/compile: 3 passed; UGW/array AD: 11 passed. Final supplemental
  GMRES HVP and broad downstream smoke remain in the final report.
- GPU serial projected products were slower at small Q. Tile 4/8 experiments
  were not retained. The full production driver uses the original GPU dense
  batch only when its entire working set fits the common budget. Actual water
  GPU warm GW remained approximately 68 ms, with matching QPs and certification.
- GW-02/05/06 implemented and under final acceptance: selected data fit the
  budget, imaginary factors serve RHS blocks and both UGW frames, optional
  checked shared GMRES, independent libcint RI after direct HF. Full packed
  3c supply remains a declared bounded input, not an out-of-core provider.
- The CPU8 benzene attempt produced no QPs before resource replanning and
  was stopped after 14m31s. It is not a timeout result or an accepted speed
  comparison. The paired scale configuration is CPU64 with BLAS/OpenMP1 and
  native threads64; candidate runs first, unchanged physical inputs.
- Optimized benzene/def2-SVP completed: 120 AO, 438 auxiliary dimensions, all
  G/W states and 6 requested QPs; scientific gates passed. Parent CPU64 and
  external references are still pending here, not inferred from old reports.

## Recovery checkpoint

The preceding completed checks describe the preserved pre-interruption remote
snapshot. The temporary local tree subsequently disappeared; source/tests were
replayed into a persistent worktree and committed. Final numerical acceptance
is NOT closed: the weighted-factor fixture budget was corrected (60 ->100 MB)
and needs remote revalidation. The original complete raw dataset, final trace,
27-file remote byte comparison and allocation state require restored SSH. See
RECOVERY_STATUS.md, recovery.json and recovered_measurements.jsonl.

Do not treat source AST checks as numerical passes or these checkboxes as
certification of the restored source. No new remote jobs have been submitted
during the connection outage.

## Final acceptance (2026-10-30)

SSH restored. The retained failure traceback confirmed the weighted-factor
fixture budget issue. With 100 MB requested, the final narrow gate passed
35/35; geometry/compile passed5/5; GPU array/HVP passed2/2. Fresh accepted
benzene/GPU measurements and environment-accepted.json were downloaded.
All27 source/test hashes match the recovered checkout. Native actual CDLL
path and process mappings point into the isolated candidate; SHA c4756801... .
The independent stock MOLGW and explicitly RI-exchange PySCF numerical gates
pass without changing production equations or scientific tolerances.

The CPU8 pilot was resource-stopped, not accepted. CPU64 pairs and CPU1/GPU
water are reported separately. CH4/aug-cc-pVDZ and naphthalene were not run
in this bounded matrix; no scaling exponent or universal capacity claim is
made. Full packed3c/selected-cache capacity and full RI nuclear-AD boundaries
remain explicit implementation limits, not silently bypassed dependencies.

Allocation113327 released; squeue subsequently returned no active job.
No automatic merge or push. Final report/CSV/raw evidence supersede recovery
checkpoint status. All TODOs are closed for this documented core scope; no
conditional out-of-core/new RI geometry implementation is implied.
