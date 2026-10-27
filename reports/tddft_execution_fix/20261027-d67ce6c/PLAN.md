# TD response execution implementation

Baseline: d67ce6c (production sources identical to 33a7cab). Original and
probe worktrees remain intact. All scientific runs use the authorized remote
V100 allocation, at most one GPU and 64 logical CPUs; local checks parse data.

- [x] Correct evidence and freeze the implementation baseline.
  - [x] Re-summarize benzene: first solve succeeded, repeated process timed out.
  - [x] Separate root-residual acceptance, phase completion and process status.
  - [x] Record current source/library/environment and independent time budgets.
  - [x] Capture exact same-signature compile signatures and misses.
- [x] Stabilize numerical response entry points without stale physical state.
  - [x] RED/GREEN same/new-object, dynamic input and structural-change tests.
  - [x] Preserve explicit model parameters, SCF/array/geometry derivatives.
  - [x] Verify forward and backward executable reuse separately.
  - [x] Measure parent versus candidate, including first execution.
- [x] Explain and improve solver workload and grid preparation.
  - [x] Runtime purpose, effective columns, iteration/subspace and root residuals.
  - [x] Identify a safe repeated-work candidate from real benzene batches.
  - [x] Retain original guesses, guards, dimensions and precision.
  - [x] Verify partial candidate and final combined effects.
- [x] Unify packed J/K and formalize device boundaries.
  - [x] CPU and GPU budgeted public packed contracts, selective J/K and AD.
  - [x] GPU tiles, measured workspace and clean peak memory.
  - [x] CPU RI preparation, explicit GPU residency and freshness/AD tests.
  - [x] Independent commits, report, measurements and reproduction commands.

Acceptance closed against final-source CPU1/CPU8/GPU records, source_audit.json and final XML suites.
No merge or push; allocation 110155 was released.
