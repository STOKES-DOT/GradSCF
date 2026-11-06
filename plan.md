# HF/KS performance optimization — fixed TDD plan

Status: IN PROGRESS. Baseline: `58b01bb3c2e0d9efb3fd9298cba7686746e597b3`.
Date: 2026-11-06. Worktree: `/Users/jiaoyuan/.codex/worktrees/5fbb/GradSCF`.
Branch: `perf/scf-100-grid3-20261105`.

## Contract and change control

This is the single plan for the next optimization round. The scope, acceptance
criteria and TODO identifiers below are frozen. Update status/evidence in place;
record any necessary scope or threshold amendment explicitly at the bottom.
Never replace a failed gate with an easier one after seeing the result.
Each TODO has at most one child level and at most four children.
A task is complete only after its stated tests and applicable measurements pass.
Planning, source inspection and job submission are not validation.

Production scope: RHF, UHF, GHF, RKS, UKS and GKS, through shared integral,
grid/XC and solver preparation code. Do not create six parallel implementations.
Cover conventional/direct and true auxiliary-basis RI consumers; do not switch
the conventional benchmark to RI to claim an implementation speedup.

All currently registered functional categories are in scope: LDA; GGA; kinetic
energy-density MGGA; Laplacian MGGA; global hybrids (including hybrid MGGA);
long-/short-range separated hybrids; recipes with dispersion/nonlocal terms;
and traditional custom_xc and neural/global functionals. Named recipes and
aliases must be enumerated from the existing registry. Metadata existence does
not prove end-to-end support: retain and separately test explicit rejection for
unsupported combinations (e.g. incomplete additional terms/PT2 or spin modes).
Do not silently drop NLC, dispersion, exchange or post-SCF terms. This is a
performance project, not authorization to claim previously absent physics.

Neural XC: preserve fixed-density, explicit and implicit training. NNAO:
preserve existing parameterization, masks, normalization and supported derivative
paths. No stop_gradient, frozen numerical cache, NumPy conversion of tracers,
or stale SCF result may be introduced to improve timing.

## Measured starting evidence

Evidence: `reports/scf_cpu_gpu_stages/20261106-58b01bb3/`.
96-atom beta-carotene, 296 electrons, Cartesian def2-SVP, 880 AO,
PBE, grid_level=3, 1,113,968 grid points, conventional direct J.
CPU64 means 64 logical / 32 physical cores in the recorded allocation.
Native libcint is CPU-only; GPU J uses the existing CPU callback.

| Quantity | CPU | RTX 5090 |
|---|---:|---:|
| Clean warm full call (one sample), seconds | 645.248809 | 266.680955 |
| SCF cycles | 26 | 22 |
| Production direct J fixed-density median, seconds | 10.085081 | 10.322627 |
| Whole-grid density/XC/Vxc median, seconds | 10.295625 | 3.150912 |
| Diagnostic AO cache materialization, seconds | 102.926933 | streaming |

Keep clean versus diagnostic, call versus kernel, initialization versus warm,
and worker elapsed sums versus wall time distinct. Raw diagnostic timings have
measured instrumentation overhead. Do not infer an asymptotic scaling law here.

## Frozen acceptance gates

1. Every behavior change: write a meaningful failing test, execute RED, then
   implement the minimum change, execute GREEN and narrow regressions. Save both
   logs with source/library hashes; skips do not satisfy RED or GREEN.
2. Array kernels: independent dense oracles, J-only/K-only/J+K, real/complex,
   nonsymmetric and batched densities, Cartesian/spherical where supported.
   Exact rearrangements: atol/rtol 1e-10; dense AD comparisons 1e-9; directional
   finite differences at two steps, generally error <=1e-7. Preserve tighter
   existing thresholds when already specified by a fixture.
3. Screening: hold the configured cutoff fixed. Derive bounds for the requested
   products. Positive-cutoff selection-specific masks may differ from the old
   union mask; document the change, validate an independent mask oracle and
   molecular accuracy. Unscreened values/AD remain unchanged. A local quartet
   bound is not a global error certificate. Geometry AD at cutoff=0 and fixed-
   reference density AD are separate acceptance paths; preserve existing guards.
4. Molecular SCF: finite energy/matrices, convergence and actual orbital
   residual; energy difference <=1e-7 Ha; nuclear gradient <=1e-5 Ha/Bohr.
   Do not loosen the input SCF tolerances or reduce orbital space/grid size.
5. Large anchor: atom guess, max_cycle=200, energy 1e-10 Ha, density RMS 1e-8,
   orbital gradient 1e-7 Ha, direct cutoff 1e-12, max_memory=180000 MB, float64.
   Final promotion requires one cold and >=3 warm calls, same geometry/guess,
   no converged restart. A partial run stays partial. Record compilation,
   intor/action counts, RSS, GPU allocator metrics, grid/factor capacities,
   raw samples, medians and dispersion. Main-library timing is uninstrumented.
6. Neural/custom: value, parameter JVP/VJP, a real optimizer update, and supported
   geometry/mixed derivatives; parameter/geometry changes must invalidate numerical
   preparation. A local block optimization cannot change an opaque/global model's
   definition. Screen boundaries and rank changes remain explicit limitations.

## Test matrix and staged scale

| Layer | Coverage | Purpose |
|---|---|---|
| Shared kernels | dense/screened, selections, batches, transpose, derivatives | Fast TDD loop |
| Functional inventory | every recipe/alias; local/full-term capability contract | No silent family loss |
| Molecular families | SVWN, PBE, r2SCAN/SCAN, supported Laplacian representative, PBE0, supported hybrid-MGGA, CAM-B3LYP/LC-wPBE/HSE/wB97X-D; additional-term cases and explicit unsupported cases | Verify every execution category |
| Six facades | closed-shell water; open-shell small molecule; GHF/GKS spin blocks | Shared code actually reaches all six methods |
| External/neural | existing traditional custom_xc, local and global neural, NNAO fixtures | Preserve training and AD |
| Large anchor | beta-carotene/def2-SVP/grid3 PBE/direct, then representative RHF/hybrid/unrestricted and RI consumers | Production cost and convergence |

Use the existing registry/tests to select an actually supported representative
for each family before running it. Record missing support, never substitute PBE
for an unsupported meta/nonlocal case and mark it passed. Small tests cover the
matrix; the 96-atom calculation is not multiplied by all names/aliases.

## TODOs (fixed)

### HFKS-01 — Baseline, contracts and RED fixtures

- [x] 01a. Freeze plan, source/input fingerprints and registry-family inventory.
- [x] 01b. Add independent selected-product screening/call-count RED fixtures;
      verify the old native library fails for the intended reason.
- [x] 01c. Add real multi-block AO materialization/consumer tests and identify
      the 103-second preparation cost with bounded whole-grid evidence.
- [ ] 01d. Add preparation-lifetime tests: density update, geometry update,
      coefficient/exponent/model update and same-signature new object.

### HFKS-02 — Direct J/K and libcint work

- [x] 02a. Specialize the density screening bound for J-only/K-only/J+K;
      preserve fixed-reference linearity and AD, and avoid evaluating the
      unrequested product. Measure native calls and compare dense outputs.
- [ ] 02b. Audit shell-level symmetry and threading for compute-then-discard
      work; retain only an independently tested, measured local improvement.
- [ ] 02c. Improve the observed contraction/scheduling hotspot if warranted;
      preserve bounded private output, deterministic reduction and derivatives.
- [ ] 02d. Validate Coulomb/RSH, CPU/GPU bridge, six facades and large anchor;
      report clean before/after and changed call/iteration counts separately.

### HFKS-03 — AO cache and XC integration

- [x] 03a. Fix measured AO materialization/layout/copy waste; preserve the exact
      grid and differentiable geometry/basis dependence. Caller-owned arrays
      needed by AD cannot be donated or mutated.
- [ ] 03b. Improve shared rho/gradient/tau/Laplacian and Vxc contractions using
      device/memory-appropriate tiles; test all supported functional families.
- [ ] 03c. Make cache/stream choice account for preparation amortization and
      actual CPU/GPU ownership. First implement exact changes; any new numerical
      AO screening is a separate approximation experiment with derivative gates.
- [ ] 03d. Verify conventional and RI six-method consumers, custom/neural XC
      and complete 96-atom cost, including preparation and memory.

### HFKS-04 — Repeated preparation and final integration

- [ ] 04a. Bind normalization/ENV once per parameter state, outside the SCF
      density loop but inside the differentiable energy graph. Do not infer
      repeated runtime work solely from Python source calls.
- [ ] 04b. Reuse proven invariant topology/preparation; key geometry-dependent
      objects by actual geometry, basis and Coulomb operator. Separate program
      caching from numerical caching; prioritize measured costs.
- [ ] 04c. Run the narrow six-method/all-category + supported AD/training
      regressions and final CPU/GPU clean comparisons, without a Cartesian-product
      large-system campaign. No unsupported combination may count as passing.
- [ ] 04d. Review this plan, remove failed/no-benefit candidates, record all
      remaining boundaries and produce local reviewable changes. Merge/push
      requires a separate user instruction.

## Resource and execution rules

All numerical tests/builds run on jy_gpu_2 under Slurm. Read the cluster manual
and check live allocations before every new submission. At most one allocated
V100/5090 (or another explicitly authorized equivalent), CPU64, 256 GiB host;
heavy tests serial. No login-node or local Mac scientific runs. Individual
calculation cap 6000 seconds; default validation round <=2 hours. Process RSS
hard guard 200 GiB for the large anchor. Keep a visible job ID and immutable logs.
A job allocation is not a completed test. Retain original worktree user changes.

## Review log (append evidence; do not rewrite the contract)

- 2026-11-06: User authorized freezing this TDD plan and starting implementation.
  Existing isolated worktree selected; production src is clean at the baseline.
  First target is HFKS-01b/02a. No large parameter grid is being restarted.

- 2026-11-06, HFKS-01b: RED observed on job 116507: four intended call-count
  failures, zero collection errors. Test file `test_direct_selected_screen.py`.
  Implementing 02a; J+K and cutoff=0 behavior unchanged. Positive-cutoff J-only/K-only
  masks are now product-specific, with a dense oracle and fixed-reference AD checks.

- 2026-11-06, HFKS-01a/01b/02a checked: frozen catalog/registry fingerprints in
  `reports/hfks_optimization/20261106/`; RED 4 intended failures, fresh-build
  GREEN 88 CPU + 27 consumers + 6 CUDA = 121 passed. Six facades agree between
  direct and dense on the small fixture. Native call counts 32 -> 21 (J),
  32 -> 29 (K); joint remains 32. See HFKS-02a.md and checkpoint.json.
  The stale incremental build is retained as rejected evidence. HFKS-02d and
  all grid/remaining-family/training/large-scale gates remain unchecked.

- 2026-11-06, HFKS-01c/03a candidate (not finally accepted): RED compiled-operation
  gate 2 failed / 7 passed; common Cartesian coordinate powers still used generic
  floating power. Candidate replaces integer powers 0..4 by exact multiplication,
  retaining the existing custom JVP and higher-power path. Remote observations:
  CPU tests 47 passed, GPU 12 passed, six facades 6 passed; independent whole-grid
  energy/Vxc and 257-point AO comparisons passed. AO cache preparation median
  103.415 -> 33.983 s. CPU cached XC 10.529 -> 11.419 s (no speedup claimed);
  GPU streamed XC 3.151 -> 2.280 s. See ao_grid/observed_summary.json.
  Compiled buffers do not show a full 31-GB cache copy on each iteration; the
  confirmed first target is AO production, not an assumed cache-copy bug.
- 2026-11-06, maintenance boundary: job 116546 completed the kernel round. Job
  116579 started clean full-SCF one-cold/three-warm validation; GPU calls 0/1
  passed, subsequent calls remain unverified. SSH then timed out, and the user
  confirmed maintenance and requested local organization. No resubmission or
  further remote polling until the user reports restoration. Keep 01c/03a/03d
  unchecked until raw evidence and the applicable final checks are reconciled.

- 2026-11-06, recovery and HFKS-01c/03a acceptance: existing job 116579 had
  COMPLETED (0:0) during maintenance; no calculation was repeated after recovery.
  All 66 XML-recorded tests and CPU/GPU 4/4 full-SCF calls passed; source hashes
  and the uninstrumented production library were verified. Raw files are now
  synchronized. CPU AO preparation 103.415 -> 33.983 s; GPU streamed XC
  3.151 -> 2.280 s. CPU cached XC 10.529 -> 11.419 s remains an open consumer
  performance issue, not a claimed improvement. Whole-call warm medians are
  CPU 459.674 s / GPU 232.254 s, including both recent changes; previous-run
  comparisons have different affinity and CPU iteration count. See
  `reports/hfks_optimization/20261106/ao_grid/verification.json` and README.md.
  Only 01c/03a are newly checked. Remaining family/RI/training and preparation-
  lifetime gates remain open; this is not completion of the full project.
