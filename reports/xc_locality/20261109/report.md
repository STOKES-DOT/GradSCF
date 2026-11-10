# XC locality and conditional density execution: implementation and acceptance

This bounded round is complete. It implements the three approved axes in the
existing isolated worktree: spatial grid/AO locality, conditional density
representations, and measured cache/tile/device policies. The production code
is in commit `db9700ff`; no merge or push was performed. The fixed checklist is
[XC_LOCALITY_PLAN.md](../../../XC_LOCALITY_PLAN.md). Older HFKS reference-recovery
and full large-system matrix gates remain separate.

## Baseline and execution

Baseline: `20a47391`. Worktree: `/Users/jiaoyuan/.codex/worktrees/5fbb/GradSCF`,
branch `perf/scf-100-grid3-20261105`. All 555 candidate Python source files were
matched to the final remote source receipt. All four complete-SCF processes
loaded the same unchanged native binary hash, recorded in [manifest.json](manifest.json).
The final delivery additionally clarifies one size-gate comment; its Python
AST is identical to the tested source, and both hashes are recorded.
No numerical kernels, tests or builds ran on the local Mac or login node.

Hardware: n006, AMD EPYC 9654, 64 logical CPUs spanning 32 physical cores,
one RTX 5090, 256 GiB allocation. Native workers: 64; BLAS/OpenMP threads: 1;
JAX/jaxlib 0.10.0, float64. libcint/direct J/K stays on CPU; the GPU cases
explicitly stage JAX arrays and run grid/XC/SCF algebra on the allocated GPU.
Heavy work was serial. Jobs 118316 and 118350 have ended and released resources.
Slurm accounting is disabled; process receipts, controller logs and the
successful `afterok` successor establish completion. The final successor was
observed as `COMPLETED`, `ExitCode=0:0`.

The anchor is 96-atom beta-carotene (C40H56), Cartesian def2-SVP, 880 AOs,
296 electrons, grid level 3 with 1,113,968 points, PBE0, conventional direct J/K.
All 880 orbitals are retained. The full geometry/configuration is in [cases](cases/).
The frozen controls are energy tolerance 1e-10 Ha, density tolerance 1e-8,
orbital-gradient tolerance 1e-7 Ha, direct-J/K cutoff 1e-12, max_cycle 200,
atom initial guess and max_memory 180000 decimal MB. These controls are the
same before and after. The general library default memory budget is unchanged.
The new candidate alone opts into AO screening at 1e-14.

## Production changes

### Spatial grid and local AO contractions

`integrals/grids/locality.py` provides reversible spatial ordering and
current-parameter Gaussian envelopes over each grid block. Coordinates and
weights move together; no quadrature points are discarded. AO centers,
exponents and coefficients remain dynamic. Only selected AO columns are
actually evaluated, then the matching density submatrix is contracted and
the local potential is scattered into the full AO matrix.

`_LocalAOGrid` reserves dense overflow workspace, uses fixed capacities and
recomputes support for current parameters. Overflow evaluates the exact full
AO block. Two fixed capacities remove padding: 256 or 512 columns for the
880-AO anchor. At 1e-14 there are 272 blocks, 123–390 active AOs (median 249.5),
149 blocks use 256 columns and 123 use 512; none overflows. These counts are
from a separate support probe, while runtime-width tests verify compact
execution. They are not integral counts.

The generic `block/map/reduce` interface preserves original grid ordering and
semantics for global/external/neural definitions. RKS, UKS and the existing
supported GKS forms share local matrix gather/scatter. Conventional and true
auxiliary-RI SCF consumers use the same XC path.

### Conditional density representation

The shared density helper may evaluate occupied orbitals when finite real
hints exactly reproduce the current density and the occupation tail is zero.
Mixed/stale/general density inputs keep the dense route. A custom JVP uses
the full physical AO/dAO/D mapping, retaining arbitrary density perturbations
and counting D(C,occupation) response once. Hints do not freeze physical data.

Automatic hints are limited to MGGA and avoid small blocks. The current
128-AO / 256-point gate is a heuristic, not a measured crossover. Performance
measurements cover 25/260/880 AOs, 4096 points and occupied fractions about
0.168–0.20; threshold-neighborhood and high-occupancy performance is unmeasured.
GGA and small GPU cases showed insufficient gains or regressions, so they do
not receive the automatic shortcut. No claim of a universally optimal gate is made.

### Cache, tile and device policy

The existing memory ownership/reservation machinery accounts for sorting,
support metadata, full fallback workspace and output matrices. No second
memory-budget interface or new integral backend was introduced. Both 2048-
and 4096-point tiles were tested; 4096 was faster for the selected cutoff on
both devices, so no blanket smaller-tile policy was added. CPU AO caching
remains available on the exact route when admitted; it is not forced on GPU.

## TDD and differentiation

Three concrete derivative issues were covered by RED/GREEN tests:

1. Inactive padded alpha=0 was normalized before masking, causing nonfinite
   higher derivatives in both old dense and local AO paths. Only inactive
   primitive entries now receive a positive sentinel before normalization.
2. A zero active contraction coefficient can have a nonzero parameter response.
   A unit-coefficient envelope preserves its support; physical coefficients
   and trainable parameters are not frozen.
3. A real zero-weight point is not padding. An explicit index-derived mask
   alone excludes tail padding, preserving weight JVP/VJP. The anchor itself
   contains 5,578 genuine zero-weight points. The final correction was tested
   before any candidate complete-SCF timing; the frozen baseline was unchanged.

| Accepted gate | Evidence | Scope |
|---|---|---|
| Final zero-weight/provider regression | CPU 71, GPU 15 passed after 12 numerical RED failures | Real/interior zero weights, signed weights, tail padding, energy/potential JVP/VJP and two FD steps |
| Compact local consumers | 78 passed | LDA/GGA/hybrid/RSH/MGGA representatives, RKS/UKS/GKS supported forms, actual AO widths |
| Conditional hints | 38 passed; separate molecular MGGA 4 and grid-JIT 1 passed | Arbitrary dD, stale/mixed hints, C/occupation chain, center/exponent response and higher AD |
| Capacity buckets | 53 passed | Correct compact widths, exact overflow and default path |
| HF/KS facade/family regression | 33 passed | Six facades, conventional/true RI, independent small PySCF references and raw-Fock stationarity |
| Public screened SCF | 6 passed | RKS/UKS/GKS, direct and true RI, full orbital space and stationary energies |
| External neural XC | 8 passed | Existing fixed-density/explicit/implicit training and force-supervision mixed gradients |
| NNAO | 1 small actual MACE two-geometry update/checkpoint test passed | Model and log-exponent FD, CPU integrals/GPU model and implicit solve; not a large training campaign |
| Other GPU operator AD | 59 passed | Value and array-AD regressions before the final zero-weight correction |
| Existing stability/boundaries | 7 and 26 passed | Default stability remains available; positive-screening export is guarded |

Counts overlap and must not be summed as unique tests. Full XML and failed
attempts remain in `raw/`; [regressions.json](regressions.json) records accepted
counts and hashes. One earlier combined process ended with exit 139 in JAX
backend compilation. Its cause is unestablished; it is not relabeled OOM or
success. Fresh isolated processes passed all required groups. A first harness
attempt lacked a shared helper; this was restored without numerical changes.

Hard screening is a default-off, piecewise-smooth approximation. The AO-channel
cutoff in atomic units is not a global energy, force or arbitrary-order
derivative error bound. The same selected support is used in primal and AD;
no new stop_gradient is applied to physical parameters. Use cutoff zero where
screened convergence has not been established. The local-provider fixtures
cover JVP/VJP/HVP and directional FD; they do not certify cutoff crossings,
a 96-atom nuclear Hessian or screened full NNAO force training.

## Measured performance

All warm numbers below are synchronized medians of three samples. The complete
SCF uses a fresh molecule/input and the same atom guess each time. Ordinary
same-geometry grid and JIT caches may persist; converged states are never reused.

### Complete PBE0 SCF

| Device | Timing scope | Before (s) | After (s) | After/before |
|---|---|---:|---:|---:|
| CPU | First SCF call | 515.9554 | 346.9422 | 0.6724 |
| CPU | Warm complete SCF | 454.5869 | 301.3738 | 0.6630 |
| GPU | First SCF call | 286.4265 | 272.3091 | 0.9507 |
| GPU | Warm complete SCF | 222.8001 | 182.2576 | 0.8180 |

Warm raw samples (seconds):

- CPU before: 473.9363, 454.5869, 414.9044; after: 301.3125, 301.3738, 301.5663.
- GPU before: 222.8001, 223.0670, 221.8111; after: 182.2576, 183.2822, 182.2442.

CPU warm elapsed time fell 33.7%; GPU warm elapsed time fell 18.2%. CPU baseline
variation is substantial; these are bounded single-node observations, not a
universal speedup. CPU iterations changed from 20 to 19 under unchanged
criteria; GPU used 19 before and after. CPU speed ratios include that one-cycle
difference. No kernel ratio is substituted for a complete-SCF ratio.

All 16 calls passed. Maximum energy difference from the saved unscreened
GradSCF baseline is 1.59e-10 Ha. The maximum independently computed orbital
residual is 4.31e-8 Ha. Residuals use each result's unextrapolated Fock under its
chosen XC definition, not a DIIS extrapolate or a manufactured diagonal Fock.
A separate unscreened Fock at the final screened density was not evaluated.
Small independent PySCF comparisons are a different gate; this table is
GradSCF before/after, not a new large-system PySCF timing comparison.

| Complete-SCF memory metric | Before | After |
|---|---:|---:|
| CPU process sampled peak RSS, decimal GB | 36.0885 | 5.0921 |
| GPU-run host process sampled peak RSS, decimal GB | 4.1651 | 4.7494 |
| GPU JAX allocator peak bytes-in-use, decimal GB | 1.0679 | 1.1256 |

CPU RSS fell 85.9%, primarily avoiding the full AO cache in this selected
path. GPU host and allocator peaks increased modestly; this is a throughput
tradeoff, not an all-device memory-reduction claim. RSS is sampled every 0.2 s;
ru_maxrss is recorded separately. GPU allocator high-water is not total reserved
pool memory or whole-card consumption. Unknown metrics remain null.

The first-call timer starts after Python imports/native registration and
includes input preparation, staging and the first solver execution/compilation.
The four-call process wall times are CPU 1863.179 -> 1255.311 s and GPU
1100.669 -> 1019.489 s. Additional process overhead outside those timed calls
is about 3.796/4.116 s CPU and 146.564/199.396 s GPU before/after. This includes
initialization, validation, I/O and GC; it was not isolated into individual
causes and must not be called pure backward or compilation cost.

### Fixed-density complete XC operators

| Operator scope | CPU before -> after (s) | GPU before -> after (s) |
|---|---:|---:|
| PBE0 semilocal XC, admitted CPU cache / GPU raw stream -> local blocks | 9.9384 -> 6.5497 | 2.2814 -> 1.0340 |
| Whole r2SCAN XC at saved PBE0 C/D/occupations, no hints -> hints | 30.4291 -> 19.6127 | 8.2773 -> 5.4148 |

The r2SCAN rows are not r2SCAN SCF. Both actual saved-state hint guards passed;
D was not reconstructed to force a match. CPU/GPU use their own saved states,
so no cross-device ratio is inferred from these two r2SCAN rows. Hinted energy
errors are 8.53e-14 Ha CPU and zero to recorded precision GPU; maximum potential
errors are 1.78e-15 Ha on both. Whole-operator gains are about 1.55x/1.53x,
smaller than the approximately 4x synthetic density-helper result.

For final PBE0 locality, energy errors are 9.04e-12 Ha CPU / 1.14e-13 Ha GPU;
maximum potential errors are 1.55e-15 / 3.55e-15 Ha. Registered operator gates
were 1e-8 Ha energy and 1e-9 Ha maximum potential entry. Local-order preparation
costs 1.017 s CPU / 12.948 s GPU, versus 33.183 s for the CPU AO cache.
Local lowering/compile costs are 0.373/1.284 s CPU and 0.380/6.805 s GPU.
Those costs are separate from warm operator timing and prevent a cold-start
claim based on warm arithmetic alone.

Decision experiments retained for audit, before the final zero-weight fix:
CPU local 4096/1e-14 was 6.602 s, versus 6.970 s at 2048 points; GPU 1.034 versus
1.060 s. A looser 1e-12 cutoff was faster, but was not selected as the final
cutoff or made a default. The final zero-weight correction retained the
selected 4096/1e-14 gains. MGGA orbital hints increase compiler temporary
estimates: CPU 634.5 -> 836.4 MB and GPU 432.7 -> 490.3 MB. These estimates
are not measured RSS, and the same-process variants do not have isolated RSS peaks.

## Public use and boundaries

```python
mf = dft.RKS(mol, xc="pbe0", grids_level=3,
             grid_screen_tol=1e-14, max_memory=60000).run()
```

`grid_screen_tol=0.` remains the exact default. Positive screening currently
supports SCF energy/orbitals/density; reference export and orbital stability
are explicitly rejected until they share the same screened response definition.
UKS returns its normal SCF result without exporting an inconsistent reference.
Existing GKS spinor-response limitations remain. Global/external neural models
keep original ordering and definitions; they are not assumed locally additive.
This round does not establish screened TDDFT/GW/BSE or exported-reference nuclear
response. It also does not turn a measured small regression set into a full
large-molecule/family/RI validation matrix.

## Delivery and reproduction

- [Measurements and raw samples](results/measurements.jsonl),
  [before/after CSV](results/before_after.csv), [machine-readable summary](results/summary.json).
- [Source/environment manifest](manifest.json), [acceptance](acceptance.json),
  [regression receipts](regressions.json), [commands](commands.md).
- Five thin scripts in `benchmarks/probes/scf_100/`: `density_routes.py`,
  `local_xc_benchmark.py`, `local_scf_benchmark.py`, `whole_mgga_benchmark.py`,
  `summarize_xc_locality.py`.
- Original XML, JSONL, source manifests and scripts: `raw/` locally and the
  remote archive in the manifest. Large arrays and raw logs are excluded from Git.
- `harness_guard.patch` records the pre-existing monitoring fix used by this
  round, separately from numerical changes. Main-workspace source/index and
  unrelated worktree changes were preserved.

No further numerical tests are queued for this round. The measured CPU/GPU
SCF and whole-XC improvements support retaining these local patches. Threshold
crossover tuning and extension of positive screening to response are separate
future tasks, not hidden claims of completion here.
