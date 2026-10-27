# TD response execution: implementation and acceptance

## Scope and provenance

This round implements production changes to response execution, reusable grid
fields, shared packed J/K and CPU RI → GPU response. It does not change the
functional, grid, orbital windows, requested roots, starting guesses, solver
stopping rules, derivative guards or physical approximations. It does not
construct a dense transition-space A matrix.

Baseline: `d67ce6c8689ead3f79b7c6d9785754b5a82a1bf8` on the original
`release/v1.0.0` checkout. Its production sources equal the historical
`33a7cab` probe sources. The implementation is in
`/Users/jiaoyuan/.codex/worktrees/9515/GradSCF`, branch
`perf/td-response-execution-20261027`. The five production commits and ordered
patches are listed in [implementation_commits.json](implementation_commits.json)
and [experiments/](experiments/).

All chemistry ran in Slurm allocation **110155**, on **n002**, Intel Xeon Gold
6230, one V100-SXM2-32GB, at most 64 logical CPUs and 256 GiB allocation RAM.
The initial pending A800 allocation was cancelled before starting this V100
allocation. Main comparisons use CPU1; CPU8 uses affinity
`0,1,2,3,40,41,42,43`, four physical cores and eight logical CPUs. JAX/jaxlib
0.10.0, NumPy 2.4.4, SciPy 1.17.1 and float64 were used. See
[manifest.json](manifest.json) and [environment.json](environment.json).

The unchanged native C/C++ extension was reused, not rebuilt. It uses pinned
libcint 6.1.3 (`c72d30662ba08048940c9000cc352a220c867712`), with library SHA256
`c4756801f7cad8aa826760fb5d761ba7ea6f345606c17402401cc8f4ac6b384b`.
No native C/C++ source was changed in this round. Each end-to-end run records
its actual Python module and loaded native path/hash. Remote source archives
have `SOURCE_REVISION=d67ce6c` plus the recorded patches; they are not falsely
described as clean d67ce6c checkouts.

The original checkout remains at d67ce6c. Its index hash and all 4096 original
untracked file hashes are unchanged; see
[original_workspace_verification.json](original_workspace_verification.json).
No merge or push was performed.

## Measurement protocol and admissibility

All molecules use Cartesian def2-SVP, level-2 grids, five lowest singlet roots
(spin-conserving U-TDA for OH), SCF tolerances `1e-10/1e-9/1e-8`, Davidson
tolerance `1e-8` Ha and `max_memory=4096` decimal MB. CH₂O/benzene use genuine
auxiliary RI with **weigend**; water/OH use exact packed integrals. Complete
geometries, occupations, dimensions and settings are in [cases/](cases/).
Benzene retains **21 occupied / 99 virtual / 120 AO / 438 auxiliary** orbitals
and **99,480 grid points**.

Each clean configuration runs in a fresh subprocess. A first execution and
three new-object warm executions are recorded, except the expensive CPU1
parent benzene baseline, which has one warm sample. All JAX outputs are
synchronized. End-to-end means independent geometry → SCF → response →
oscillator strengths; no reference-program orbitals/integrals are substituted.
Warm means fixed geometry with the existing compilation/integral cache policy,
not a new-geometry throughput claim. No converged response-vector restart is
fed into subsequent solves. Persistent compilation caching is disabled.

`response_solve` includes prepared data, tracing/compilation, numerical solve,
guards and output construction. Fused preparation and solve are not assigned
invented independent timings. Compiler and solver diagnostics run separately.
Their times are not substituted into the clean totals.

After the clean-phase memory marker, each molecular run checks
`||Az−ωz||/||z||` with its original operator. Exit status, SCF convergence and
finite excitation energies alone do not confer scientific acceptance. Gates:
SCF error ≤ `1e-7` Ha, excitation error ≤ `1e-4` eV, per-root original residual
≤ `1e-8` Ha and cluster-summed oscillator-strength error ≤ `1e-7`.
Near-degenerate states are compared by complete clusters, not column signs.
Ratios are **candidate time / parent time**.
The independent post-clean certificate applies to the final repeat in each
process. Earlier repeats retain the solver's original-operator guards, public
convergence and energies; separate post-clean residual arrays for those
earlier repeats were not saved. This is a correctness gate per fixed-input
configuration, not a claim of a separately replayed certificate on every sample.

Sampled process-group RSS (50 ms), Linux `ru_maxrss`, XLA temporary estimates
and GPU allocator in-use/pool metrics are separate. The reported clean RSS
includes initial compilation and all clean repeats; it is not an isolated
warm-only peak. Sampling can miss short peaks. Neither `max_memory` nor an
XLA estimate is a hard process-memory limit. Missing counters remain null.

## A. Stable response execution

[execution.py](../../../src/gradscf/tddft/execution.py) provides a bounded
32-entry numerical-program cache. Molecule arrays and external XC parameters
are dynamic arguments. Immutable structural controls, abstract shapes/dtypes
and the actual execution device select the program. This does not cache
converged states or numeric physics. `scf_cycles` is a dynamic diagnostic leaf
so changes in SCF iteration count do not select another response executable.

Opaque functionals, identity-hashed producer metadata, unknown device contexts
under outer tracing, occupation inference and shell-direct providers retain
their existing execution path. Unknown/global neural callbacks are not
rewritten into point-local kernels. A trace-local finite-check collector
preserves eager NaN/Inf rejection across the new outer JIT. There is no new
physical `stop_gradient` boundary.

The parent water TDA warm trace compiled **four different executables**:
grid action at 12 columns, the Davidson iteration program, final Ritz checking
at six columns, and spectral-observable reconstruction at five columns.
Their shapes and purposes differ. Cache-miss logs also identify freshly
redefined inner functions. It was incorrect to describe these simply as four
identical compilations. The final TDA and full-TDDFT diagnostics each have two
first-call compilations and **zero backend compilations on the new-object
warm call**. Array/reference updates and explicit/eager NN gradients have
separate reuse regressions; shape/dtype/root-count changes legitimately compile.
See [compile_counts.json](compile_counts.json) and [compile_events.jsonl](compile_events.jsonl).

## B. Real workload and grid-field reuse

The first dense-AO-only cache prototype did not apply to benzene: its AO
container is streamed (`ao=(0,120)`). That prototype is marked superseded,
not presented as a successful benzene optimization. The retained implementation
projects streamed AO blocks once to occupied/virtual MO value/derivative fields,
then reuses those fields within each prepared numerical response. It stores
neither full AO grids nor `[feature,grid,occ,vir]` transition tensors.

[response_options.py](../../../src/gradscf/tddft/response_options.py) charges
retained fields and scan padding within the existing action budget; insufficient
budget keeps the original block recomputation. Packed gather workspace has a
separate reservation before assigning MO cache, batch and grid blocks.
The AO provider's rematerialization and all parameter/geometry dependencies
remain differentiable. Fields are recomputed from new dynamic inputs, not
persisted across geometry/NN updates. Known restricted semilocal kernels use
this optimization; unrestricted/global/opaque HVP callbacks keep their scope.
Retained MO fields are an explicit time/memory tradeoff. In the intermediate
CPU1 benzene run, clean sampled RSS rose from 3,004,534,784 to 3,239,321,600
bytes despite the runtime improvement. The separate final CPU8 comparison
below has a lower peak; this does not establish a universal RSS reduction.

The CPU1 solver trace has 66 Ritz iterations, 65 six-column appends, fourteen
12-column restarts, one 12-column initialization and one six-column final
Ritz check. These are **81 Davidson/Ritz operator calls**, not 81 iterations
or a complete workflow-wide action count.
Of 576 submitted columns, 538 are effective. Padding is about 6.6%; removing
it cannot explain or reproduce the observed multi-fold gain. A planner batch
of 32 does **not** mean each small action is padded to 32: widths 6/12 are
passed directly. No hard locking is implemented, and no guard was removed.

Spectral-observable reconstruction occurs outside Davidson's own counting.
A second complete CPU8 diagnostic explicitly includes its five-column action;
see [solver_events.jsonl](solver_events.jsonl). Callback intervals do not
reliably bracket XLA execution; they are not summed as kernel timings. The
clean shared-input probe at actual widths 6/12 provides the independent timing
evidence in [action_comparisons.json](action_comparisons.json). It uses one
saved reference on both sides and is not an end-to-end result. Preparation
and compilation are separately recorded.

## C. Shared packed J/K

TD private gather code now calls the common packed contraction. It preserves
arbitrary nonsymmetric batched/complex densities, s4/s8 pair multiplicities,
J-only/K-only selection, ERI/density JVP/VJP and existing geometry products.
No density symmetrization is used to hide transition-density differences.
Reverse scans checkpoint gather tiles instead of retaining all row indices.
The native s8 ERI transpose now receives the same selection and workspace
budget as the forward public call; it no longer silently uses dual J/K and
the default 64 MB in the portable transpose.

CPU uses native s8 or bounded scalar rows. GPU uses bounded row tiles, not the
CPU one-row schedule. The default extra workspace is 64 decimal MB, with the
existing reserve. Explicit 4/8/16-row probes use 256 MB. These are exact layout
strategies, not additional physical approximations.

CPU90, batch five: TD-private warm action **0.474614 s → 0.160413 s**; XLA
temporary estimate **525,043,800 → 5,999,648 bytes**. This is a microkernel
improvement, not a claim of 2.96× whole-TDDFT acceleration.
Measured clean process-group RSS in that isolated probe is separately
1,366,278,144 → 849,416,192 bytes, including input preparation and compilation.

GPU60, batch five: private **0.8257 ms**, old public one-row **2.9102 ms**,
new bounded default **1.0260 ms**. The default trades about 24% action time
relative to the fast private implementation for XLA scratch reduction from
107,874,576 to 16,308,512 bytes. Tile16/256 MB gives **0.8658 ms** and
31,882,016 bytes. The bounded common implementation is retained for memory,
contract consistency and maintenance, without claiming universal GPU speedup.
All saved-array gates and raw samples are in
[packed_comparisons.csv](packed_comparisons.csv).
The GPU allocator's measured in-use high-water mark for the private/default
probes is 205,809,408 → 138,558,720 bytes; peak pool is
335,544,320 → 268,435,456 bytes. Both have 33,986,816 bytes actively allocated
at the final sample. These include inputs/preparation/compilation, and are
distinct from the XLA scratch figures. `bytes_reserved=0` in this backend's
statistics does not mean the reported pool is empty.

## D. CPU RI and GPU response

Explicit auxiliary RI preparation, necessary SCF/reference completion and
committed input copies are performed in a CPU context. The prepared reference
is then explicitly transferred to the requested response device. Repeated
access reuses the resident reference; geometry changes reject stale access
until reconvergence. No CPU FFI is registered as a fake CUDA handler, no
NumPy conversion is inserted into a traced GPU numerical kernel, and no
integral approximation/backend is silently substituted.

CH₂O/PBE0 RI → GPU TDA completes with a warm response around **0.163 s** and
warm end-to-end around **1.325 s**, with eight logical CPU threads for CPU
preparation. Its CPU1 result is not used as a matched GPU speed ratio.
The old failed CUDA-handler entry cannot supply an acceleration denominator.
CPU/staged-GPU CH₂O excitation energies agree to 7.991e-13 eV;
see [gpu_cpu_numerical_comparison.json](gpu_cpu_numerical_comparison.json).
Tests count one 2c/3c preparation, one reference completion and one logical
reference transfer per converged reference. These counts are not PCIe bus
traffic measurements. Native basis coefficient/log-exponent JVP/VJP survive
CPU→GPU array transfer. Full traced RI geometry remains explicitly unsupported.

## Final-source performance and numeric acceptance

All headline pairs below passed the numerical gates. Seconds are synchronized; warm values are medians of three independent new-object repeats.

| Case | Logical CPUs | First E2E before → after (s) | Warm E2E before → after (s) | Time ratio | Warm response before → after (s) |
|---|---:|---:|---:|---:|---:|
| benzene-pbe0-tda | 8 | 168.616 → 80.043 | 145.625 → 64.804 | 0.445 | 129.371 → 48.503 |
| ch2o-pbe0-tda | 1 | 46.570 → 37.043 | 12.743 → 6.177 | 0.485 | 11.172 → 4.567 |
| oh-pbe-uks-tda | 1 | 115.819 → 76.720 | 70.636 → 14.841 | 0.210 | 65.860 → 9.924 |
| water-pbe-full | 1 | 34.843 → 32.319 | 7.821 → 1.448 | 0.185 | 7.299 → 0.871 |
| water-pbe-tda | 1 | 41.385 → 32.804 | 9.308 → 2.369 | 0.254 | 8.806 → 1.846 |

| Case | Parent clean sampled RSS peak (bytes) | Final clean sampled RSS peak (bytes) | Maximum excitation error (eV) |
|---|---:|---:|---:|
| benzene-pbe0-tda | 3,414,294,528 | 3,043,344,384 | 1.133e-14 |
| ch2o-pbe0-tda | 2,296,680,448 | 2,192,588,800 | 1.662e-14 |
| oh-pbe-uks-tda | 4,148,105,216 | 2,720,608,256 | 1.208e-14 |
| water-pbe-full | 1,893,027,840 | 1,831,796,736 | 6.042e-15 |
| water-pbe-tda | 2,112,036,864 | 1,864,736,768 | 1.511e-14 |

RSS is a process-group measurement spanning cold compilation and clean repeats, not compiler scratch or a warm-only metric.


The headline CPU1 small cases use `final-verified`, remeasured after review
fixes. Benzene CPU8 uses `final-cpu8`, also after those fixes. Earlier `final`
CPU1 measurements and stage-A/B prototypes remain as intermediate evidence;
they are not relabeled as measurements of a later source snapshot. Independent
patch gains are not multiplied together.

Benzene final states also pass the **independently imported parent operator**:
maximum residual `3.39e-9` Ha. Against the mathematically equivalent stage-A
state file, maximum excitation difference is `1.21e-14` eV and maximum
AO-metric cluster-projector distance `2.98e-8` (threshold `1e-4`, clustering
gap `1e-5` Ha). The near-degenerate fourth/fifth states are compared jointly.
The parent clean CPU1 run did not export state vectors; this is explicitly a
parent-operator certificate plus a stage-A subspace comparison, not a claimed
comparison to unavailable parent vectors.

## Derivatives, failures and retained limits

Final narrow CPU regression: **55 passed**; final GPU packed/RI regression:
**13 passed**; solver/status regressions: **8 passed**. Additional complete
physical regressions pass PBE full-TDDFT geometry gradients, global neural XC
RKS/UKS full-TDDFT geometry gradients, two-geometry NNAO coefficient and
log-exponent directions, and a small MACE energy optimizer/checkpoint update.
The physical tests reconverge SCF on both FD sides. Geometry tolerances are
specified in their fixtures (Ha/bohr); NNAO directions are dimensionless
coefficient/log-exponent directions with two FD step sizes. See
[test_results.json](test_results.json) for exact commands, suites and skips.

The MACE example is a small randomly initialized two-H₂-geometry fixture, not
a trained production checkpoint. It does not certify force-supervised mixed
NNAO derivatives or end-to-end GPU neural-XC RI training. Arbitrary complex
density contraction support does not assert general complex-orbital TDDFT.
Full RSH performance and RI geometry expansion were not added to this matrix.

Failures are preserved: an initial multi-case geometry suite reached its
600 s aggregate budget (remaining cases were split and passed); a projector
script initially double-counted equal-spin occupations (fixed and re-run);
new JAX API/finite-check/metadata boundary regressions were observed RED then
fixed. Historical raw records were not edited. The old benzene statement is
corrected separately in [benzene_erratum.md](benzene_erratum.md): its first
318.490191 s response did return, and the 600 s timeout covered the whole
repeated process. Historical missing residuals remain unknown.

## Next priorities, bounded by this evidence

1. **Remaining grid/response contractions and real 6/12-column execution.**
   Final benzene still spends most of its warm total in response. Saved action
   probes identify the relevant batch sizes; 6.6% padding gives only a small
   possible upper bound for padding-only work. Investigate exact contraction
   order/threading at those widths before adding solver algorithms. Risk:
   transpose memory, opaque/global XC and higher response. Acceptance: same
   original operator, independent X/Y, array/geometry AD and the same workload.
2. **GPU tile throughput within an explicitly shared budget.** The measured
   60-AO default/tile16 difference is a local opportunity, not an end-to-end
   promise. A second relevant size is needed before changing a general policy.
   No new approximation is needed; guard peak memory and ERI transposes as
   well as value throughput. The existing explicit tile override is retained.
3. **Cold-start compilation and reference preparation.** Same-signature warm
   response recompilation is fixed; first calls still carry compilation/lazy
   setup and SCF. Persistent cache experiments should isolate this from data
   preparation and test parameter invalidation. Do not treat first-minus-warm
   as pure compilation cost or promise that entire difference as recoverable.

No new ISDF/THC, NTO truncation, replacement Davidson, mixed precision or
multi-GPU machinery was retained. DF-demand/share rewrites were not repeated:
the previous outer-JIT DCE/CSE evidence did not justify another implementation.
No full PySCF/MOLGW matrix was rerun and no cross-hardware historical ratio is
reported as a current result. This round's authoritative speed comparisons
are controlled parent/candidate pairs on n002.

## Files and rerunning

Raw logs, XML, source archives, RSS traces, HLO and saved states are preserved
locally under `raw/evidence/` and remotely under
`~/private/homefile/gradscf-td-execution-20261027/evidence/`. Large raw outputs
are excluded from Git; compact records, hashes, configurations and patches are
tracked. [commands.txt](commands.txt) and the
[harness README](../../../benchmarks/probes/tddft_execution/README.md) provide
smoke and main reproduction commands. No quantum-chemistry tests ran locally.
