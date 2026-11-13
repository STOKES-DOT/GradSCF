# Shared ground-state and response execution

Date: 2026-11-10. Baseline: `c5c8b28e8838132c30b6074c0690f029a87edabe`.
Implementation branch: `perf/shared-scf-response-20261110`, isolated worktree
`/Users/jiaoyuan/.codex/worktrees/bd0a/GradSCF`.

This report concerns a small shared-execution change, not a rewrite of the
six ground-state solvers. RHF/UHF/GHF/RKS/UKS/GKS retain their existing common
solver, integral-provider and grid architecture. Production changes are only
four Python files: 79 insertions and 30 deletions. There is no new integral
backend, approximation, numerical-field cache, `stop_gradient`, root-window
restriction, density symmetrization or full response matrix.

## Implemented behavior

1. `tools/features.py`: the built-in streamed grid providers now use one
   stable JIT entry for physical density/gradient/tau/Laplacian projections.
   Provider arrays and density are dynamic inputs. All spin/feature consumers
   use the existing common helpers; opaque custom map providers keep their
   original execution contract. Only the requested small physical fields are
   returned, rather than AO-grid tensors.
2. `tddft/response.py`: share the local XC contribution between both full
   TDDFT rows. A declared linear kernel evaluates `XC(X+Y)` once. An opaque or
   clipped callback retains separate `XC(X)` and `XC(Y)` evaluations. A/B
   exchange and nonlocal contributions retain their original index structure.
   Exact standard semilocal adapters also avoid a density projection whose
   result was unused in their constant hybrid fraction.
3. `tddft/casida.py` and `response.py`: pass the actual builder Fock diagonal
   to the restricted full-response preconditioner. Legacy mocked callables
   without this metadata retain the old fallback. This fixes a noncanonical
   reference inconsistency; it is not a claim about canonical excitation shifts.
4. `tddft/execution.py`: allow the existing bounded program cache to serve an
   exact `NativeDirectBasis` whose physical parameters are dynamic PyTree
   leaves, only when this preserves the eager integral-cache policy. Explicit
   no-cache or screened direct policies qualify. Automatic zero-cutoff policy
   qualifies only when twice the exact ERI-cache storage exceeds the entire
   software budget, proving eager admission impossible. Small/ambiguous exact
   caches and unknown/mutable providers retain the old route.

The program cache and numerical preparation remain separate. Every fresh
workflow still recomputes SCF and response inputs; cached executables receive
new coordinates, basis parameters, density, orbitals and model parameters.
Changing block topology, shape, dtype or method structure can legitimately
compile a different program. No general cache of traced XC Hessians was added.

Commits:

- `87ab1bb9`: restricted TDDFT XC reuse and builder gaps.
- `37ba5352`: physical grid field program reuse.
- `7dd87dfa`: shell-direct response execution reuse and conservative policy.
- `463212dd`: isolate the streamed-grid tracing observer; preserve width and
  dense numerical parity checks.
- `f24cfbe2`: explicitly verify relaxed coordinate AD through the shared direct
  response program.
- `11ad34e0`: bounded reproducible probes and real six-facade/GW/BSE integration.

## Environment and protocol

All scientific execution used the allocated `jy_gpu_2` Slurm job **119123**,
node n006, one RTX 5090, 16 logical CPUs and 128 GiB. This is inside the
approved one-GPU/64-logical-CPU/256-GiB envelope. Heavy experiments were serial.
The main GPU workflows used eight logical CPUs `[8,9,10,11,200,201,202,203]`,
BLAS/OpenMP one thread, native integrals eight threads. CPU downstream probes
used one logical CPU/native thread. These are not eight physical cores.

Python 3.11.15, JAX/jaxlib 0.10.0, NumPy 2.4.4, SciPy 1.17.1, PySCF 2.13.0;
libcint 6.1.3. Native binary SHA256:
`9f8da65e0e55029dbfcf137eeefd328b5e947015ac0c40a4ae468c3810787e89`.
No native source changed; the same existing Release native build was used for
parent and candidate. `runtime.json` and the workflow manifests record actual
Python/source/library paths, hashes, device, affinity and thread settings.
Persistent JAX compilation caching was disabled for the clean paired runs.
The probe prepares the eager SCF state with a CPU default, exports the response
with the explicit `execution_device="gpu"` facade option, and executes response
under the GPU device context. The response-reference stage includes its device
placement. Libcint/direct integral work remains CPU code. This is a staged
CPU-SCF/native + GPU-response workflow, not an all-GPU integral/SCF claim;
transfer time is included in the public stages, not independently isolated.

Water and benzene: Cartesian def2-SVP, PBE0, grid level 3, all occupied and
virtual orbitals, five singlet full-TDDFT roots, float64, identical atom guess,
SCF energy/density/gradient tolerances 1e-10/1e-8/1e-7, response tolerance
1e-8 Ha, `max_memory=4096` MB, response grid block 1024. Conventional direct
uses `direct_tol=1e-12`, `integral_cache=none`. The RI pair instead uses the
same explicit def2-universal-jkfit auxiliary basis in both versions.
Representations are never mixed in a speed comparison.

Each main process has one first call and two warm calls with fresh SCF/TD
objects and no restart eigenvectors. Synchronization occurs at public stage
boundaries. Startup/import/native-load cost precedes this workflow boundary;
first call is not the entire Python-process wall time. Original-operator
residuals are independently computed after the clean phase for the final
repeat. Other repeats report their actual convergence flags and eigenvalues;
only the final repeat has a direct independent residual certificate.

`timings.csv` preserves every clean raw sample plus medians. Process-group RSS
is sampled every 0.2 s by `guard.py`; Linux cumulative `ru_maxrss` is separate.
GPU memory snapshots/pool/peak allocator counters have their original metric
names. Compiler temporary estimates are not substituted for actual memory.
SCF stage includes integral/grid preparation and iteration, not iteration alone.
Profiler/counter runs are separate from clean timings.

## Measured results

The machine-readable final comparison is generated by
`benchmarks/probes/shared_execution/summarize.py`. Initial direct pair:

| GPU workflow stage, warm median (s) | Parent | Candidate |
|---|---:|---:|
| Water full response | 3.2859 | 0.1873 |
| Water geometry-to-spectrum workflow | 3.5301 | 0.4338 |
| Benzene full response | 25.6643 | 21.5437 |
| Benzene geometry-to-spectrum workflow | 34.5298 | 34.8427 |
| Benzene adjacent direct full response | 25.8529 | 16.9713 |
| Benzene adjacent direct geometry-to-spectrum workflow | 34.8338 | 25.8657 |
| Benzene RI full response | 2.2301 | 1.2311 |
| Benzene RI geometry-to-spectrum workflow | 6.7549 | 5.6435 |

Water first-call response was 18.5358 -> 12.2263 s; first workflow was
31.8194 -> 25.6631 s. Benzene first-call response was 42.3415 -> 49.4397 s;
first workflow was 63.3104 -> 82.4121 s. Cold compilation is not uniformly
improved. The unchanged benzene SCF stage/cycle count also varied substantially
between these separated batches, so this first pair does **not** establish a
benzene end-to-end gain. The registered adjacent direct and RI reruns are
retained separately rather than replacing inconvenient measurements.

The adjacent direct rerun kept SCF warm medians at 8.8737/8.7857 s and response
reference preparation at 0.0931/0.0935 s. Response samples were
[25.4420, 26.2639] -> [16.7620, 17.1806] s; full workflow samples were
[34.3936, 35.2741] -> [25.6858, 26.0456] s. Here the unchanged SCF phase is
stable, supporting a response-stage improvement rather than a general CPU
speed change. First-call response was 42.4761 -> 35.7398 s and first workflow
63.6461 -> 56.8337 s. Excitation difference is 1.11e-16 Ha. This rerun was
registered after observing the first pair's SCF drift, before inspecting its
results; both pairs remain in the evidence and summary.

The RI pair's SCF warm median was 4.4121 -> 4.2827 s, while response
reference preparation was 0.0976 -> 0.1142 s. Its response improvement is
therefore not explained by SCF becoming much cheaper. RI final residual maxima
remain below 9.72e-9 Ha, and excitation parity is 2.22e-16 Ha. This particular
short, two-warm-sample experiment is evidence of benefit at these dimensions,
not an asymptotic or cross-hardware speed claim.

Actual measured memory is a tradeoff (decimal GB):

| Initial paired case | Group RSS peak, parent -> candidate | JAX allocator cumulative peak, parent -> candidate |
|---|---:|---:|
| Direct water | 3.072 -> 2.860 | 0.611 -> 0.496 |
| Direct benzene | 3.666 -> 3.227 | 0.810 -> 1.639 |
| RI benzene | 3.740 -> 3.544 | 1.654 -> 1.654 |

The allocator peak includes first compilation/execution and all clean repeats;
its counter is read at stage boundaries before the independent certificate.
It is neither device-wide reserved memory nor an isolated warm-solve peak.
The combined compiled direct program increases device peak for benzene in
this test. There is no claim of uniform memory reduction. Exact sampled RSS
and `ru_maxrss` remain separately available in `process_memory.csv`.

The initial direct final root residual maxima were 6.51e-9 Ha for water and
9.69e-9 Ha for benzene, both before and after. Same-code excitation parity is
checked to 1e-9 Ha. This is a numerical implementation regression, not a basis
or functional accuracy study.

An instrumented fresh-public-object benzene run recorded three compilation
loads on the first response call (`jit_solve`, `jit_apply`, `jit_multiply`),
and **zero on its second response call**. Counter overhead and JAX compile
logging mean its wall times are not clean benchmark samples. Synthetic narrow
regressions independently require zero unexpected compilations for changed
same-shape physical data and neural parameters, plus fresh-result agreement.

## Numerical and AD acceptance

The fixed plan and `test_receipts.json` link the tests to each contract.
After explicitly superseding fixture/instrumentation failures with their final
isolated reruns and deduplicating CPU/GPU identities, `acceptance.json` records
**254 passed, 11 failed, 0 skipped**. All eleven failures are independently
reproduced on the untouched parent, as recorded in `baseline_failures.json`.
These totals include capability/error-contract tests, not 254 successful
molecular calculations. This is not a claim that the repository suite is green.
The mixed compiler crash remains separate from those XML totals.
New regression tests were run RED against the parent before production edits:
17 intended failures/9 passes, plus the later policy extension's 2 intended
failures/4 passes. The new mathematical, field-reuse and direct-execution gates
then passed on CPU; the complete 36-test new group also passed on GPU.

Coverage includes non-symmetric density, independent X/Y, unequal occupied and
virtual counts, batched actions, raw factors, linear and clipped/opaque XC,
noncanonical Fock gaps, adaptive/block/tail grids, LDA/GGA/MGGA/hybrid/RSH,
array and model JVP/VJP, changed physical inputs, fixed-density/explicit/implicit
external neural training, global-XC HVPs, native UKS/GKS geometry and all four
molecular GW geometry JVP/VJP with two finite-difference step sizes.

An additional two-test gate explicitly forces `jax.default_device(cpu)` and
observes admission of exact `NativeDirectBasis` into the shared response program.
For relaxed H2/STO-3G full TDDFT, PBE and the existing four-layer global neural
functional both rebuild integrals, moving atom-centered grid, SCF state and
response at displaced geometries. At the existing 2e-4 Bohr FD step:

| Observable | PBE | Global NN |
|---|---:|---:|
| Direct/full energy difference (Ha) | 0 | 5.55e-16 |
| Direct/full derivative difference (Ha/Bohr) | 2.72e-15 | 7.77e-16 |
| Complete geometry FD error (Ha/Bohr) | 9.85e-9 | 5.34e-9 |
| JVP/VJP difference (Ha/Bohr) | 2.22e-15 | 0 |

These are actual XML properties, not the acceptance tolerances. This closes
coordinate AD for the new direct execution admission, separately from the
array, Gaussian-parameter and model-parameter gates. It is a bounded fixture,
not a new large-system Hessian or arbitrary-order AD validation.

A separate five-test completion run passed the existing neural force training
and NNAO direct-basis/all-electron parameter-gradient fixtures. The preceding
large AD run had 93 passes and two missing-example fixture errors; the exact
unchanged example was copied before the successful completion run. Those two
setup failures remain in the raw record and are not rewritten as passes.

Seven existing `test_direct_response_cache.py` assertions expect NumPy host
exponent storage but receive JAX arrays. They fail identically on the untouched
parent (10 passed/7 failed) and are explicitly excluded from new-regression
success claims. This patch does not alter that serialization policy.

Before that mixed process crashed, four additional
`test_direct_response_reuse.py` tests had failed. Isolated parent and candidate
runs both return exactly **3 passed / 4 failed**, with identical failing test
names/messages: three expect an admitted 440-byte ERI cache but the current
planner admits zero; one expects basis AD to raise, whereas current basis AD
returns. These predate the four-file production patch, are distinct from the
seven host-type assertion failures, and remain unfixed in this scoped change.
Raw XML/process records are `parent-direct-reuse.*`/`candidate-direct-reuse.*`.
This is not evidence that the new direct execution loses derivatives.

A mixed regression process exited -11 in JAX
`backend_compile_and_load`, at the RKS integration response call. Its raw
process/log receipt is retained. This is not a passing suite and its root cause
has not been established. The remaining streamed grid, real workflow integration and BSE/fusion modules
pass in separate processes (25 + 14 + 17 tests). A final separate 20-test
SCF/GW/BSE program-reuse/physical-leaf-response run also passes, with complete
XML and zero process exit codes. A successful isolated run
cannot prove why a combined compiler process crashed.
The streamed-grid tracing observer uses a test-local JIT identity, preserving
production caches; this instrumentation adjustment is not a crash diagnosis.

## GW and BSE workflow scope

Paired CPU public workflows cover RHF/UHF/GHF/RKS/UKS/GKS, and RKS-derived
G0W0, evGW, qsGW, Matsubara scGW on bounded H2/STO-3G. G0W0/evGW/qsGW each
continue to both BSE-TDA and full BSE with their actual QP/screening inputs.
A larger water/PBE0/def2-SVP/grid3 G0W0 -> both BSE variants was also run
before/after. No QP/screening input was frozen. All three saved downstream
comparison sets pass their numerical/mask gates; full detail is in
`downstream_validation.md`.

Current capability limits remain explicit:

- GHF/GKS ground states are supported; generalized spinor TD response is not.
- Matsubara scGW has no validated real-axis QP continuation for BSE. Its
  BSE rejection is a tested capability guard, not a successful BSE calculation.
- GW/BSE production kernels were unchanged. Single-warm downstream timings
  have broad CPU variation and are regression coverage, not evidence of a
  new GW kernel speedup.
- BSE array/response AD is covered separately; no claim is made that every
  public GW-to-BSE geometry/frequency continuation is available.

## Decisions and remaining boundaries

The implemented changes target demonstrated work. We did not add a global
prepared-response cache or mutate reference objects to cache traced values.
Such a cache needs explicit geometry/model invalidation and separate lifecycle
validation. We also did not retain speculative DF OV/VO reshaping solely on
source inspection: previous outer-JIT tests showed substantial CSE/DCE and no
reliable end-to-end benefit. Raw nonsymmetric factors keep their semantics.
No warm-start eigenpairs, changed Davidson strategy, tolerance relaxation or
virtual-space truncation is hidden in the timings.

Large direct benzene still spends real time in CPU libcint/direct contractions;
a GPU response executable does not turn those CPU handlers into CUDA kernels.
The conservative cache-admission check leaves some possible GPU/RSH automatic
policies on their existing eager route. This limits performance coverage while
preserving their representation and derivative behavior.

Original workspace tracked/untracked status and all pre-existing source
fingerprints are checked against the starting snapshot. All implementation
and generated evidence live in this new worktree; no merge or push is performed.

## Final delivery and resource closure

The original workspace remains at c5c8b28e with its initial unrelated dirty
status; none of the snapshotted original source/test contents changed. All
590 remote candidate Python files match this worktree, and all 590 parent
Python files match the initial baseline fingerprints. See `manifest.json`
and `raw/final-source-audit.json`; remote parent/candidate labels are source
export labels, not Git commit IDs. Production native paths/hashes agree.

Job119123 was allowed to finish its final task and the controller ended at
2026-11-10 14:38:11 +08; subsequent squeue was empty. Slurm accounting is
disabled, so there is no sacct completion record. No further tasks, automatic
merge or push were submitted. Local scientific execution was not used.

Reproduction commands: `benchmarks/probes/shared_execution/README.md` and
`benchmarks/probes/workflow_reuse/README.md`; actual submitted queue scripts
and receipts are under `raw/queue/`. Raw inputs, logs, XML, compile events,
measurements and process records remain locally available under `raw/` (Git
ignored); `raw_inventory.json` fingerprints them. Summary CSV/JSON and this
report are committed; large arrays and build artifacts are excluded.
