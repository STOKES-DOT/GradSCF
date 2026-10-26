# libcint-only production interface and performance implementation

Baseline snapshot commit: `7d2e859`, restored from current HEAD `70aea47` plus
81 current dirty source/config/test files. Historical performance is context;
all speed claims require a fresh baseline in this round's isolated environment.

Public interface name: **libcint**. No cpu/native backend aliases. CPU/GPU
execution settings remain distinct. High-level examples use ordinary RHF/RKS
constructors, without an explicitly selected integral engine.

## Tasks and acceptance

- [x] S0: preserve main index, staged/unstaged binary patches and relevant
  untracked content; create managed worktree and separate source snapshot commit.
- [x] S1: enumerate operators, layouts, dynamic parameters and actual NNAO
  consumers; register numerical, AD and resource gates in migration_matrix.md.
- [x] S2: repair public TDDFT/tools imports; first verify a failing regression,
  then converge independent RHF/PBE and TDA/full-TDDFT reference cases.
- [x] S3: route public Gaussian/Coulomb producers and consumers to libcint;
  preserve packed/direct/RI/selected products and transparent special operators.
- [x] S4: preserve actual NNAO contraction/exponent dependencies;
  verify parameter FD, JVP/VJP transpose, an actual optimizer step and required
  mixed derivatives. Record real-model versus fixture coverage separately.
- [x] S5: remove the redundant JAX Gaussian engine and Python runtime integral
  bridges after consumer migration; reject old selectors, retain legal notices.
- [x] S6: fix stable SCF and CC dynamic compilation boundaries; preserve outputs,
  DIIS/residual/stopping and implicit-response equations; measure compile counts.
- [x] S7: independently validate bounded geometry-DFT, GW and BSE/TDDFT changes;
  preserve all physical windows, QP candidates/selection/certification and AD.
- [x] S8: clean isolated build without PySCF production dependency, narrow
  regressions and matched cold/three-warm comparisons; local focused commits,
  final removal audit and proof main source/index were not changed by this task.
- [x] S9: audit and optimize derivative symmetry across all actual consumers;
  completion is part of this same project and the final S8 acceptance.
  - [x] S9a: record operator/product/method ownership and new-node before
    timings, actual libcint evaluation/query counts, RSS and device boundaries.
  - [x] S9b: implement shared independent-quartet scheduling and covariant
    JVP/HVP emissions; retain selected-window/slab/packed semantics.
  - [x] S9c: aggregate all symmetry-related VJP cotangents with correct repeated
    shell/AO multiplicities; share scheduling across existing derivative families.
  - [x] S9d: pass one combined value/AD/FD/thread/device/consumer acceptance
    matrix and matched before/after measurements; remove transitional duplication.

Verified S9 checkpoint (2026-10-26, allocation109318): the independent
four-center basis schedule passed21 selected tests; the combined basis,
coordinate and bounded scratch-cache group subsequently passed41 in14.09s.
Existing higher-coordinate, shell-window, packed, slab, direct-J/K and fixed
range regressions passed128 in39.65s. These groups overlap and are not added
into an all-method-suite claim. The all-consumer before/after scan, clean
same-node timing, GPU placement and final installed build were subsequent gates, now closed by the final sign-off below.
Scratch caching stores at most4096 kernel/actual-shadow-shape size entries
per driver; physical integral values and converged states are never cached.
The obsolete ordered-quartet traversal has no remaining consumers and is
removed; both library modes rebuilt and the41-product group passed again.
Thirteen paired fixed-input counter cases have bit-identical numerical inputs,
three consistent counter repeats and maximum before/after array differences
below7.2e-13. These are native products, not end-to-end timing claims.
The broad after-validation scan uses8 logical CPUs to reduce repeated JAX
compilation time; its before scan used1. These instrumented validation scopes
do not produce method speed ratios. Clean product CPU1/CPU8 pairs and public
TDDFT CPU1 pairs retain identical thread settings on both sides.

All12 before consumer cases completed successfully. A genuinely fresh process
then exposed a bounded-geometry handler-registration dependency masked by
earlier value/diagnostic calls. Twelve isolated JVP/VJP/HVP regressions failed
before the fix; all12 plus18 existing packed tests passed afterward (30 total,
32.51s). Commit37eb17e initializes the geometry family itself. The first clean
attempt is preserved under `clean-attempt1-20261027`; commit31bb012 reports
library loading separately in both microkernel harnesses. Complete workflows
retain their original geometry-to-result timing scope. The current post-fix
clean and consumer/device/install round proceeded to the final sign-off below.

Final sign-off (2026-10-27): all registered bounded S0--S9 gates are complete.
Both12-case consumer scans passed;44 final regression tests, the N2 CPU/5090
MP2 geometry/HVP gate, four complete MACE fixture trajectories, and12 installed
fresh-process derivative cases passed. The final16 micro pairs and12 formal
processes passed their input/numerical/provenance gates. See `s9_results.json`
for exact scopes, raw arrays, timing eligibility and preserved regressions.
All486 checked installed code/resource files match this worktree. The original
HEAD/index,81 selected overlay files and all1901 starting source entries remain
byte-identical. Allocation109318 ended normally with exit0:0 after03:46:59.
Code and tests have focused local commits; no merge or push was performed.
The separate TDA profile records a remaining compilation cost, not a new
acceptance requirement or a claimed completed TD operator redesign.

No merge/push. Native C vendor sources remain unchanged. Missing rules must
raise explicitly; no automatic numerical fallback, zero tangent or frozen
physical parameter is acceptable. Keep JAX grids/AO/XC/normalization/projection.

## Resource policy

Reuse no historical job ID. One newly allocated GPU and at most 64 logical CPU,
256 GiB host memory; heavy runs serial. A800 substitution was authorized when
V100 is unavailable. The initial validation limit was600 s/case and7200 s/round.
On2026-10-26 the user explicitly increased the case limit to6000 s and required
completion of the original full GW/BSE calculation. These new long runs use
separate output directories, keep the original mathematical task, and have a
6300 s controller budget per6000 s case. The same resources are reserved for
6 h; earlier600 s timeout records remain intact. Resource renewal/extension
does not certify an unfinished calculation. Release the allocation after use.

On2026-10-26 the user added S9 and explicitly requested one V100 or5090,
64 logical CPU,512 GiB and48 hours. The current V100 nodes cannot satisfy
512 GiB total memory; n006/NV5090 satisfies the request. Read lwcluster0.8.0
manual again and check live resources/budget before submission. The replacement
allocation depends on the current project allocation ending; do not overlap
project GPUs or exceed64 logical CPU. Preserve n005 results as phase1 evidence;
measure the symmetry before/after on the same new node rather than using
different-node timings as an optimization ratio.

## S9 scope, implementation boundaries and risks

Keep this work in the existing isolated branch/worktree. The committed
pre-symmetry implementation and its independent comparison snapshot remain
available for numerical and performance regression. No production legacy/JAX
integral engine is reintroduced. Profiling instrumentation and reference
snapshots are explicitly separate from the production build.

Affected implementation owners are `_native/csrc/eri_basis.{h,cc}`,
`eri_window.h`, `coulomb_shells.h`, `eri_geometry_entries.h`,
`geometry_contraction.h`, `geometry*.cc`, `coefficients.cc`, `exponents.cc`,
and the necessary private FFI/build bindings. Extend an existing shared helper
when possible. Method modules consume the same products and must not acquire
their own integral scheduler or derivative implementation.

Use only the symmetry group of each actual operator/product. Four-center
real Coulomb values have eightfold symmetry, but a fixed derivative-leg
pattern has a smaller stabilizer. Move derivative positions and Cartesian
components with each index permutation. Do not promise eightfold wall-time
gain for every derivative family. Two-/three-center RI and one-electron
operators have different roles and pair symmetries; do not exchange an
auxiliary shell with an orbital shell.

Selected rectangles may admit only part of an orbit. Enumerate canonical
quartets whose orbit intersects the requested window, without retaining an
O(nshell^4) connection list or computing all global quartets for a small window.
For repeated shell tuples, full native shell blocks already contain multiple
AO entries: deduplicate shell emissions and account for derivative-position
multiplicity separately. VJPs must pool every admitted cotangent copy after
inverse permutation, including nonsymmetric cotangents/densities.

Parallel workers own mutable tables and shell scratch. For ERI tensor products,
canonical ownership must prevent two workers writing the same JVP output block;
never replicate the full output tensor per worker. Direct J/K actions have
cross-quartet output collisions and retain a separate valid row-ownership or
private-AO-buffer reduction policy. VJPs reduce small parameter/coordinate
vectors in a fixed order. Preserve zero-coefficient derivatives, JVP/VJP
transpose consistency, all currently supported second/higher products,
and explicit errors for unsupported combinations. Do not add new physics,
arbitrary derivative orders, screening approximations or looser tolerances.

The consumer survey includes HF/DFT (restricted/unrestricted and existing GKS
paths), TDA/TDDFT/CIS, CI/FCI, MP2/RI-MP2 and recursive MP, CC/CCSD(T)/EOM,
GW/BSE/OFDFT and NNAO. Use representative existing public/test fixtures for
each distinct integral/AD path, not a Cartesian product of all methods and
functionals. Mark FFT/GTH and unsupported existing derivatives explicitly.

Validation must cover distinct and repeated shells, general contractions,
individual zero coefficients, arbitrary cotangents, nonidentical directions,
Cartesian/spherical order, ordinary/fixed-range operators, s1/s4/s8,
non-symmetric and overlapping windows, and padded slabs. Keep independent
baseline/PySCF/finite-difference oracles. Register tolerances before inspecting
new results; report actual call counts and isolated-process RSS separately
from clean cold/three-warm timing. Retain final-output tasks and convergence
standards. Completed phase1 gates are repeated where S9 changes their path.

If a new scheduler fails an invariant, keep the numerical failure and stop
scale-up; repair the smallest shared product before continuing. Recovery uses
the committed pre-symmetry snapshot in an isolated comparison directory,
never reset/stash/clean of the original worktree. Final acceptance combines
S0--S9, the removal/migration audits and original-worktree preservation proof.

## Progress discipline

Review this plan after each task. Never label implemented, queued, tested,
numerically passed, performance-accepted and committed as the same state.
Record limitations and complete unaffected tasks when a specific path blocks.
