# HFKS-01d / 04a / 04b — direct preparation lifetime

Status: scoped correctness and metadata-budget GREEN; CPU/GPU same-input
preparation probes and both final four-call PBE anchors verified. Remaining
overall-plan representative/count evidence is identified below.
No commit made. Baseline checkout: f0ae2d9b. All numerical execution belongs to
the coordinator's jy_gpu_2 Slurm allocation.

## Observed RED and rejected representation

The coordinator observed job 116663 on the original 16-case fixture:
9 failed / 7 passed in 5.09 seconds. Intended failures covered repeated eager
normalization/ENV packing, workspace-only replacement, and mutable NumPy input
snapshot ownership. Values, mixed coordinate/density response, and basis AD
guards passed. The added angular-topology replacement case extends those
already observed lifetime failures.

A first candidate exposed both canonical raw parameters and independently
prepared ENV as PyTree leaves. Review identified a correctness problem:
a primal tree update to exponents/coefficients could retain the old ENV.
Job 116663 phase 03 confirmed two stale-value failures; the center update
passed. That representation is rejected, retained only as
preparation-rejected-independent-env.patch. An extra tangent guard did not
repair its primal-state inconsistency and has been removed.

The corrected PyTree roundtrip test verifies no array work during reconstruction,
canonical raw leaves only, and live numerical values. Actual repeated
preparation is tested at consumer boundaries, rather than requiring every
arbitrary reconstructed provider to carry an independent cached ENV.

## Current ownership and mathematical dependence

- Immutable topology retains the existing make_plan LRU. No global numerical
  or mutable-self cache is introduced.
- A newly constructed NativeDirectBasis owns a normalized zero-coordinate
  ENV derived from its parameters within the caller's differentiable graph.
  The private tuple records plan and parameter owners. Identity is used only
  for reuse within immutable object replacements, never as a JIT key.
- Mutable external leaves are copied into immutable JAX arrays at binding.
  Later caller edits require a new provider. Existing JAX/tracer leaves are
  retained without copying or stop_gradient.
- The public PyTree contains canonical raw parameters only. Reconstruction
  performs no numerical work and clears derived preparation. A raw leaf
  update therefore necessarily reaches fresh normalization/ENV packing.
- get_jk does not write freshly traced ENV back into a reconstructed input.
  The shared private _prepare_direct_basis helper returns a new bound object
  where necessary. Existing prepare_jk, the restricted JK builder, and the
  unrestricted/generalized solver closures call it before their repeated
  density actions. Projected providers reuse this same primitive preparation.
- A standalone compiled action taking a provider as a dynamic argument
  legitimately rederives ENV from raw parameters. The optimization claim is
  limited to eager immutable bindings and explicitly prepared consumer
  closures; it is not a claim that every JIT boundary avoids preparation.
- Coordinates are always read from the live parameters and passed separately
  to the existing native geometry family. Nonzero omega is applied by an
  immutable ENV update; ordinary Coulomb retains the zero-omega template.
  Geometry-dependent Schwarz and optimizer data are not cached here.
- Normalization keeps the original exponent/coefficient dependency graph.
  All original native AD guards remain, including higher-order rejection of
  unsupported direct basis derivatives. No new differentiation capability
  or unsupported primal-update boundary is introduced.

Preparation storage is O(ENV), not O(nao^4). The private preparation is absent
from public PyTree leaves and is re-created at consumer entry.
Existing native workspace estimates still reserve ENV/table storage; no
workspace estimate or numerical operation has been removed by this patch.

## Tests and measurement

test_direct_preparation_lifetime.py has 17 cases covering density changes,
center/exponent/coefficient changes, same-signature new objects, angular
topology, Coulomb/LR alternation, projected model changes, raw PyTree
reconstruction, quota-only replacement, mutable NumPy snapshots, mixed
geometry/density HVP with 1e-4 and 3e-5 finite differences, and the existing
first/higher-order basis AD guards.

test_direct_preparation_state.py adds seven cases: exponent/coefficient/center
primal tree updates through a reused JIT action; real RKS-builder and response
closures; and unrestricted/generalized SCF compared with independent dense
ERI runs while counting preparation outside their density loops. Dense J/K
value tolerances remain atol/rtol 1e-10; finite differences use 1e-7; molecular
energy uses absolute 1e-7 Ha.

benchmarks/probes/scf_100/preparation_lifetime.py records binding, eager action,
lowering, compilation, warm samples, HLO, XLA estimates, RSS, allocator state,
and source/native/input hashes. It explicitly synchronizes privately owned ENV.
The dynamic raw-parameter and canonical-provider paths are separate from the
actual fixed-state prepare_jk response closure. Every path uses identical
density/operator values and verifies output equality. Optional config and
density-state arguments select the exact 96-atom geometry and captured density.

These are fixed-density/action measurements, not full SCF. Earlier baseline
normalization/packing warm medians were only approximately 0.95/0.53 ms CPU and
4.49/2.87 ms GPU. Neither Python call counts nor those isolated measurements
establish an optimized SCF-loop runtime benefit.

## Observed GREEN

Direct inspection of [preparation-green.xml](final_raw/preparation-green.xml)
confirms job 116663 phase 09: **97 passed in 105.515 seconds, zero skips**, covering both lifecycle/state fixtures plus narrow direct geometry,
response, and CC regressions. This verifies the corrected canonical-raw-state
representation and the selected consumer/AD paths; it is not a performance
measurement or completion of the overall fixed plan.

A later callsite audit found that assembly._build_common constructs a numerical
provider solely for native_jk_workspace_bytes. That helper reads only topology
and the workspace quota, so preparation here is unnecessary and also breaks
existing metadata-only budget stubs. After coordinator-observed phase 14 RED (3 failed / 4 deselected in 1.52 s),
the single-line follow-up now uses the existing NativeDirectBasis(plan, None)
placeholder in this estimate only;
_budgeted_direct_basis continues to bind real parameters for execution.
All seven test_scf_integral_budget.py cases subsequently passed in
[policy-green.xml](final_raw/policy-green.xml) and again in
[final-cpu.xml](final_raw/final-cpu.xml). Final CPU totals are 94 passed and
final CUDA totals 28 passed, with zero skips. The CPU set repeats all 24
lifecycle/state cases. These counts overlap with prior runs.

## Independent code review

A separate read-only review of the five production files, two fixtures,
benchmark, and this report found zero Critical issues and no Important issue
beyond the identified metadata-only assembly constructor. Canonical state,
owner invalidation, NumPy snapshotting, trace-local binding, live coordinates,
and preserved AD boundaries were checked. The review also clarified that
benchmark RSS is a cumulative process peak (including earlier variants and
compilation), while GPU allocator data are process snapshots. The manifest now
states that neither is an isolated per-variant memory saving.

## Measured preparation scope

[CPU](final_raw/preparation-small-cpu/measurements.json) and
[RTX 5090](final_raw/preparation-small-gpu/measurements.json) probes ran under
job 116774 on n006, float64, H2/3-21G Cartesian, four AOs, two native workers,
zero cutoff, one initial plus three warm calls per variant. The random density
seed is 945. Native J remains CPU libcint, including the GPU callback.

| Quantity | CPU | RTX 5090 |
|---|---:|---:|
| Initial provider binding, s | 0.685593 | 1.172193 |
| Eager raw-parameter warm median, ms | 6.439714 | 13.863822 |
| Eager bound-provider warm median, ms | 0.704380 | 1.584865 |
| Dynamic raw compiled warm median, ms | 0.063336 | 0.465913 |
| Dynamic provider compiled warm median, ms | 0.055032 | 0.470828 |
| Fixed-state response closure compiled warm median, ms | 0.053188 | 0.348357 |
| Raw / fixed-state closure compile time, ms | 69.589 / 5.083 | 212.946 / 18.880 |

Eager binding and fixed-state closures avoid repeated preparation. The dynamic
canonical-provider compiled path still rederives ENV; its warm behavior is
similar to raw parameters, not a claimed compiled-loop speedup. Initial binding
cost is reported separately and must be amortized by real reuse. Maximum
cross-variant J difference is 4.441e-16 on CPU and zero on GPU, against
atol/rtol 1e-10. RSS is cumulative process peak, not a per-variant memory saving.

## Final source and large-anchor check

The synchronized final CPU and GPU runtime/summary/measurement/process records
were inspected directly. Both use job 116774, beta-carotene (96 atoms),
Cartesian def2-SVP, 880 AOs, 1,113,968 grid3 points, PBE/direct, atom guess,
64 logical CPUs, max_memory=180000 MB, float64, cutoff=1e-12 and the frozen
energy/density/orbital tolerances. Both processes exited zero, and every one
of their four fresh-input calls is converged, finite and scientific_pass=true.

| Final device | Cold/initial total, s | Three-warm total median, s | Three-warm solver median, s | Cycles | Max energy error, Ha |
|---|---:|---:|---:|---:|---:|
| CPU | 541.870387 | 498.655296 | 493.644826 | 26 | 3.820e-11 |
| RTX 5090 | 287.644215 | 224.100243 | 218.992625 | 22 | 1.301e-10 |

CPU/GPU orbital residuals are 6.084e-9 / 8.928e-8 Ha. Sampled process RSS peaks
are 35,542,392,832 / 3,827,851,264 bytes, below the 200-GiB hard guard.
The records are [CPU summary](final_raw/final-anchor-cpu/summary.json),
[CPU process](final_raw/final-anchor-cpu.process.json),
[GPU summary](final_raw/final-anchor-gpu/summary.json) and
[GPU process](final_raw/final-anchor-gpu.process.json).
Total time includes preparation, initial transfer and SCF; compilation is
included in solver time. These combined-candidate measurements do not isolate
preparation's contribution or establish an equal-affinity before/after gain.

Both runtimes record effective input SHA256
`8672285e72c2be606303efe5fe453d06b15b426c867752b891b461606427027d`
and native binary SHA256
`8a1f50dcbecec3831446bd60ae9df65796fa359e532c197e4a7811b4272b8b68`.
Their seven Python source hashes match the current checkout. The runtime
source map does not enumerate C++ headers: the header below was checked
against FINAL_STATUS.json; the production build receipt is
[build-final-production.log](final_raw/build-final-production.log).
All eight current production hashes match that status registry:

| Source | SHA256 |
|---|---|
| src/gradscf/integrals/_native/csrc/coulomb_shells.h | `fcacc71137adab14f1f943413786dd55637cadfa9ed59641780ab18a40401b20` |
| src/gradscf/integrals/backends/native/jk.py | `7a363a8c42db1b5f6e423fc900776fbaf83f57d1b42d8762e71e56f77c8e8273` |
| src/gradscf/integrals/plan.py | `dcc755b7c1b24f4cd8a111c21a8fef5de9cc42b7f5b692a1df0603d15d68cdcb` |
| src/gradscf/scf/rks.py | `5c24f04363cbc9d9305d2748c70ca0bad012ec1897452a1855e1967fa0df397f` |
| src/gradscf/scf/uks.py | `b9f4b92c75c3711c8601904fbafacf204cd6cb6f73b3b18ddd1b4d1eecf790f3` |
| src/gradscf/scf/gks.py | `1ef200e594f91a798bce9ea2b4fa4653a157ec700b30db3bf80e1f1422d825ed` |
| src/gradscf/scf/inputs/grid.py | `4a1c0ab420c591ff0902962bd4a194443747e39a53ed206f87cb39847aaa9d29` |
| src/gradscf/scf/inputs/assembly.py | `03e5fde2eb3ce612e20f28ca25019afd28b8b24f48da0e85da2c6dd2a21e04a3` |

The complete retained production diff was reviewed again, including shell-pair
canonicalization and AO tuple exchange, raw-only PyTree reconstruction,
object-local preparation, metadata-only assembly, host/device reservations and
the three SCF consumer entry points. No new correctness blocker was found.
The exact-value [quartet-strict.xml](final_raw/quartet-strict.xml) also reports
40 passed, zero skips. No source or plan was changed in this final audit.

## Remaining boundaries

- The representative campaign is being rerun after a harness-only
  guard-argument error; completion is not inferred from the two PBE anchors.
- Final intor/action counts require the separate diagnostic sample. Clean
  elapsed time remains separate from instrumentation; cycles are not counts.
- Input assembly still uses a conservative combined fixed_workspace estimate
  before runtime GPU cache admission; direct GPU-context assembly with host
  work exceeding VRAM is not established here.
- No automatic full-to-direct fallback is generated by molecular assembly.
  A manually mixed low-level GKS call supplying both valid ERIs and a direct
  provider is a separate preparation-performance boundary, not this assembly
  contract.
- Existing unsupported basis/third-coordinate derivatives and physics
  rejection boundaries remain. Read final_coverage.md for category coverage.
  The four converged MGGA cache/stream tests pass; no universal MGGA whole-SCF
  speedup is claimed.

## Evidence fingerprints

| Artifact | SHA256 |
|---|---|
| final_raw/preparation-green.xml | `1a6d753ed541473480d176d49fb72c8aff04eb18cb38056c9ec0677631b08659` |
| final_raw/policy-green.xml | `86fcea465a4181d1c394c06018f96f2a5d53fbd5502a85077ddcfac9c249555f` |
| final_raw/final-cpu.xml | `dfff341e6364932ae60d981b3d009e23382ffb76edddb5bcda844f7bffb82c80` |
| final_raw/final-gpu.xml | `98572d0454dfcdd33e42496f747f03d2fb3a2b9674127d7af0b282999a3c6767` |
| final_raw/mgga-real.xml | `fbfe1263a90c7d51655bce3f287f8722c0ae043d6a8ee2ff689bf8895bf63286` |
| final_raw/quartet-strict.xml | `79e916aeab561756bf2620d5ab44cb75300c80c1c7c894b811987b656e5be367` |

