# libcint production interface and derivative-symmetry acceptance

This report combines completed phase1 and S9 bounded acceptance. Phase1 timings
use n005; the subsequent symmetry campaign uses n006. Ratios are calculated
within matched hardware/task scopes, never between these nodes. No merge or
push was performed. Final provenance, resource release and limits are below.

**Methods.** Water at `O 0 0 0; H 0 -.757 .587; H 0 .757 .587` Angstrom, Cartesian def2-SVP (25 AO, 5 occupied, 20 virtual), CPU float64. PBE uses grid level 2 (21,952 points); true RI uses Weigend (retained rank 85). Phase1 CPU: n005, Intel Xeon Gold 6430; Python 3.11.15, JAX/JAXlib 0.10.0, PySCF 2.13.0. Thread counts and actual affinity are retained per process.

Each clean set has one first calculation and three same-process repeats with fresh molecular/method objects, synchronized results, persistent JAX compilation cache disabled, and no final-result cache. The primary time is the raw `stages.task_wall`: core imports precede its timer; remaining lazy imports/JIT and the first GW screening snapshot remain inside the measured scope. Wrapper and whole-process times are retained separately in the JSON. Times are not reduced by profiler or component measurements.

**Results.** Seconds below use exactly `before-final/after-final`, `cpu8-before/cpu8-after`, `mp2-before/mp2-after` and `ri-gwbse-before/ri-gwbse-after`. Superseded `after-stage2` data are excluded. A ratio below 1 means less time. The PySCF columns use the matched after reference run.

| Task | Threads | Cold before → after (s) | Warm median before → after (s) | After/before cold / warm | After/PySCF cold / warm |
|---|---:|---:|---:|---:|---:|
| water-pbe-gradscf-cpu1 | 1 | 11.308357 → 10.070024 | 3.368258 → 0.319088 | 0.890 / 0.095 | 19.906 / 1.103 |
| water-ccsd-gradscf-cpu1 | 1 | 18.528514 → 13.854749 | 6.417648 → 0.250382 | 0.748 / 0.039 | 36.280 / 1.127 |
| water-pbe-gradscf-cpu8 | 8 | 6.698632 → 6.756364 | 1.400858 → 0.237561 | 1.009 / 0.170 | 2.728 / 0.105 |
| water-mp2-gradscf-cpu1 | 1 | 7.252698 → 7.193577 | 1.353581 → 0.352697 | 0.992 / 0.261 | 32.853 / 9.544 |
| water-dfmp2-gradscf-cpu1 | 1 | 8.379117 → 8.356562 | 1.810671 → 0.705121 | 0.997 / 0.389 | 44.181 / 18.360 |
| water-ri-gwbse-gradscf-cpu1 | 1 | 156.448359 → 95.493845 | 114.868675 → 57.684669 | 0.610 / 0.502 | no matched reference |

All original samples, medians, matched reference samples, ratio eligibility, process RSS and raw paths are in `before_after.csv`; `phase1_test_evidence.json` additionally retains commands, input SHA256, all per-repeat numerical results and artifact SHA256. RSS is process-wide across startup/compilation/repeats, not an isolated warm peak; group RSS is sampled every 50 ms and worker `ru_maxrss` is a separate high-water statistic.

| Task | Peak group RSS before → after (MiB) |
|---|---:|
| water-pbe-gradscf-cpu1 | 1206.8 → 1155.0 |
| water-ccsd-gradscf-cpu1 | 1480.4 → 1165.6 |
| water-pbe-gradscf-cpu8 | 1231.3 → 1130.6 |
| water-mp2-gradscf-cpu1 | 1068.7 → 1032.3 |
| water-dfmp2-gradscf-cpu1 | 1167.7 → 1204.2 |
| water-ri-gwbse-gradscf-cpu1 | 3668.0 → 2669.1 |
| water-gwbse-gradscf-cpu1 | 9459.0 → 2827.9 |

**Ordinary spectral GW/BSE.** `gwbse-long-before/runs/water-gwbse-gradscf-cpu1/process.json` records a 6000.227477 s process timeout, with two valid complete results: r0 = 2538.074358 s and r1 = 2491.742208 s task time; r2 was unfinished. Each valid row has all 25 QP roots and five TDA plus five full-BSE roots. One cold run also had external py-spy concurrency (root-reported provenance). These raw times are retained, but the before run does not provide cold + three warm samples, so no formal speed ratio is reported. The initial 600 s timeout directories remain historical evidence.

`gwbse-long-after` completes all four full conventional/spectral calculations. Raw task times (s): [118.28234359482303, 73.59557470493019, 73.6032909960486, 73.74325858615339]. This is separate from true 2c/3c RI; neither timing substitutes for the other task.

**Numerical gates.** Recorded total/HF/correlation energies are compared at 1e-7 Ha; matched QP/BSE energies at 1e-3 eV. The table reports maxima over every available before/after and reference repeat, not only the last row. Convergence flags are true on every accepted completed row.

| Task | Max before/after energy difference (Ha) | Max after/PySCF energy difference (Ha) | Max SCF commutator |
|---|---:|---:|---:|
| water-pbe-gradscf-cpu1 | 0.000e+00 | 0.000e+00 | 3.950e-09 |
| water-ccsd-gradscf-cpu1 | 2.776e-17 | 7.711e-11 | 2.989e-11 |
| water-pbe-gradscf-cpu8 | 0.000e+00 | 7.105e-14 | 3.950e-09 |
| water-mp2-gradscf-cpu1 | 0.000e+00 | 2.619e-11 | 2.989e-11 |
| water-dfmp2-gradscf-cpu1 | 0.000e+00 | 2.591e-11 | 2.920e-11 |

CCSD residual maximum is 3.536e-10 (1e-9 bound). SCF commutators are below the recorded 1e-8 gradient bound. GW equation residuals use the separately registered 1e-6 Ha certification bound; BSE residuals use 1e-8. The QP certification bound is distinct from the solver `conv_tol=1e-10` setting.

| GW/BSE task | Max QP difference (Ha) | Max TDA / full-BSE difference (Ha) | Max QP residual (Ha) | Max TDA / full residual | Candidate counts / branches |
|---|---:|---:|---:|---:|---|
| water-ri-gwbse-gradscf-cpu1 | 4.441e-15 | 8.327e-16 / 2.776e-16 | 1.693e-08 | 9.811e-09 / 7.342e-09 | identical / all valid |
| water-gwbse-gradscf-cpu1 | 3.553e-15 | 1.499e-15 / 1.055e-15 | 1.865e-08 | 4.734e-09 / 8.375e-09 | identical / all valid |

All 25 candidate-count and branch-valid vectors are preserved for every completed repeat in the JSON. Ordinary spectral numerical parity uses the valid r0/r1 before outputs; timeout status remains visible and excludes a timing ratio. RI before/after have four completed repeats each. Shared-input `qp-equation-check.json` and `bse-solver-boundary.json` are equation/solver diagnostics, separate from independent geometry-to-result timings; no universal dense/Davidson threshold is inferred.

**Selected validation.** The following are bounded selections. They overlap and are not summed into a total method-suite count.

| Evidence log/XML basename | Passed | Seconds |
|---|---:|---:|
| final-core | 19 | 133.212 |
| final-geometry | 9 | 324.603 |
| final-neural-degeneracy | 9 | 92.102 |
| public-final-green | 11 | 8.727 |
| binding-threads-green | 28 | 18.018 |
| nnao-green | 24 | 141.804 |
| cache-nnao-green | 11 | 38.349 |
| molecular-forward-green | 4 | 98.883 |
| gwbse-green | 15 | 66.532 |
| resolvent-green | 45 | 132.349 |

`final-geometry` deselects six cases. Earlier `core-stage2` (48/50), `public-window-green` (one failure), and `scf-first-green` (one failure) are nonpassing historical selections. `consumers-green` exited 124 with an incomplete log containing a failure marker; it is not a completed passing broad consumer suite. Exact test IDs, exit states and raw hashes are retained in the JSON.

**Installed production smoke.** The final wheel `gradscf-1.0.0-py3-none-any.whl` has SHA256 `f9e4b2f88fe1e3876385d8ea9615e7dca4b4458e1bdc2afcc2174eb3f56f4560`. Wheel creation and installed native build exit 0; the build log retains clock-skew warnings. `no-pyscf-final.log` exits 0 and reports `pyscf_available=false`, with both package and native library under `env-production`; native SHA256 is `8ac8ce4f16f82d07fe15c73b9094517163d804cc0472278214881369fc8264f3`. The smoke executes RHF, PBE, MP2, complete tiny GW/TDA/full-BSE, and an NNAO coefficient/exponent pullback and update. It is an installed-package smoke, not full numerical acceptance of every method.

**NNAO fixture.** The real MACE model uses seed 3, one interaction and four channels on H-only H2 geometries 0.74/0.91 Angstrom. Its loss is a reconverged RHF energy mean; clean full-step times include loss, parameter gradient and JointAdam update. The historical trained checkpoint is unavailable. Fixture assembly bypasses the full 7171-state CHNOF campaign constructor and its filesystem/source guards; neither trained accuracy nor campaign constructor/resume is validated. No active force-supervision workflow or new mixed basis/geometry capability is claimed.

| Small NNAO fixture placement | Cold full step (s) | Warm median (s) | Process peak RSS (MiB) |
|---|---:|---:|---:|
| cpu model/SCF + CPU libcint | 9.095310 | 0.019056 | 1351.7 |
| NVIDIA A800 80GB PCIe model/SCF + CPU libcint | 13.803670 | 0.058483 | 1881.5 |

The four CPU/GPU loss samples agree to 1.776e-15 Ha. Complete-step timings and component runs are separate; backward cost is not inferred by subtraction. These timing JSONs identify source-v2/stage2 and a small fixture, whereas the final installed smoke identifies source-v3. Integral projection and pullback remain on CPU with explicit transfers. Detailed limits are in `nnao_validation.md`.

**Implementation and provenance.** The phase1 chain is:

- `2d965a8` — test: preserve bounded libcint and NNAO validation workflows
- `22251aa` — fix: bind NNAO integral caches to libcint and physical inputs
- `c1d5a6f` — perf: reuse exact RPA factors for bounded CPU GW
- `ac10712` — perf: add DIIS seeding for implicit molecular geometry solves
- `ad65caf` — perf: reuse restricted GW and iterative BSE executables
- `ddba6cc` — perf: reuse SCF and CC executables across fresh inputs
- `e43d0f2` — refactor: unify production integrals behind libcint
- `ad994bb` — fix: restore public TD response imports

`7d2e859` is the preserved pre-migration source snapshot; the phase1 tip is `2d965a8`. Exact retained source files are indexed by `final-source-manifest.json`; original-checkout preservation is separately recorded in `original-worktree-verification.json`. Fixed outer source paths/PYTHONPATH and environment establish run-level provenance. Per-case process records do not independently hash the imported GradSCF tree, so directory labels are not presented as per-case hash attestation or as one uniform before commit for every stage.

Production Gaussian integrals use the single libcint path; pinned licensed filling/ECP C subsets, JAX normalization/projection/grid AO/XC/linear algebra and periodic FFT/GTH responsibilities remain. Removal, ECP and derivative-order limits are documented in `backend_removal_audit.md`; arbitrary higher/mixed AD or a GPU-native integral engine is not claimed.

**S9 mathematics and implementation.** The new scheduler owns canonical real-Coulomb shell quartets with `i>=j`, `k>=l`, and `(i,j)>=(k,l)`. It retains only selected shell-pair membership metadata, O(nshell²), and streams pair combinations whose eightfold orbit intersects the requested rectangle. It does not build an O(nshell⁴) quartet connection list or scan all global quartets for a small selected window.

For independent normalized ENV coefficients, the first directional product is

\[
\mathrm d I[c;u]=\sum_{a=0}^{3} I(c_0,\ldots,u_a,\ldots,c_3),
\]

and the bilinear coefficient product is

\[
\mathrm d^2 I[c;u,v]=\sum_{a\ne b}I(c_0,\ldots,u_a,\ldots,v_b,\ldots,c_3).
\]

All four first-order and twelve ordered second-order occurrences are retained. Only their complete contracted sum is emitted to distinct shell tuples. For example, a single shared coefficient gives `I(c)=c^4 I(1)`, so its products must contain `4 c^3 u` and `12 c^2 u v`; repeated-shell deduplication cannot remove these factors. Primitive/contraction normalization and log-exponent chains remain in JAX outside these bare native products.

For an arbitrary cotangent and window W, the canonical pullback first forms

\[
\bar I_Q=\sum_{\pi Q\in W}P_\pi^{\mathsf T}\bar I_{\pi Q},
\]

then contracts each free derivative leg with this pooled tensor. This is a sum, not an average. The identity-shadow axis has `nprimitive*angular` entries while the pooled cotangent has physical `ncontraction*angular` entries; the implementation maps between them explicitly. A repeated shell already supplies all local AO entries in its native block, so shell-tuple emission deduplication is separate from derivative-position multiplicity.

Coordinate kernels have smaller stabilizers than a complete value tensor. The first derivative retains four differentiated-leg modes. The Hessian retains four same-leg, two same-pair and four cross-pair modes, with both coordinate orders for cross terms. A block is reused only for the same kernel and exactly equal ordered source shell tuple. Cartesian components, AO axes, differentiated positions, and higher-product direction data follow those permutations. Existing Cartesian third/fourth products inherit the shared contraction; unsupported combinations remain explicit errors.

Canonical outer pairs own disjoint basis-product output blocks, allowing parallel JVP writes without a per-worker four-index output. VJPs reduce small private parameter vectors in fixed worker order. Direct J/K has overlapping AO outputs and keeps its existing distinct ownership/reduction strategy. Scratch-size keys include the kernel and actual shadow-shell `l/nprimitive/ncontraction/kappa`, with a 4096-entry bound per driver. No physical integral values or converged electronic states are stored by that cache. The superseded ordered-quartet helper was removed after its consumers migrated.

The matched water/def2-SVP fixed-input diagnostic runs report the following actual libcint evaluations. All three counter repeats agree; all numerical inputs match, and output-array differences are below7.2e-13. These are native product counts, not end-to-end speed ratios. The coefficient/exponent chart holds the other normalized ENV quantities fixed; full raw-parameter normalization is checked separately by the NNAO/consumer gates.

| Fixed-input product | Evaluations before | Evaluations after |
|---|---:|---:|
| coefficient JVP / VJP | 82,944 | 12,324 |
| coefficient bilinear JVP / HVP-adjoint | 248,832 | 36,972 |
| exponent JVP / VJP | 152,064 | 22,594 |
| coordinate JVP / VJP | 20,736 | 11,232 |
| coordinate bilinear JVP / HVP-adjoint | 62,208 | 27,822 |

An intentionally narrow selected window has32 coefficient-VJP evaluations on both sides; symmetry gives no extra reduction there. Packed Hessian products retain packed storage and use the same canonical physical kernel evaluations. Older geometry traversal had no scheduled-quartet/emission hook, so its raw zeros for those counters are unobserved metrics, not zero physical work. `symmetry_counts.json` preserves complete counter snapshots, library identities and paired array errors.

**S9 verification checkpoint.** New product/FD/transpose/thread/counter gates passed41 tests in14.09s; existing higher/window/packed/slab/direct/range regressions passed128 in39.65s. After removing the dead traversal, both library modes rebuilt and the41-product group passed again. Independent metadata enumeration checked11,378 small nonempty rectangular windows without a coverage/ownership discrepancy. These selections overlap and are not summed into a full-suite count.

A fresh derivative-only process exposed an existing FFI initialization dependency: opaque bounded primals could be eliminated before registering handlers. Twelve fresh-process JVP/VJP/HVP cases failed before the fix; all12 and18 existing packed cases passed afterward (30 in32.51s). The geometry family now registers its own native handlers. The initial clean attempt is retained as `clean-attempt1-20261027`; microkernel library loading is reported separately and identically for both versions. Full geometry-to-result workflow timing is unchanged.

**S9 final scientific acceptance — 2026-10-27.** Offline checks of the recorded calculations pass the registered numerical and provenance gates. This closes the bounded scientific S0–S9 acceptance for the stated products and representative consumers. It does not mean every test in every method suite was run, or that every task became faster. Original-worktree and installed-source proofs, focused commits and normal resource release are recorded in the finalization section.

**Authoritative clean record.** `clean-v3` contains 32 micro-product JSON/NPZ records forming 16 matched pairs (13 CPU1, three CPU8) and 12 formal processes (PBE0, PBE-TDA and full PBE-TDDFT × GradSCF/PySCF × before/after), each with one cold and three warm samples. All run on n006, AMD EPYC 9654; GPU fixture/device tests use one GeForce RTX 5090. Actual imported package tree hashes, loaded native library paths/SHA256 and probe hashes are recorded after numerical samples. The before native SHA256 is `8ac8ce4f16f82d07fe15c73b9094517163d804cc0472278214881369fc8264f3`; after is `c4756801f7cad8aa826760fb5d761ba7ea6f345606c17402401cc8f4ac6b384b`. These actual identities, rather than old display labels, establish the comparison.

Every micro pair has bitwise identical input arrays, equal dtype/shape and numerical-input hashes. Both cold and warm result arrays pass atol=3e-10/rtol=3e-9; the largest absolute difference is 7.105e-13. Native diagnostic counters are disabled in every clean micro run, `timing_eligible=true`, and persistent compilation caching is off. Cold timing is lower+compile+execute; library loading and preparation are reported separately. Warm timing is the median of three synchronized executions. The CSV retains all raw samples and min/max, not only rounded table values.

| Fixed-input product / CPU threads | Cold before → after (s) | Warm median before → after (s) | Warm after/before | Peak process RSS before → after (MiB) |
|---|---:|---:|---:|---:|
| coefficient_hessian_jvp-cpu1 | 0.400119 → 0.142668 | 0.314973 → 0.053822 | 0.171 | 653.8 → 656.7 |
| coefficient_hessian_vjp-cpu1 | 0.470590 → 0.158876 | 0.375576 → 0.065343 | 0.174 | 640.5 → 638.4 |
| coefficient_jvp-cpu1 | 0.192157 → 0.098741 | 0.106430 → 0.021160 | 0.199 | 653.1 → 656.4 |
| coefficient_vjp-cpu1 | 0.224796 → 0.105932 | 0.128327 → 0.027217 | 0.212 | 641.1 → 640.7 |
| coefficient_vjp-cpu8 | 0.089730 → 0.058824 | 0.029857 → 0.004435 | 0.149 | 658.2 → 657.4 |
| coefficient_vjp-window-cpu1 | 0.083149 → 0.082780 | 0.000052 → 0.000062 | 1.175 | 638.8 → 636.6 |
| exponent_jvp-cpu1 | 0.379577 → 0.149942 | 0.284899 → 0.053017 | 0.186 | 660.7 → 653.3 |
| exponent_vjp-cpu1 | 0.378408 → 0.143688 | 0.282493 → 0.051255 | 0.181 | 640.8 → 639.2 |
| exponent_vjp-cpu8 | 0.120914 → 0.065285 | 0.066404 → 0.008802 | 0.133 | 662.8 → 659.2 |
| geometry_jvp-cpu1 | 0.142036 → 0.124280 | 0.054106 → 0.040439 | 0.747 | 655.8 → 658.6 |
| geometry_vjp-cpu1 | 0.149444 → 0.127962 | 0.059924 → 0.042077 | 0.702 | 644.3 → 645.2 |
| hessian_jvp-cpu1 | 0.509499 → 0.318853 | 0.400585 → 0.210423 | 0.525 | 659.7 → 659.7 |
| hessian_vjp-cpu1 | 0.523348 → 0.366381 | 0.403446 → 0.248193 | 0.615 | 648.0 → 643.1 |
| hessian_vjp-cpu8 | 0.468401 → 0.322008 | 0.405335 → 0.248390 | 0.613 | 661.7 → 663.0 |
| hessian_vjp-s4-cpu1 | 0.453223 → 0.291498 | 0.338859 → 0.179261 | 0.529 | 636.3 → 638.5 |
| hessian_vjp-s8-cpu1 | 0.436562 → 0.293151 | 0.327428 → 0.173619 | 0.530 | 640.2 → 642.3 |

Lower ratios mean less time. The narrow coefficient-VJP window regresses from 52.398 to 61.572 microseconds; its diagnostic evaluation count is 32 on both sides, so it has no symmetry work reduction. This measured overhead is retained. CPU8 rows compare CPU8 before with CPU8 after; they do not establish an independent scaling law.

**Matched public forward workflows.** All 48 completed worker results satisfy the 1e-7 Ha total/HF-energy and 1e-4 eV excitation gates before timing ratios are admitted. Actual per-element basis arrays match; their Cartesian AO count is 25 (spherical would be 24), and saved overlap arrays confirm the 25×25 representation. Every row has grid size 21,952, 5 occupied/20 virtual orbitals and 100 transitions. TDA/TDDFT request five states; completed SCF/recorded response flags are true. Before/after and GradSCF/PySCF overlap/hcore arrays also agree. Each raw worker record contains actual imported source and library provenance.

| CPU1 workflow | Cold GradSCF before → after (s) | Warm GradSCF before → after (s) | After/before cold / warm | After/PySCF cold / warm | Max after/PySCF energy error (Ha) / excitation error (eV) |
|---|---:|---:|---:|---:|---:|
| pbe0 | 9.597429 → 9.568010 | 0.278972 → 0.270626 | 0.997 / 0.970 | 23.801 / 1.083 | 1.705e-13 / N/A |
| pbe_tda | 32.148237 → 32.664674 | 6.816957 → 6.781479 | 1.016 / 0.995 | 30.929 / 7.342 | 7.105e-14 / 1.256e-08 |
| pbe_td | 27.440142 → 27.725909 | 5.849330 → 5.873606 | 1.010 / 1.004 | 24.595 / 5.891 | 7.105e-14 / 1.256e-08 |

These forward timings remain nearly unchanged between versions. TDA/TDDFT remain slower than the matched PySCF calls; derivative-kernel count reductions are not presented as an end-to-end forward speedup. `stages.task_wall`, wrapper time and whole-process time are retained separately. RSS is a whole-process high-water mark including preparation, compilation and repeats, not isolated warm memory. Formal group RSS comes from the controller’s nominal 50 ms sampler. All 12 time-series files were independently parsed: sample counts match the controller records, and each observed maximum equals its reported process-group peak.

**Representative consumer acceptance.** Before and after each complete all 12 cases across 11 consumer groups. These are diagnostic validation fixtures: before uses CPU1, after CPU8, and all have `timing_eligible=false`; no time ratio is calculated. Their exact derivative scopes, requested outputs, selected JUnit IDs and results are preserved in `s9_results.json`.

| Consumer probe | Before / after | Recorded scope |
|---|---|---|
| cc_eom_response | ok / ok | selected relaxed UCCSD(T) R2; native EOM supplied-MO/ground response |
| ci_fci | ok / ok | selected relaxed CI/FCI; supplied-frame UCI and FCI eigenpair partials |
| excitation_geometry | ok / ok | relaxed R/U response and CIS transition-amplitude geometry |
| gaussian_ofdft | ok / ok | fixed Gaussian S/T/V/s4 inputs, self-consistent potential response |
| gw_bse_response | ok / ok | RHF relaxed geometry/optical R1; UGW/qsGW/scGW fixed-input partials |
| mp2 | ok / ok | fresh conventional RHF/MP2 value |
| nnao | ok / ok | unchanged real-MACE fixture; native fixed-aux c/log-alpha energy gradient |
| periodic_na | ok / ok | FFT/GTH periodic control; molecular native 4c counts N/A |
| recursive_mp | ok / ok | complete selected R/U MP4 geometry; recursive order distinct from AD order |
| ri_mp2 | ok / ok | fresh true-weigend-RI RHF/MP2 value; no RI nuclear AD claim |
| scf_geometry | ok / ok | complete moving-integral/grid/SCF response |
| spin_generalized | ok / ok | UHF/UKS relaxed geometry; GKS fixed-input partial and value |

The periodic FFT/GTH case is an N/A control for molecular libcint counts. True-RI MP2 remains distinct from conventional 4c MP2; the RI value fixture makes no new RI nuclear-AD claim. GKS fixed-input partials, relaxed spin/geometry responses and selected CC/EOM/GW/BSE products retain their stated scopes rather than implying arbitrary self-consistent derivatives.

**Final selected test groups.** Selections overlap and are not added into a full-suite total.

| XML basename | Passed | Seconds | Scope |
|---|---:|---:|---|
| cleanup-final-products | 41 | 14.092 | basis/coordinate products, FD/transpose/thread/counters, shell scratch cache after cleanup rebuild |
| higher-packed-direct-green | 128 | 39.648 | existing higher-order geometry, windows/slabs, packed, direct J/K and range-separated regressions |
| first-call-green | 30 | 32.513 | 12 fresh derivative-only cases plus 18 packed regressions |
| harness-provenance-green-v2 | 9 | 0.202 | actual provenance, synchronization and consumer-probe metadata |
| probe-adapter-final | 4 | 0.077 | consumer adapter boundary records |
| nnao-recovery-regression | 2 | 1.221 | existing optimizer failure/checkpoint recovery behavior |
| final-regressions | 44 | 106.640 | public libcint interface/binding, solver/GW-BSE reuse, NNAO cache/consumer gates |
| mp2-5090-geometry | 1 | 113.702 | N2 MP2 CPU/RTX5090 SCF, geometry gradient and HVP parity |
| installed-first-call-green | 12 | 22.673 | 12 derivative-only fresh-process cases from the final installed package |

The final 44-regression group records one Flax deprecation warning. The RTX5090 MP2 log records hwloc affinity warnings and then successful CPU and CUDA SCF/gradient/HVP completion. These logs are retained; neither warning is suppressed or used as a substitute for the numerical test result.

**NNAO before/after and CPU/GPU acceptance.** All four full-step runs complete four measured steps plus an untimed final validation and checkpoint roundtrip. They use the same random seed-3 real-MACE H-only two-H2 fixture, model/solver settings and masks. Loss/gradient-norm/beta trajectories agree before/after; final checkpoint parameter, gradient and safely decoded optimizer arrays agree within atol=3e-10/rtol=3e-9. Checkpoint metadata reports epoch 4 and the JSON final loss; optimizer count is independently 4. Frozen theta indices [1,258] have zero gradients/moments, inactive beta coordinates remain zero, and frozen parameter values agree. Checkpoint arrays were loaded directly without importing JAX/model code or rerunning chemistry.

| Prepared small fixture | First full step before → after (s) | Three warm samples before (s) | Three warm samples after (s) | Warm median before → after (s) |
|---|---:|---|---|---:|
| cpu | 17.460279 → 17.288471 | 0.019832, 0.018423, 0.018370 | 0.019884, 0.018165, 0.017990 | 0.018423 → 0.018165 |
| gpu | 12.705400 → 12.650939 | 0.051373, 0.050100, 0.049904 | 0.052009, 0.050443, 0.050470 | 0.050100 → 0.050470 |

CPU/GPU final parameter and gradient checkpoints also pass the same numerical comparison. CPU uses one logical CPU; the GPU fixture uses RTX5090 model/SCF plus CPU libcint and its recorded eight-CPU affinity. The separately instrumented `nnao-after-gpu-components.json` is a diagnostic and is excluded from these clean full-step timings. This small random fixture does not validate trained-model accuracy, an unavailable historical checkpoint, or the real 7171-state training constructor/resume. The existing force/mixed-derivative limits remain unchanged.

**Final installed production check.** `installed-final-s9.exit` is 0. The fresh no-PySCF smoke reports `pyscf_available=false` and both module and native library under `env-production-final-s9`, with the same after-production native SHA256 listed above and counters disabled. It completes tiny RHF/PBE/MP2/GW/TDA/full-BSE plus NNAO coefficient/exponent pullback/update. Its 12 fresh derivative-only installed-package tests pass in 22.673 s. Installation, smoke and targeted numerical/AD evidence remain distinct checks.

**Preserved recovery and metadata evidence.** `clean-attempt1-20261027` remains archived. The registration gap produced 12/12 failing fresh derivative-only cases; after handler initialization was made independent of an eliminable primal, 12 fresh cases plus 18 packed cases passed (30 in 32.513 s). Historical requested-engine/source display labels are preserved rather than rewritten. Every authoritative micro pair and formal process was remeasured in metadata-rich `clean-v3`, using actual imported tree and loaded-library identities, so old labels do not stand in for those checks.

**Remaining TD response cost.** The separate CPU1 PBE-TDA profile measures four
`backend_compile_and_load` calls in the second fresh-object forward task,
4.981881s cumulative, with 6.8723s in the public response call. Lazy reference
preparation accounts for 0.050817s. These nested times overlap and are not added
or interpreted as backward cost. Source review identifies fresh Davidson/grid
control-flow/operator closures as candidates; the existing PBE `grid_hessian`
already caches its compiled callable. Causal isolation of the particular four
modules remains future work. Simply caching a molecule would freeze physical
inputs, and wrapping the entire facade in JIT can change tracer-dependent
exchange/direct-integral branches. No such redesign was introduced in this
bounded migration. Exact profile stats and limits are in `tddft_profile.json`.

**MOLGW reference boundary.** The existing 3.4 reference executable at upstream
revision `b831818d7a845c36f295d036dc8ceef59daf9991`, SHA256
`bedee25b4d99844d9a59cb834a9a509dc0d276b48103aaa5a71da0ab105075bf`,
completes the unchanged upstream CH4 BSE-scissor input in 0.841395s with finite
results. Its input SHA256 and empty comparison-override environment are in
`evidence/molgw-upstream-scissor/result.json`. This differs from water with 25
nonlinear QP targets and five BSE roots, so its speed ratio is null. The earlier
water/restart NaN case has not been independently closed here; the successful
capability diagnostic is not a same-task MOLGW baseline.

**Finalization.** All registered bounded scientific gates are complete. The
installed package has no PySCF import available; all 486 checked Python/C++/basis
resource files match this worktree, and the loaded library belongs to that
isolated prefix. The original checkout retains its initial HEAD/index, all 81
selected overlay files, and all 1901 starting source entries byte-for-byte.
Allocation 109318 ended normally with `COMPLETED / exit0:0` after 03:46:59;
only the project's idle holder was released. Source/test changes have focused
local commits. The repository was not merged or pushed.

**Raw provenance.** `before_after.csv` consolidates 56 rows with explicit phase,
task, scope and device/thread fields. The byte-identical historical table is
retained as `phase1_before_after.csv`; `phase1_test_evidence.json` preserves its
historical measurements. Detailed S9 data are in `s9_results.json` and
`s9_before_after.csv`. Its 348 reviewed artifact hashes are relative to the
durable `evidence/s9/` copy; `/private/tmp/gradscf-s9-evidence` records the
offline review location. Additional final proofs are
`evidence/s9/installed-source-proof.json`,
`original-worktree-verification.json` and `evidence/s9/allocation-final.txt`.
All numerical calculations ran remotely in the allocated Slurm resources.
Local work only inspected source and postprocessed already computed arrays.

Validated platform: Linux, Python3.11.15 and JAX/JAXlib0.10.0. macOS builds and other JAX releases were not exercised in this round.
