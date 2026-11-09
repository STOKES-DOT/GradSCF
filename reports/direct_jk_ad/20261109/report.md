# Direct J/K basis and geometry response — 2026-11-09

The fixed Direct J/K AD plan is complete within the derivative boundaries below.
CPU, CUDA-stage, screened spherical, training and forward-preservation gates have
recorded results. The initial UHF interface failure was fixed and the exact
failing case passed in the subsequent 17-case consumer regression. Original
failure receipts remain in the ledger. No full-suite or new NNAO force-training
claim is made. [Acceptance ledger](acceptance.json), [provenance](manifest.json).

The baseline is `96ad38e5`, after the separately completed [XC-locality round](../../xc_locality/20261109/report.md).
This report does not reopen that round or claim completion of a broader HF/KS benchmark matrix.
Scope and resource amendments are recorded in [the fixed plan](../../../DIRECT_JK_AD_PLAN.md).

## Methods and implementation

The existing libcint shell-direct value algorithm remains in use. Coefficient
and exponent products reuse canonical shell evaluators and pool/scatter J/K
within one shell block, without production AO ERI tensors or AO-parameter
Jacobians/cotangents. Tiny dense tensors are test oracles. Basis JVPs accumulate
serially; parallel VJPs reduce parameter/coordinate-sized vectors.

JAX retains primitive and contracted-shell normalization. For each contraction,
`w_p = N_p(alpha) c_p` and `cbar = w / sqrt(w^T S_rad(alpha) w)`.
With `F = (J,K)(cbar, alpha, R, D)`, the implemented chain is

```text
dF = F_R dR + F_cbar [(d cbar/dc) dc + (d cbar/dalpha) dalpha]
     + F_alpha dalpha + F_D dD;    dalpha = alpha dx for log exponents x.
```

The native exponent term holds normalized coefficients fixed; the additional
coefficient term supplies normalization response. Independent `cbar`, `alpha`,
and coordinate operands preserve selective AD: coefficient-only gradients do
not schedule exponent pulls. One-leg identity contractions preserve derivatives
at individual zero coefficients. See [packing](../../../src/gradscf/integrals/backends/native/packing.py)
and [AD rules](../../../src/gradscf/integrals/backends/native/autodiff/direct_basis.py).

Positive-cutoff derivatives use the **primal quartet mask** built before shadow
shells are modified. Its indicator derivative is zero away from threshold
crossings. Tangent/cotangent densities do not replace the screening reference.
This is the derivative of a selected approximation, not a global physical
gradient-error bound. Tests separately check stable masks and unscreened truth.
Static `omega` retains the ordinary Coulomb / erf long-range operator; no derivative with respect to `omega` is claimed.

Supported orders are density response, first raw-coefficient/exponent response,
first/second coordinate response, and mixed `R-c` / `R-alpha` products with their
transpose rules. Pure/cross basis Hessians `c-c`, `c-alpha`, `alpha-alpha` and
third physical derivatives explicitly remain unsupported. A raw packed ENV
entrypoint remains a fixed-basis interface. Bound/projected providers retain
live basis leaves; `prepare_jk` selects direct response for traced parameters.
Its separately budgeted, fixed-basis s8 cache remains an existing optional path.

`MolecularEnergy` now inherits a direct seed's representation, cutoff, worker
count, and memory budget; explicit memory overrides remain effective. Full
workflow defaults remain unchanged. The metadata-only budget probe does not
normalize parameters. Local AO-XC screening still triggers the existing
response guard: it is distinct from ERI quartet screening.

Review exposed missing quadratic schedule storage and a missing HF budget
inheritance. Before repair, the 3000-shell metadata estimates were 1,641,592 /
62,905,312 bytes for 1 / 64 workers, below even one 126,042,000-byte logical
schedule; the adapter returned `None` instead of its seed's 1024 MB budget.
[Three RED cases](raw/budget-red.log) reproduced these defects. Basis and
geometry VJP workers now share one immutable call-scoped schedule. Full windows
reserve arrays directly, eliminating map-construction and vector-growth peaks;
the estimator charges `32 * nshell * (nshell+1)/2 + 256` bytes once per call.
This is an LP64 allocation estimate, not a process RSS cap. Ownership/lifetime
review found no remaining blocker; both metadata tests and budget inheritance
passed within the final CPU sweep. See [workspace estimate](../../../src/gradscf/integrals/backends/native/jk.py)
and [schedule](../../../src/gradscf/integrals/_native/csrc/eri_symmetry.h).

All numerical execution was remote under Slurm on `jy_gpu_2`, node `n006`, using
Python 3.11.15 / JAX 0.10.0 and float64. Final jobs used 8 allocated logical CPUs,
one NV5090 and a 256 GiB allocation; A800/V100 use was authorized but not exercised.
Native integrals still execute on CPU when JAX arrays reside on CUDA, through
the bounded host callback. GPU results establish placement/AD-stage correctness,
not a CUDA-native integral kernel or GPU acceleration. Exact invocations and
resource guards are retained in each `raw/*.process.json` and `raw/queue/*.sh`.

## Numerical results

Counts below overlap and must not be summed into a unique-test total.

| Gate | Observed result | Evidence |
|---|---:|---|
| CPU parameter, preparation, geometry, mixed and shared ERI controls | 86 passed | [XML](raw/parameter-complete.xml), [process](raw/parameter-complete.process.json) |
| Genuine partial-mask spherical s/d, complex batches and zero coefficient | 4 passed | [XML](raw/spherical-masked.xml) |
| New relaxed basis-energy tests on immutable baseline | 2 failed at unsupported direct basis AD | [log](raw/training-red.log) |
| Candidate training consumer checks | 3 passed | [XML](raw/training-green.xml) |
| CPU sweep after schedule/budget repair; UHF failure superseded below | 107 passed, 1 failed | [log](raw/final-cpu.log), [XML](raw/final-cpu.xml) |
| Final GPU stages: ordinary/RSH geometry, c/alpha first and mixed products | 6 passed | [XML](raw/final-gpu.xml), [process](raw/final-gpu.process.json) |
| Final UHF fix, six HF/KS facades, direct/DF None contract and neural-XC training | 17 passed | [XML](raw/consumer-final.xml), [process](raw/consumer-final.process.json) |
| Instrumented symmetry and shell-work checks | 35 passed, 1 missing-env skip | [XML](raw/diagnostic-counts.xml) |
| Exact skipped counter case, correct diagnostics environment | 1 passed | [XML](raw/diagnostic-single.xml), [process](raw/diagnostic-single.process.json) |

The spherical fixture has 14 AOs including five-component d shells, a nonempty
proper subset of quartets at cutoff `1e-3`, and mask identity at every displaced
FD point. First products and both mixed adjoints agree with masked dense first-
derivative/FD oracles. Standard array gates use `atol=3e-10, rtol=3e-9`; mixed FD
uses two steps `2e-4 / 1e-4` and the fixture's declared tolerances.

The two [new training tests](../../../tests/nnao/test_direct_basis_energy.py)
use synthetic H2 at 1.6 Bohr, three primitives contracted to one AO per H, and
train the first H's raw coefficients/log exponents. Every energy includes live
S/T/V, nuclear repulsion and metric transport, with fully relaxed implicit HF.
Full/direct gradients and two-step reconverged FD pass; one synthetic MSE/SGD
update changes both parameter blocks and overlap while decreasing the loss.
These are not MACE training or force-training tests. The third passed node checks
existing all-electron CH4 fixed-pool coefficient training through `BasisRHF(szp3_direct)`
and its [ProjectedNativeDirectBasis path](../../../src/gradscf/model/nnao/training/objective.py).

The initial CPU failure was the UHF `MolecularEnergy(forward="scf")` path
passing `eri=None` into `jnp.asarray`, before derivative evaluation. Commit
`b9a38f7f` converts absent ERIs to the established empty placeholder for direct/DF
and rejects missing full ERIs. The [17-case consumer command](raw/queue/22a-consumer-final.sh)
passed the exact UHF failure, direct/DF None parity, full-None rejection, all six
HF/KS forward modes, three external neural-XC training modes, and existing
neural force-training checks. The latter concern neural XC, not new NNAO
basis-force supervision. The earlier 108-case sweep was not rerun in full.

The diagnostic build passed 35 tests; one counter test initially skipped because
`GRADSCF_EXPECT_DIAGNOSTICS=1` was missing. The exact case subsequently passed
with that flag in CPU-only job 118703. Its recorded one/four-worker counts are
both 55 ERI evaluations for eight real density payloads. This verifies no
worker-multiple evaluation in that fixture, not a universal scaling assertion.
The diagnostic library is separate from all clean performance measurements.

## Fixed-density product cost and numerical checks

[products.py](../../../benchmarks/probes/direct_jk_ad/products.py), `--workers 8`,
uses three water fragments, Cartesian 6-31g*, 9 atoms / 30 shells / 57 AOs, seed
20261109 and a Frobenius-normalized rank-25 PSD density. The scalar is
`Q = 0.5 sum(D*J) - 0.25 sum(D*K)`, not a total SCF energy. Directions have unit
Euclidean norm; geometry coordinates are Bohr and exponent directions are log
exponents. Preparation took 1.302525 s. Each case has one first compiled call
and three warm calls; lowering/compilation are recorded separately in the raw.

| Product | Cutoff | First compiled (s) | Warm median (s) | dQ/dt (Ha per defined direction) |
|---|---:|---:|---:|---:|
| Value | 0 | 0.061195 | 0.056714 | — |
| Value | 1e-12 | 0.035095 | 0.043590 | — |
| Geometry | 0 | 0.713360 | 0.723846 | -0.0210911170469210 |
| Geometry | 1e-12 | 0.463175 | 0.424283 | -0.0210911170469280 |
| Raw coefficients | 0 | 0.377135 | 0.334322 | -0.0155776734312852 |
| Raw coefficients | 1e-12 | 0.210064 | 0.226957 | -0.0155776734312838 |
| Log exponents | 0 | 1.112709 | 1.176005 | -0.0116003348485177 |
| Log exponents | 1e-12 | 0.516312 | 0.495269 | -0.0116003348484854 |

`Q(0)=2.860983984207696 Ha`; at cutoff `1e-12`, `Q=2.860983984207694 Ha`.
All eight declared consistency gates passed: value error <= `1e-8 Ha` and
directional error <= `1e-7`; observed maxima were `1.78e-15` and `3.23e-14`.
The probe's cumulative process RSS high-water reached 777,334,784 bytes;
the guard separately reported sampled process-group peak 796,065,792 bytes
and child high-water 853,000,192 bytes over 28.743763 s. These are different
measurement scopes, not per-kernel incremental memory. These costs include
normalization and scalar contraction but exclude SCF, neural-network execution
and training. [Measurements/provenance](raw/products.jsonl), [process guard](raw/products.process.json).

## Large-system forward preservation

The anchor uses the saved 96-atom / 880-AO GradSCF density, cutoff `1e-12`,
static omega 0, complete J/K outputs and a 1 GiB private-output quota; no SCF.
Earlier 64-worker warm medians initially changed from 10.695451 to 12.598672 s,
triggering regression investigation. Repeating the unchanged baseline gave
12.312232 s versus candidate 12.509017 s; the density-reference variant gave
11.835034 versus 11.456577 s. All compared arrays had maximum error exactly 0.
The observed baseline drift prevents a definitive speed claim. The earlier
[inspection record](handoff.md) found no forward call to the shared predicate
in objdump; a helper-call/forced-inline explanation was not established.
[Initial baseline](raw/forward-before/summary.json), [initial candidate](raw/forward-screen-after/summary.json),
[repeated baseline](raw/forward-repeat-before/summary.json), [repeated candidate](raw/forward-repeat-after/summary.json).

The final CPU8 pair uses identical affinity
`[128,129,147,156,320,321,339,348]` (four physical cores / eight logical CPUs).
Both processes completed. All J/K output arrays match exactly, and neither warm
median exceeds the registered 10% regression target.

| Mask | First compiled, before -> after (s) | Warm median, before -> after (s) | After / before |
|---|---:|---:|---:|
| Basis-only screening | 106.798285 -> 139.996457 | 141.320094 -> 132.092175 | 0.934702 |
| Density-reference screening | 138.867020 -> 121.601871 | 136.178186 -> 117.124977 | 0.860086 |

Preparation is 4.751211 -> 5.615618 s. XLA temporary estimates remain
12,390,400 bytes. Sampled process-group peak RSS is 890,753,024 -> 901,369,856
bytes (0.2 s sampling); child `ru_maxrss` agrees for these processes. These are
whole-process high-water marks, not derivative-workspace estimates. Lowering,
compilation and every warm sample are retained in
[measurements](results/measurements.jsonl) and [CSV](results/before_after.csv).

Baseline has five warm calls, candidate has three, following a registered
amendment before candidate start after Slurm denied a time extension. Physical
inputs, cutoffs, workers, quota and numerical gates are unchanged. Whole-process
wall ratios are invalid. Shared-node drift and the candidate density series
(121.27, 117.12, 98.65 s) prevent a stable speedup claim. The defensible outcome
is preserved forward performance under this bounded test, not a promised
6.5%/14% end-to-end gain. See [amendment](sampling_amendment.md),
[harness diff](sampling_amendment.patch) and [runner](raw/queue/23-final-forward-pair.sh).

## Provenance and remaining limits

The [manifest](manifest.json) audits 586 final source files against both the
local files and committed production HEAD `b9a38f7f`. It includes native library
hashes, input/runtime fingerprints, build receipts, test-source hashes and an
index of raw evidence. Production library SHA256 is
`9f8da65e0e55029dbfcf137eeefd328b5e947015ac0c40a4ae468c3810787e89`;
baseline is `8a1f50dcbecec3831446bd60ae9df65796fa359e532c197e4a7811b4272b8b68`.
Both use libcint 6.1.3. Release builds use `-O3 -DNDEBUG`, system BLAS and
diagnostics disabled for production. Fresh builds passed with clock-skew
warnings retained in raw logs. macOS native compilation was not executed.

Raw logs remain git-ignored under `raw/` and archived on jy_gpu_2; large NPZ
outputs remain remote. Curated measurements, receipts, hashes and the replay
commands are committed. Final numerical jobs have ended. No local scientific
tests or dependency installs were performed; delivery checks only parse
evidence, compare hashes, check syntax and inspect Git changes.

Supported orders remain geometry first/second, basis first and mixed
R-c/R-alpha. Pure/cross basis Hessians, third physical derivatives, a general
GKS/MGGA nuclear-gradient adapter and complete NNAO force learning are outside
this completion claim. Positive screening has a fixed-mask derivative only
away from threshold changes. The separately completed XC locality round and
older HFKS benchmark/recovery gates retain their original scope. This delivery
is authorized for local integration only; no GitHub push is performed.
