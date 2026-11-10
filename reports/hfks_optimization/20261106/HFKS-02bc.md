# HFKS-02b / 02c — shell-quartet symmetry and scheduling audit

Status: narrow shell-level improvement passed remote RED/GREEN; additional
thread scheduling and contraction rewrites are not justified by measured
hotspot evidence. Large-system clean timing and final acceptance remain with
HFKS-02d. This report does not change `plan.md` or certify a full-SCF speedup.

Baseline checkout: `f0ae2d9b`, including selected-product screening from
`e1537173`. Numerical execution was coordinated in Slurm job `116663`; no
local JAX/NumPy/PySCF execution or native build was used for this audit.

## Observed waste and the retained change

`unique_eri` in `coulomb_shells.h` previously visited `i >= j`, `k <= i`,
`l <= k`, called libcint for the complete shell block, and only then rejected
AO entries with reversed pair order. For `i == k`, both `(i,j,i,l)` and
`(i,l,i,j)` were evaluated. With one AO per shell one of those blocks is
discarded completely. With multiple AOs per shell both blocks can contribute,
so restricting the shell loop alone would lose valid values.

The candidate sets `l <= j` when `k == i`. In each retained block it emits a
reversed AO-pair tuple in canonical order when the two shell pairs differ.
When the shell pairs are equal, the block already contains both AO-pair
copies, so the existing AO duplicate exclusion remains necessary.

For `N` shells with zero screening, let `P = N(N+1)/2`. The independent
unordered shell-pair-pair count is `P(P+1)/2`. The previous traversal evaluated
an additional `N(N-1)(N+1)/6` shell blocks. In the four-shell fixtures the
observed count changes from 65 to 55 (10 fewer calls, 15.38% of evaluations).
The same count applies to the shared packed `s4` and `s8` value traversal.
This is a measured call-count reduction, not a measured wall-time speedup.

## Ownership, screening and derivative boundaries

- `unique_eri` is used by `direct_jk.cc` and `packed_eri.cc`; no new backend,
  tensor cache, retained quartet list or worker buffer is introduced.
- Basis Schwarz and product-specific fixed-reference screening are unchanged.
  Both tests are invariant under pair exchange, including nonsymmetric and
  complex density payloads. No numerical cutoff or accuracy gate was relaxed.
- Private-output workers still partition the outer shell by the existing
  stride. Admission retains the eight-real-batch limit, the caller quota
  (default 64 MiB), the worker cap, and fixed worker-order reduction.
- When private outputs are unavailable, workers retain disjoint output rows.
  A quartet can still be evaluated by several row workers because different
  output rows depend on it. Removing this deliberate bounded-memory tradeoff
  is outside this patch and would require its own measurements and tests.
- Geometry and primitive-basis derivatives use `ERICanonicalQuartets` from
  `eri_symmetry.h`, through `eri_geometry_entries.h` and `eri_basis.h`.
  These paths already separate shell orbits from differentiated-leg modes.
  They were inspected and regression-tested, but not modified.
- Pair canonicalization changes some floating-point accumulation order and
  which symmetry-related libcint block supplies an entry. Cross-path checks
  use explicit tolerances; repeated execution at fixed worker settings is
  checked for exact repeatability.

## RED, GREEN and independent reference

The coordinator ran Slurm job `116663`. Its synchronized raw XML was read
independently for this audit; both phases ran on host `n006`:

| Phase | Selection | Observed result |
|---|---|---|
| RED | New quartet test module | 9 intended count failures, 40 passed, 0 errors; XML time 17.183 s |
| GREEN phase02 | New module plus selected/density screening, parallel direct, ERI basis/geometry symmetry regressions | 167 passed, 6 CUDA skips, 0 failures/errors; XML time 46.526 s |

All nine new count gates passed in GREEN. The other 40 new tests passed
already in RED and protect the existing values and derivatives. Test-suite
elapsed times include compilation and instrumentation and are not clean
before/after performance measurements. Every failing RED count is 65 versus
the required 55; all nine corresponding GREEN counts are 55. XML timestamps
are 2026-11-06 16:08:43 +08:00 (RED) and 16:26:40 +08:00 (GREEN).

- RED: [direct_quartet_schedule-red.xml](final_raw/direct_quartet_schedule-red.xml),
  diagnostic library SHA256
  `a2d2089ab1ad16b0678b75b9d7c484353e40e09f0d6015e897ca537a972da8f8`.
- GREEN: [quartet-green.xml](final_raw/quartet-green.xml), diagnostic library
  SHA256 `81825e204306f2b1c55b6c879026515dad6255bbbcebc6674d81eb8a48963928`.

Raw artifacts remain outside the intended committed report/source subset.

The new `tests/integrals/test_direct_quartet_schedule.py` uses independent
PySCF full ERIs, not a dense tensor produced by the modified shared traversal.
It covers one-AO shells and mixed generally contracted s/d shells in Cartesian
and spherical representations, repeated shell indices, nonsymmetric complex
density batches, J-only/K-only/J+K, Coulomb and `omega = 0.3 Bohr^-1`, requested
workers 1/4, default private quota and explicit zero-quota row fallback, density
JVP/VJP and exact repeated-run agreement. Native active workers remain capped
by the available XLA pool and shell count.

Coordinates are in Bohr and computations use CPU float64. Values and density
JVP use `atol = rtol = 1e-10`; density VJP uses `1e-9`. Geometry JVP and the
VJP directional contraction are compared with independent PySCF central
differences at a `1e-4` directional step, using `atol = rtol = 3e-8`. The same
finite-difference tensor checks packed `s4`/`s8` JVP. Six GPU tests were skipped
in this CPU phase, so this result does not certify the modified CUDA bridge.

Final specification review tightened the new helper's default exact-value
and density-JVP comparison from `3e-10` to the frozen `1e-10` requirement.
The RED/GREEN XML above used the earlier test SHA256
`602fc6594c537d5f11957c0341a692c2f9aacb8da6ef9fd08b453c012c7a88ef`;
it proves the historical 65-to-55 diagnostic count result and the earlier
numerical gate. The stricter production-library rerun is now independently
verified from [quartet-strict.xml](final_raw/quartet-strict.xml) and
[quartet-strict.log](final_raw/quartet-strict.log): **40 passed, 9 deselected,
zero failures/errors/skips**, XML time 10.486 s, host `n006`, timestamp
2026-11-06 19:43:51 +08:00. Its test SHA256 is
`32c52348ce071cf2e1750ae1a674ea7be231056026a6f7b6006b4e734da2eede`,
matching the current file. The coordinator identifies the unchanged production
library as SHA256
`8a1f50dcbecec3831446bd60ae9df65796fa359e532c197e4a7811b4272b8b68`;
the XML itself does not embed a library fingerprint. XML SHA256 is
`79e916aeab561756bf2620d5ab44cb75300c80c1c7c894b811987b656e5be367`.
Explicit density-VJP and geometry-FD tolerances are unchanged, and no
production code changed for this tightening. The nine historical diagnostic
count gates were deliberately deselected, not silently counted as passes.

Strict value/AD reproduction on the ordinary production library:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q \
tests/integrals/test_direct_quartet_schedule.py \
-k 'repeated_multiao_shells or shared_derivative_paths'
```

Minimal reproduction, in the allocated Slurm environment with a freshly
built diagnostic library:

```sh
GRADSCF_EXPECT_DIAGNOSTICS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
python -m pytest -q -s tests/integrals/test_direct_quartet_schedule.py
```

Candidate fingerprints at audit time:

| File | SHA256 |
|---|---|
| `src/gradscf/integrals/_native/csrc/coulomb_shells.h` | `fcacc71137adab14f1f943413786dd55637cadfa9ed59641780ab18a40401b20` |
| `tests/integrals/test_direct_quartet_schedule.py` | `32c52348ce071cf2e1750ae1a674ea7be231056026a6f7b6006b4e734da2eede` |

Static self-review found no missing AO-pair class, worker ownership change,
new allocation or derivative-path modification. `git diff --check` passed.

## HFKS-02c decision and remaining gates

The audit identifies possible outer-shell load imbalance and repeated work
in the row-owned fallback, but neither has an isolated current hotspot
measurement demonstrating that a new scheduler or contraction rewrite would
improve the target workloads. Keep the existing scheduling and contraction
design. HFKS-02c can be closed as an audited conditional decision with no
additional rewrite; it must not be described as a scheduling speedup.

The redundant-call fraction decreases with shell count, and Schwarz screening
can change the retained workload. Consequently the four-shell count result
does not predict the 96-atom runtime. The coordinator still needs separate
production-library before/after timings on identical large inputs, clean
CPU/GPU bridge checks, numerical/iteration/call-count comparisons and the
applicable memory measurements. HFKS-02b's historical diagnostic counts and
the tightened `1e-10` value/AD rerun are recorded above as separate evidence.
Final retention and HFKS-02d remain contingent on the remaining large-system
measurement gates; this numerical rerun does not complete those measurements.
