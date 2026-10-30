# Molecular G0W0 optimization: measured results

This report analyzes saved remote evidence from 2026-10-30. The baseline is
`a10b8655b99a07dfd53557deda4d6481718534a0`; the implementation/recovery commits
are `a799f0e`, `c79d488`, `be50190` and `28af7c8`. No merge or push is implied.
The resumed corrected-source narrow gate is **passed**: `raw/results/resumed-narrow.xml`, 35 tests, 0 failures, 0 errors, 0 skipped, exit 0; 95.473 s.

## Methods

All molecular probes use Cartesian Gaussian orbitals, RHF, true RI with the
`weigend` auxiliary basis, float64, full G and W orbital spaces, finite eta,
and the original nonlinear CD/QP equations. The response entry is explicitly
pinned to `resolvent_expansion=False`. Water/def2-SVP has 25 AOs, 85 auxiliary
dimensions, four requested QPs (zero-based orbitals 3–6), 32 imaginary nodes,
and eta=0.001 Ha. Water/def2-TZVP has **48 AOs**, 85 auxiliary dimensions,
six QPs, 64 nodes and eta=0.001 Ha. Benzene/def2-SVP has 120 AOs, 438
auxiliary dimensions, six QPs (18–23), 64 nodes, 21 occupied/99 virtual
orbitals and 2079 screening transitions. These distinct workloads do not
define a universal scaling exponent or an aggregate speed average.

SCF energy convergence is 1e-12 Ha and requested QP residuals must be below
1e-10 Ha, with convergence/computed coverage, valid branches and finite
positive Z. Exact rearrangements use atol=1e-10 Ha, rtol=1e-9; same-branch QP
changes must be at most 1e-8 Ha. Candidate counts, branches, residuals and Z
are compared as well as QP energies. No orbital window, node count, seed,
dtype or scientific tolerance is relaxed for a performance comparison.

Each variant starts in a fresh process with the compilation cache disabled.
Repeat 0 includes first-use compilation; warm repeats reuse compiled programs,
rebuild RI/SCF from a new facade, and never reuse a saved QP result. Timers
synchronize the whole JAX result. End-to-end includes molecular/native RI
preparation, SCF, explicit transfer (where applicable) and GW. Separate SCF,
transfer and GW timers are retained in the machine-readable records. The
per-calculation OS alarm is 6000 s; voluntary resource stops are separate.
Tables give warm medians, observed [min,max] and actual sample counts.

CPU hardware is Intel Xeon Gold 6230 @ 2.10 GHz; CPU64 is 64 logical CPUs
covering 32 physical package/core pairs, native threads=64 and
OpenMP/OpenBLAS/MKL threads=1. CPU1 uses one logical CPU and native thread.
GPU response uses `cuda:0` in the V100 allocation, with native RI/SCF on CPU
and 445200 estimated physical input-transfer bytes per water repeat.
JAX/jaxlib 0.10.0, NumPy 2.4.4, SciPy 1.17.1 and PySCF 2.13.0 were recorded.
The isolated native library SHA256 is
`c4756801f7cad8aa826760fb5d761ba7ea6f345606c17402401cc8f4ac6b384b`.
`environment-final.json` checked 27 source files without mismatches; recovery
comparison matched those files except the declared test-only budget change.

The retained molecular path applies target/G/W selection before AO2MO, using
`gradscf.ao2mo` ordered Qia/Qpm/Qmp vertices. Imaginary W uses the same ordered
B[m,p] vertex on both sides; residues retain separate left/right vertices.
Bare exchange retains all occupied reference orbitals. Each fixed imaginary
dielectric factor serves RHS blocks and both UGW frames. Projected residues
solve `[I-Pi(z)]x=Pi(z)r` and immediately contract `l^T x`, preserving finite
eta, static subtraction, half-pole weights, six candidate seeds, frequency
slopes and branch certificates. CPU sequential callbacks skip inactive poles;
checkpointing recomputes local dense intermediates in reverse AD. Existing
periodic q0/off-diagonal matrix contracts are retained.

Known residents, promoted factors, primal/tangent weighted factors, complex
solves, RHS/frequency batches and caller reservations use the common decimal-MB
budget. Small GPU batches retain the dense path when their entire working set
fits. Optional matrix-free screening uses the existing checked shared complex
GMRES; real factors act separately on real/imaginary vectors. Independent
`GW(mf).density_fit(auxbasis)`/`UGW(mf).density_fit(auxbasis)` prepares fixed-
geometry libcint RI without rerunning SCF and reconstructs F-h-J from the actual
SCF Coulomb provider when fits differ. Cache ownership/capacity checks precede
preparation and transfer. Full packed 3c supply and selected-vertex caches must
still fit; there is no out-of-core provider. Production native, SCF, CC and
periodic numerical kernels were not changed.

## Results

### Paired molecular timings

All values are seconds. Ratios are before median / after median, descriptive
for each stated case only.

| Workload/stage | Before median [range] | After median [range] | Ratio |
|---|---:|---:|---:|
| Water/def2-SVP CPU1 GW | 4.55594 [4.14675, 4.58865] (n=3) | 0.102206 [0.0827751, 0.115745] (n=3) | 44.58 |
| Water/def2-SVP CPU1 end-to-end | 5.00216 [4.68432, 5.07407] (n=3) | 0.555054 [0.526939, 0.575896] (n=3) | 9.012 |
| Water/def2-TZVP CPU64 GW | 2.31556 [2.29416, 2.34548] (n=3) | 0.261071 [0.245351, 0.264406] (n=3) | 8.869 |
| Water/def2-TZVP CPU64 end-to-end | 2.70675 [2.70026, 2.72166] (n=3) | 0.631956 [0.621769, 0.642275] (n=3) | 4.283 |
| Benzene/def2-SVP CPU64 GW | 334.902 (n=1; dispersion unknown) | 7.68323 [7.57155, 7.76629] (n=3) | 43.59 |
| Benzene/def2-SVP CPU64 end-to-end | 336.568 (n=1; dispersion unknown) | 9.23162 [9.08183, 9.2428] (n=3) | 36.46 |
| Water/def2-SVP GPU GW | 0.0682464 [0.0682081, 0.06828] (n=3) | 0.0682179 [0.0681684, 0.0682217] (n=3) | 1 |
| Water/def2-SVP GPU end-to-end | 0.51926 [0.518294, 0.529224] (n=3) | 0.495798 [0.494878, 0.513269] (n=3) | 1.047 |

The water CPU1 pair is the **final** baseline/candidate series, replacing the
earlier projected-only snapshot (3.9762→0.5650 s end-to-end) in headline results.
Small CPU water already had an RPA fast path in the default facade; these
pinned direct-CD timings do not establish a default-facade speedup. Benzene's
2079-transition space naturally uses the direct large-system route.

Benzene baseline cold+one warm completed. The declared cost policy deliberately
stopped it after one complete warm; single-sample dispersion is unknown.
The candidate has three complete warm samples. The CPU8 benzene pilot was
aborted for resource replanning after 14m31s without a first QP set; the saved
343 s RSS observation was earlier than the stop. This is not a timeout, a
completed solve, or a usable timing ratio.

The GPU water GW medians are 68.246→68.218 ms: **no reliable GPU speed gain is
claimed**. End-to-end variation also contains CPU preparation/SCF. Cold
observations and every warm sample are in `before_after.csv` and
`measurements.jsonl`; they are not mixed into warm medians.

### Scientific gates

Maxima below cover all completed repeats in each paired dataset; candidate
counts and branch flags are identical, and all serialized scientific gates pass.
QP residuals are the nonlinear self-energy root residuals. Molecular records
do not serialize separate Sigma components, so no unmeasured component error
is assigned. Synthetic Sigma products and VJPs are checked separately below.

| Case | Max QP change (Ha) | Max root residual (Ha) | Max Z change | Min Z | Gate |
|---|---:|---:|---:|---:|---|
| Water/def2-SVP CPU1 | 1.11e-16 | 4.16e-17 | 2.22e-16 | 0.95010149 | pass |
| Water/def2-TZVP CPU64 | 3.33e-16 | 1.73e-16 | 4.44e-16 | 0.93734869 | pass |
| Benzene/def2-SVP CPU64 | 3.72e-15 | 1.08e-16 | 3.33e-16 | 0.92860199 | pass |
| Water/def2-SVP GPU | 4.44e-16 | 3.47e-17 | 3.33e-16 | 0.95010149 | pass |

### Memory measurements and kernel experiments

These are **actual Linux process RSS high-water observations**, in bytes,
including imports, preparation, compilation, allocator retention and repeats.
They are not the sum of live GW tensors or per-iteration active peaks.

| Molecular process | Before high water (bytes) | After high water (bytes) |
|---|---:|---:|
| Water/def2-SVP CPU1 | 1,322,995,712 | 1,243,791,360 |
| Water/def2-TZVP CPU64 | 3,221,172,224 | 1,662,070,784 |
| Benzene/def2-SVP CPU64 | 69,647,646,720 | 3,719,426,048 |
| Water/def2-SVP GPU | 1,544,568,832 | 1,553,743,872 |

For benzene, before is VmHWM at the voluntary budget stop, possibly including
the start of an interrupted optional repeat; after is complete-process
ru_maxrss. The observation windows differ, so no per-iteration memory ratio
is inferred from 69647646720 versus 3719426048 bytes.

Synthetic residue probes use seeded (31) arrays, 32 MOs, 8 occupied/24 virtual
orbitals, eta=0.01 Ha and omega=0.86 Ha. They measure isolated Sigma products;
`residue-ad128` measures value plus a scalar-parameter VJP. Exact synthetic
Sigma real/imaginary and parameter-gradient differences pass the registered
atol=1e-10/rtol=1e-9 gate. This is kernel evidence, not a molecular total.

| Synthetic execution | Before median [range] (s) | Experiment median [range] (s) | Ratio |
|---|---:|---:|---:|
| residue128-cpu1-projected | 0.0612188 [0.0605091, 0.0661496] (n=10) | 0.0186818 [0.0179033, 0.0231028] (n=10) | 3.277 |
| residue512-cpu1 | 1.64591 [1.64588, 1.74631] (n=3) | 0.279364 [0.27218, 0.280771] (n=3) | 5.892 |
| residue-ad128-cpu1 | 0.210609 [0.207447, 0.223287] (n=3) | 0.0617871 [0.0610617, 0.0618345] (n=3) | 3.409 |
| residue128-gpu-serial | 0.00201205 [0.00193083, 0.00299287] (n=10) | 0.00575937 [0.00564246, 0.0061616] (n=10) | 0.3494 |
| residue128-gpu-tile4 | 0.00201205 [0.00193083, 0.00299287] (n=10) | 0.00751189 [0.00737116, 0.00808392] (n=10) | 0.2678 |
| residue128-gpu-tile8 | 0.00201205 [0.00193083, 0.00299287] (n=10) | 0.00406826 [0.004042, 0.0042499] (n=10) | 0.4946 |

| CPU synthetic probe | Before XLA temporary (bytes) | After XLA temporary (bytes) | Before process RSS (bytes) | After process RSS (bytes) |
|---|---:|---:|---:|---:|
| residue128-cpu1-projected | 25,264,480 | 1,281,968 | 651,300,864 | 634,519,552 |
| residue512-cpu1 | 403,046,752 | 9,328,112 | 1,028,276,224 | 639,205,376 |
| residue-ad128-cpu1 | 63,635,680 | 2,763,592 | 710,680,576 | 681,320,448 |

XLA `temp_size_in_bytes` is a compiler estimate for one compiled executable,
not a measured process peak. The smaller serial/projected and tile4/tile8 GPU
kernels were slower than the dense small-GPU kernel and were not retained for
budget-fitting batches. The production budget-dependent dense choice was
retained; kernel memory reductions are not used to claim a GPU total speedup.

Water GPU allocator counters (per-run manifest) are separate from CPU RSS:

- `bytes_in_use`: 471808 before, 471808 after (bytes; null means unavailable).
- `peak_bytes_in_use`: 285662464 before, 285662464 after (bytes; null means unavailable).
- `pool_bytes`: 538968064 before, 538968064 after (bytes; null means unavailable).
- `peak_pool_bytes`: 538968064 before, 538968064 after (bytes; null means unavailable).

Bytes-in-use counters describe allocator activity; pool counters describe
reserved capacity. The final snapshot is not called an active peak, and GPU
allocator counters are not a whole-process GPU-memory measurement.

### External numerical references

The independent reference case is water/def2-SVP/`weigend`, Cartesian, full G/W,
RHF, the same four QPs, eta=1e-5 Ha, 128 GradSCF/PySCF imaginary nodes and
1e-12 Ha SCF convergence. This tighter quadrature/eta case is separate from
the 32-node timing case. The external gate is 1e-3 eV; conversion is
1 Ha=27.211386245988 eV. Both references generate their own integrals, SCF
orbitals and QPs from common geometry/basis DATA.

| Reference | Max independent QP error (eV) | Warm total seconds, median [range] | Numerical gate |
|---|---:|---:|---|
| pyscf-water-reference-ri-exchange | 6.95237e-08 | 0.167294 [0.147757, 0.183948] (n=3) | pass |
| molgw-water-reference-common-basis | 5.05754e-08 | 3.39576 [3.33065, 3.44637] (n=3) | pass |

The initial PySCF default exact bare exchange differed by up to 2.631 meV
(definition mismatch). Shared-input diagnostics matched S/h, DF Gram matrices
and mean-field energies, then explicit PySCF `vhf_df=True` reduced shared-input
QP disagreement to 5.46059e-11 eV.
That diagnostic is distinct from the independent results in the table. No
production kernel or scientific tolerance was changed for reference agreement.

Original MOLGW revision `b831818d7a845c36f295d036dc8ceef59daf9991` (3.4)
matched all 95 archived source files. Its unchanged executable SHA256 is
`bedee25b4d99844d9a59cb834a9a509dc0d276b48103aaa5a71da0ab105075bf`.
Initial runs failed because libiomp5 was missing and then DEF2-SVP assets were
absent. The valid run supplied the existing runtime only to the subprocess and
exported the common Cartesian orbital/auxiliary basis DATA into its input
directory. It used analytic poles and a 40001-point Sigma grid (step 0.00005 Ha),
fresh subprocess directories and no restart state or reference source patches.
These launch/input failures remain in the evidence and are excluded numerically.

All cross-software `time_ratio` fields remain **null**. MOLGW's analytic poles/
full Sigma grid and upstream root diagnostics differ from GradSCF's CD and
six-seed certification; PySCF also exposes different root diagnostics. Passing
the numerical gate does not equate the timed work or supply missing upstream
residual/branch certificates. Warm upstream totals are descriptive only.

### Derivatives and acceptance scope

The resumed corrected-source narrow gate is **passed**: `raw/results/resumed-narrow.xml`, 35 tests, 0 failures, 0 errors, 0 skipped, exit 0; 95.473 s.

The preceding final-source gate failed only
`test_dense_frequency_batch_charges_weighted_transition_factors`: its 60 MB
fixture could not hold newly declared primal/tangent weighted scratch.
Changing that **test budget to 100 MB** leaves all numerical tolerances intact.
The full traceback is retained, not replaced by static reasoning.
Historical remote XML evidence is listed individually (overlapping suites must
not be added into a distinct-test total):

- `final-narrow.xml`: 35 passed, 0 failed, 0 errors, 0 skipped; 96.459 s.
- `final-source-narrow.xml`: 34 passed, 1 failed, 0 errors, 0 skipped; 91.500 s.
- `projected.xml`: 74 passed, 0 failed, 0 errors, 0 skipped; 310.836 s.
- `geometry-compile.xml`: 3 passed, 0 failed, 0 errors, 0 skipped; 53.862 s.
- `ugw-ad.xml`: 11 passed, 0 failed, 0 errors, 0 skipped; 137.950 s.
- `gmres-hvp.xml`: 3 passed, 0 failed, 0 errors, 0 skipped; 18.713 s.
- `integration6.xml`: 8 passed, 0 failed, 0 errors, 0 skipped; 40.651 s.
- `nnao-qp.xml`: 1 passed, 0 failed, 0 errors, 0 skipped; 14.843 s.
- `final-gpu-array.xml`: 2 passed, 0 failed, 0 errors, 0 skipped; 14.441 s.

Resumed-source evidence (separate runs, also overlapping):

- `resumed-geometry-compile.xml`: 5 passed, 0 failed, 0 errors, 0 skipped; 50.398 s, exit None.
- `resumed-gpu-array.xml`: 2 passed, 0 failed, 0 errors, 0 skipped; 13.773 s, exit None.
- `resumed-narrow.xml`: 35 passed, 0 failed, 0 errors, 0 skipped; 95.473 s, exit 0.

The synthetic four-parameter H2 radial fixture rebuilds native contracted RI,
S/h, overlap transformation, density and orbitals, then differentiates an
implicit SCF solution and nonlinear QP loss. At fixed 0.74 Angstrom it checks
two central-FD steps (2e-4, 1e-4; gradient atol=3e-7, rtol=3e-5), same-shape
parameter rebinding/cache invalidation and one Adam update on CPU. It certifies
neither trained MACE weights, basis accuracy, forces, nor full GPU NNAO training.
Its auxiliary metric is fixed for radial-parameter AD and rejects a geometry
change. Existing full-integral H2 nuclear response/compile tests remain;
full RI nuclear/auxiliary-metric/whitening geometry AD was not added or
approximated by freezing dependencies. The current report makes no claim of
CH4/naphthalene completion, full-suite acceptance or universal system scaling.

## Provenance and reproduction

From the repository root, regenerate without importing numerical packages:

```sh
python benchmarks/probes/g0w0_memory/analysis.py --report-dir reports/g0w0_memory/20261030-a10b865
```

`analysis.py` automatically prefers accepted candidate benzene/GPU files when
present. Its outputs are this report, `before_after.csv`, normalized
`measurements.jsonl` (every raw row plus relative source path, line and SHA256),
and `reference_comparison.json`. CSV stores every warm sample, median/range,
cold observation, science fields, separate RSS/XLA/GPU metrics and exact raw-file
file hashes. All earlier experiments remain available in normalized raw rows;
they are not silently substituted into final pairs.

Paired/experimental sources relative to this report directory:

- `water-def2-svp-cpu1`: `raw/results/final-baseline-water-cpu1.jsonl` and `raw/results/final-candidate-water-cpu1.jsonl`; adjacent `.manifest.json` files.
- `water-def2-tzvp-cpu64`: `raw/results/baseline-water-def2-tzvp-cpu64-flat.jsonl` and `raw/results/candidate-water-def2-tzvp-cpu64-flat.jsonl`; adjacent `.manifest.json` files.
- `benzene-def2-svp-cpu64`: `raw/results/baseline-benzene-def2-svp-cpu64-flat.jsonl` and `raw/results/accepted-candidate-benzene-def2-svp-cpu64-flat.jsonl`; adjacent `.manifest.json` files.
- `water-def2-svp-gpu`: `raw/results/final-baseline-water-gpu.jsonl` and `raw/results/accepted-candidate-water-gpu.jsonl`; adjacent `.manifest.json` files.
- `residue128-cpu1-projected`: `raw/results/baseline-residue128-cpu1.jsonl` and `raw/results/projected-residue128-cpu1.jsonl`; adjacent `.manifest.json` files.
- `residue512-cpu1`: `raw/results/baseline-residue512-cpu1.jsonl` and `raw/results/candidate-residue512-cpu1.jsonl`; adjacent `.manifest.json` files.
- `residue-ad128-cpu1`: `raw/results/baseline-residue-ad128-cpu1.jsonl` and `raw/results/candidate-residue-ad128-cpu1.jsonl`; adjacent `.manifest.json` files.
- `residue128-gpu-serial`: `raw/results/baseline-residue128-gpu.jsonl` and `raw/results/candidate-residue128-gpu.jsonl`; adjacent `.manifest.json` files.
- `residue128-gpu-tile4`: `raw/results/baseline-residue128-gpu.jsonl` and `raw/results/gpu-tile4-residue128-gpu.jsonl`; adjacent `.manifest.json` files.
- `residue128-gpu-tile8`: `raw/results/baseline-residue128-gpu.jsonl` and `raw/results/gpu-tile8-residue128-gpu.jsonl`; adjacent `.manifest.json` files.

External sources: `raw/results/reference-candidate-water-cpu1.jsonl`,
`raw/results/pyscf-water-reference-ri-exchange/measurements.jsonl`,
`raw/results/molgw-water-reference-common-basis/measurements.jsonl` and their
memory/input/stdout records; diagnostics `pyscf-alignment-matched.json`,
`molgw-provenance.json` and `molgw-water-reference-common-basis/basis-data.json`.
`raw/results/molgw-water-reference*` and `pyscf-water-reference` retain initial
failures/mismatched definitions. Resource-stop provenance is
`large_repeat_policy.json`, `parent-benzene-stop.json` and
`cpu8-benzene-aborted.json`; the 14m31s stop is recorded in `PLAN.md` whereas
the aborted JSON's RSS snapshot reports the earlier 343 s observation.
Environment/source provenance is `environment-final.json`, adjacent per-run
manifests, `recovery_byte_comparison.json` and `accepted_source_hashes.json`.
Tests use raw `.xml`, `.log` and available `.exit` files under `raw/results`.
No local JAX/chemistry calculations were run to produce these summaries.
