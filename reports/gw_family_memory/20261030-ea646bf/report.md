# Molecular GW-family execution and memory results

This iteration implements restricted molecular evGW0/evGW preparation reuse,
streamed qsGW products, bounded Matsubara factorizations, shared canonical
charge search, and actual four-center G0W0/evGW screening. Baseline is
`ea646bf`; production commits are `ca87549`, `12d2ad7`, `28475e6` on
`perf/gw-family-memory-20261030`. Earlier a10b865 results are not these
speed denominators. No merge or push was performed.

All calculations ran through Slurm allocation 113525, n003: one Tesla
V100-SXM2-32GB, 32 logical CPUs (16 physical cores in the recorded affinity,
below the authorized 64-CPU maximum), 256 GiB. Heavy work was serial.
OMP/OpenBLAS/MKL limits were one, native molecular preparation used up to
32 threads. Fixed-input CPU probes were bound to one logical CPU. JAX/jaxlib
0.10.0, NumPy 2.4.4 and SciPy 1.17.1 used float64/complex128. CPU model and
linked BLAS version were not captured; they are null, not guessed.

The unchanged production libcint library SHA256 is
`c4756801f7cad8aa826760fb5d761ba7ea6f345606c17402401cc8f4ac6b384b`.
No integral-engine, SCF, AO2MO, CC, DFT or periodic source changed. Actual
runtime GW file hashes match the local production commit. A facade-only
reservation fix followed timing; timed numerical files are unchanged and
the separate facade-final gate passes. See source_hashes.json.

## Implemented paths

**evGW:** gw/integrals.py prepares distinct ordered OV, pG, Gp and exchange
quantities once per call. evGW changes the W spectrum on every update;
evGW0 caches only original-spectrum imaginary W. Retarded quantities retain
frequency dependence. Data are regenerated on a new physical/model call.
The same stable numerical G0W0 executable serves the facade and restricted
outer loop. E/C/B, mean-field matrices, prepared vertices and W0 retain their
AD dependencies as dynamic PyTrees. No new stop-gradient boundary is added.

**qsGW:** gw/qsgw.py reduces mode-A row/column products directly into an MO
matrix, replacing the frequency-by-MO-cubed W cache and complete Sigma at
each endpoint. Each imaginary frequency factors screening once, then consumes
RHS columns. Retarded zero screening is shared across own poles, preserving
finite eta and the half-residue limits. Raw ordered vertices stay independent.
An eager wrapper reads actual device capacity; a stable numerical core avoids
recreating equivalent compiled loop closures. Public full-matrix reference
kernels remain for regression/API compatibility; production qsGW bypasses them.

**scGW:** gw/matsubara.py factors one frequency at a time with checkpointing.
The eager path shares the existing compiled canonical bracket/bisection search
instead of maintaining another Python implementation. The initial-hit fast
path keeps one Dyson evaluation. Particle number, full matrix G, tails,
reference subtraction, moments, GM energy and implicit equations are unchanged.
Full Pi/W/time outputs remain resident: this is not complete scGW streaming.

**Conventional:** `GW(mf).conventional()` and the restricted array driver use
actual full/packed Coulomb data or native selected blocks through the common
AO2MO module. No auxiliary RI or AO-pair spectral factorization is constructed.
For t=(i,a), the checked transition equation is:

```text
V_tt' = (ia|jb)
Z = (I-D V)^(-1) D C_R
Wc[L,R] = C_L^T Z
D_iw = -4 Delta/(u^2+Delta^2)
D_ret = 2[(omega-Delta+2i eta)^(-1)-(omega+Delta)^(-1)]
```

The imaginary left vertex is (mp|ia), residue left is (pm|ia), and right is
(ia|mp). Shared direct/GMRES solvers provide products and transpose response;
the existing QP selector, frequency AD, slopes/Z and branch checks are reused.
Three duplicated QP mapping paths are consolidated. Real/imaginary matvecs
avoid a whole complex transition-matrix copy. V_OV and selected cross vertices
must fit: cache-free conventional screening is not implemented.

Preflight includes source promotion, native optimizer/projection scratch,
live potential/coefficient owners and realified solver/RHS work. The actual
SCF AO parameters and normalization are bound, rather than rebuilding a
potentially different named basis. Unimplemented combinations raise explicitly.

## Matched CPU results

CH2O: repository geometry, Cartesian def2-SVP, weigend RI, RHF conv_tol=1e-12,
all G/W states, six updated targets 5–10, eta=0.001 Ha, nw=32. Both versions
use max_iter=60, outer tolerance 1e-8 Ha and damping 0.2. Outputs/initial guess
are unchanged. Each row has one first execution plus three synchronized,
new-object warm executions. Exact samples/ranges are in before_after.csv.

| Workload | First E2E before → after, s | Warm E2E median, s | Warm GW stage median, s | Warm E2E factor |
|---|---:|---:|---:|---:|
| ch2o-evgw0 | 25.4501 → 11.8166 | 8.6702 → 2.2616 | 8.2433 → 1.8436 | 3.83× |
| ch2o-evgw | 30.0327 → 10.6547 | 13.3605 → 2.0301 | 12.9192 → 1.5962 | 6.58× |

Maximum six-QP changes are 8.05e-16 and 8.33e-16 Ha; all residual/positive-Z/
branch checks pass. EV convergence certifies its local coupled equation,
not enumeration of every satellite. Whole-process ru_maxrss including
preparation/compilation/repeats changes 3.634→1.790 GB for evGW0 and
3.874→1.816 GB for evGW. These are not per-iteration peaks.

| Fixed-input CPU1 kernel | Warm median before → after, ms | XLA temporary estimate before → after, MB |
|---|---:|---:|
| qsGW mode-A, 16 MOs/64 factors/nw=16 | 81.939 → 23.249 | 4.894 → 0.363 |
| scGW screening, 33 matrices of order 64 | 4.133 → 3.824 | 1.090 → 0.099 |
| Chemical-potential search, 2 orbitals | 5.843 → 1.673 | unknown |

These are kernels, not complete qsGW/scGW workflows. Separate full-output
comparisons give qsGW max difference 1.39e-17 and zero observed screening/
charge differences. Input hashes match the formal timed workloads. Instrumented
capture runs do not enter speed ratios. Screening timing ranges overlap;
its established benefit is smaller workspace, not demonstrated E2E speed.

Startup costs matter: qsGW compilation increases 1.125→2.244 s (first compiled
execution 96.95→44.04 ms). Charge first execution including internal JIT
increases 0.680→2.618 s. Warm improvements do not erase those cold costs.
Actual qsGW whole-process RSS rises 706.8→745.2 MB despite smaller XLA scratch;
screening RSS is about 610→608 MB. Required W outputs are unchanged.

## Completed molecular capacity cases

Phenanthrene is angular C14H10, PubChem CID 995, a fixed computed conformer,
not an ab-initio optimized geometry. Both orbital bases use the declared
weigend fit (974 auxiliary functions), every occupied/virtual G/W state,
six frontier QPs, nw=64, eta=0.001 Ha, root tolerance 1e-10 Ha.

| Candidate | AO / occ / vir | Execution | CPU RI/SCF, s | GW, s | E2E, s |
|---|---:|---|---:|---:|---:|
| Phenanthrene def2-SVP, CPU32 | 260 / 47 / 213 | first | 17.647 | 165.111 | 182.759 |
| Same, CPU32 | 260 / 47 / 213 | one warm | 12.005 | 163.054 | 175.062 |
| Same, CPU32→V100 | 260 / 47 / 213 | first | 16.916 | 15.328 | 32.785 |
| Same, CPU32→V100 | 260 / 47 / 213 | one warm | 11.671 | 8.928 | 20.704 |
| Phenanthrene def2-TZVP, CPU32→V100 | 564 / 47 / 517 | first only | 42.058 | 22.868 | 65.716 |

The ea646bf phenanthrene CPU first/warm E2E was 199.773/191.863 s; candidate
QP energies are unchanged. One warm sample has unknown dispersion. CPU/GPU
QPs agree within 1.56e-14 Ha; all six positive finite weights/branch checks
pass. No virtual, G or screening states were removed to fit memory.

RI/RHF stay on CPU, followed by explicit transfer and pure-array GPU GW.
The 564-AO run transfers 2.489 GB in 0.738 s. Its host peak RSS is 10.41 GB,
GPU peak bytes-in-use 9.94 GB; JAX pool/reserved is separately 25.55 GB on the
32-GB V100. The 260-AO GPU active peak is 2.79 GB. Allocator reserve is not
active tensor storage. The larger basis is a capacity point, not a converged
basis/experimental spectrum. General GPU geometry/NNAO training is not validated.

True conventional water/def2-SVP: 25 AOs, 5 occupied/20 virtual, all G/W,
four targets 3–6, nw=64, packed four-center data, no auxiliary basis. Three
warm E2E median is 0.4796 s (GW 0.3096 s); first E2E 6.5605 s, process peak
RSS 1.098 GB. Every QP check passes. No old nonspectral route existed, so no
conventional speed factor is invented.

## Derivatives, cache checks and failures

The broad affected suite recorded 129 passes. Final affected checks recorded
74 passes; conventional-final added 20 and facade-final 8 after their last
patches. These executions overlap and are not 231 unique tests. All final
acceptance/controller exits are zero; intentional red histories remain.

Coverage includes raw ordered factors, finite eta/own poles, empty/reordered
windows, mixed-dtype/budget rejection, JVP/VJP/HVP, fixed-frame outer response,
particle/moment/energy gates and changed-input/fresh-result equivalence.
Native H2/STO-3G QP geometry JVP/VJP includes moving basis, overlap/Pulay,
relaxed HF, W and QP response, with two reconverged FD steps. This is a
QP-energy derivative, not a complete GW molecular-force implementation.

Synthetic H2 radial native RI→implicit RHF→QP loss, log-exponent/contraction
FD and Adam update pass. This is not pretrained MACE accuracy, force-loss
mixed AD, or full moving-auxiliary RI geometry support.

Measured Python traces remain 1→1 for EV outer updates/new objects and 2→2
for changed same-shape qsGW inputs. Actual backend compile-event counts were
not collected and stay null. The initial combined process segfaulted in JAX
compilation after 30 tests (exit 139). The old water qsGW case passed in an
isolated CPU1 process before the stable-JIT fix; repeated tracing was separately
reproduced/fixed. Do not claim a proven root cause for that native crash.

## Remaining scope and delivery

Conventional qsGW/scGW/U-GW, conventional BSE export, cache-free/disk
conventional screening are not implemented. Existing factor/BSE consumers
remain. Full scGW Pi/W/time cache elimination and a complete common scGW
budget stay open in PLAN.md. No fresh PySCF/MOLGW timings or ratios were
produced this round; previous aligned water results are context only.
No periodic method was expanded.

Manifest/environment/source hashes record actual imports and loaded library.
CSV/JSONL preserve sample/range/precision/memory scopes and raw path/line/SHA.
Complete-output validation is in fixed_input_numerical_check.json; GPU/large
cases are in capacity_results.json. Raw logs/NPZ/XML remain locally under
ignored raw/ and remotely, not as large committed datasets.

Allocation 113525 was released and absent from squeue. Original workspace
HEAD stays 4d67479 with empty tracked diffs and the original untracked path
list. commands.txt gives smoke/rerun commands inside a new allocation.
