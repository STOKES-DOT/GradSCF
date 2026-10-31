# Hybrid-reference molecular GW and geometry response

Current CPU/spin acceptance is summarized in [CPU_RESOURCE_BALANCE.md](CPU_RESOURCE_BALANCE.md).
It supersedes older pending statements below for native budgeting, coordinate
VJP scheduling and UKS/GKS regression. Local production commits4943d0b,
22387db and5fad939 passed current-source remote gates; d7eefc6 preserves
the thin reproduction harness. This report still does not certify the
original564-AO reconverged FD gates or large matrix-GW capacity.


This is an in-progress report, not final acceptance of large qsGW/scGW.
The starting source is f0c9b2b2a783d70c747ee91ce42a308e71363cad in branch
`perf/hybrid-gw-geometry-20261031`. Main release and the preceding GW-family
branch are preserved. All numerical runs use isolated remote sources.

## Physical and resource scope

Use PBE0 and the original omegaB97X-D (`wb97xd`, including its CHG dispersion),
real Cartesian closed-shell AOs, float64. H2/STO-3G/grid 0 checks response
composition. Main capacity cases are angular phenanthrene C14H10 with
def2-SVP (260 AOs) and def2-TZVP (564 AOs), grid level 1. G and W use every
occupied and virtual orbital. Selected frontier G0W0 outputs do not imply a
truncated screening space. Matrix scGW has density/energy/mu outputs, not
continued quasiparticle poles.

The initial n002 allocation 113608 used one V100, 32 logical CPUs and 256 GiB
and ended at its two-hour limit. Extension was denied by Slurm; successor
113629 depends on its completion and uses one NV5090, 64 logical CPUs and
256 GiB. The maximum calculation timeout is 6000 s. n002 CPU32 means 16
physical Xeon Gold 6230 cores. The n006 baseline is measured anew; timings
from these different CPUs cannot serve as a matched speed ratio.

## Accepted DFT response change

`MolecularEnergy` now connects supported DFT references to the existing
moving direct-J/K orbital optimizer. It inherits the actual XC/grid/memory
controls, forces zero screening for geometry AD, and carries the live basis
provider into the solver. Original omegaB97X-D retains moving LR exchange,
grids, overlap/Pulay response and CHG dispersion. The change introduces no
new integral engine, solver or physical approximation.

Remote command:

```sh
PYTHONPATH=candidate/src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
GRADSCF_NATIVE_THREADS=1 python -m pytest -q \
candidate/tests/test_molecular_hybrid_direct.py \
candidate/tests/test_scf_orbital_direct.py
```

Result: **31 passed in 511.06 s**, exit 0, n002. Tests include PBE0/wb97xd
full/direct value, density, Fock, complete geometry gradient, directional HVP,
two finite-difference steps, live configuration/grid dependencies and rejection
of full/LR ERI assembly. Local/remote SHA256 of the three accepted files agree.
This is small-molecule derivative acceptance, not a large-molecule force result.

## Measured large DFT baseline

Independent direct RKS, Cartesian phenanthrene/def2-SVP, grid level 1,
32 logical CPUs on n002, cutoff 0; energy/density/orbital thresholds
1e-12/1e-11/1e-10, max_cycle 200. These are single cold runs including
preparation/compilation, not warm medians.

| XC | Energy (Ha) | Total (s) | Whole-process peak RSS (bytes) | Converged |
|---|---:|---:|---:|---|
| PBE0 | -538.5455310349147 | 576.53698 | 4584751104 | yes |
| wb97xd | -538.9927870868077 | 954.37833 | 4587528192 | yes |

The library SHA256 is
`c4756801f7cad8aa826760fb5d761ba7ea6f345606c17402401cc8f4ac6b384b`.
Static screened runs, if added, are a separately labeled numerical setting;
they cannot silently replace this zero-screening baseline.

## GW response status

The committed `MolecularGW` patch allows actual restricted KS references,
subtracts the actual F-h-J potential for diagonal GW, and connects existing
qsGW/scGW implicit maps with their respective output contracts. Its geometry
path explicitly uses the small, traceable exact AO-pair factor representation;
it is **not auxiliary-basis RI geometry AD**. Large fixed-geometry RI and this
small geometry reference are separate calculations.

Ten constructor/window/output/tolerance gates passed in 15.04 s. Each of the
four methods then passed both PBE0 and wb97xd complete nuclear-response tests:
G0W0 387.47 s, evGW 408.58 s, qsGW 516.62 s and scGW 594.61 s (two tests per
process). These include JIT JVP/VJP, translation and two reconverged FD steps.
The earlier combined 850 s timeout remains a failed process; it is not used
as the evidence for these later individual passes. Eight old HF GW/BSE
response tests passed in 305.33 s, and 39 existing scGW tests in 1183.23 s.

Canonical finite-temperature chemical-potential refinement adds two bounded
AD-Newton steps after the original particle-number gate. This corrects a
reproducible charge-tolerance versus chemical-potential-accuracy distinction;
it does not tighten or relax the requested physical model. Dense small-case
response diagnostics found production versus dense mu JVP difference
4.64e-11, with condition number 1.70e5 and a residual 9.83e-11 still permitting
a 3.69e-7 mu correction. Forward iteration, not an incorrect implicit inverse,
was responsible for the FD error in that case. At fixed beta=20/nw=24,
946 iterations reach the original stricter 1e-12 fixed-point gate. The scGW
geometry tests use an explicit cap of 1600 and retain the two FD thresholds.

## Pending large-system boundaries

Source-only preflight: at 564 MOs, the current scGW reference screening uses
158766 orbital pairs; one square pair matrix alone needs 201.653 GB, and one
reference self-energy coupling array needs 404.023 GB. Streaming other
intermediates does not remove those owners. Large scGW must not be blindly
launched against the 256 GiB allocation. Large qsGW also has 134467 distinct
same-occupation retarded endpoint differences per outer iteration. These are
dimension/work estimates, not measured memory peaks or successful calculations.

The direct-J/K throughput investigation uses a separate fixed synthetic
density probe on the real phenanthrene basis. It does not stand in for an
independent SCF run. Call-count diagnostics and clean timing use separate
libraries. Any candidate is retained only after numerical/AD and matched
performance checks.

## Accepted bounded direct-J/K candidate

Independent diagnostic build: 6 expected failures before the change, solely
because 4 workers evaluated 654 shell blocks versus 266 for one worker.
Full/serial/parallel values passed before the count assertion. The new
quartet partition passes all 6 cases (JK, J-only, K-only; Cartesian/spherical),
with 266 evaluations for both worker counts. Fresh production and diagnostic
builds both exit 0. Build logs retain an NFS clock-skew warning and an upstream
XLA header return-type warning; the new libraries are separately hashed.

Production regression: **83 passed, 6 skipped in 23.00 s**. The skipped nodes
are precisely the separately executed diagnostic counter cases. Coverage includes
non-symmetric batches, large-batch row ownership, reproducibility, simultaneous
calls, operator selection, screening, density/coordinate JVP/VJP, mixed HVP,
long-range operators and the shared memory budget. No AD primitive was replaced.

The production change adds 33 net lines. Native batches up to 4 densities use
worker-private outputs only when at least two fit within 64 MiB. The actual
worker count is bounded by this cap; other calls retain disjoint output rows.
Output reduction follows fixed worker order. The Python workspace estimate
reserves the same cap from shell metadata. At 564 AO, batch-1 JK uses at most
13 private workers (66,163,968 B extra), instead of allocating 32 such arrays.
This is a workspace size calculation, not an RSS measurement.

Matched **n006 CPU32** fixed-density phenanthrene/def2-SVP probe, float64,
cutoff=0, same synthetic PSD density hash and ordinary/LR input, one first
execution plus 3 warm executions per point:

| Operator | Before warm median (s) | After warm median (s) | After/before | Max abs J/K error |
|---|---:|---:|---:|---:|
| Coulomb JK | 4.721355 | 1.754737 | 0.371660 | 9.24e-13 |
| erf(0.2 r12)/r12 JK | 5.091799 | 1.723579 | 0.338501 | 7.32e-13 |

Whole-process ru_maxrss is 780,263,424 B before and 789,614,592 B after.
These processes include imports, compilation and multiple operator/worker
points. The difference is not an isolated live-buffer measurement. Compiler
time is recorded per point in the raw JSONL; no GPU arithmetic is included in
this CPU integral probe. Full arrays were compared, not only norms. See
`jk_comparison.json` for every warm sample and relative errors.

This result does not establish a 2.7–3x SCF/GW speedup. The current large
calculation is an independent phenanthrene/def2-TZVP direct RKS → weigend RI →
NV5090 G0W0 workflow, nw=64, all 47 occupied/517 virtual G/W orbitals, six
frontier QP targets. Static direct screening is explicitly 1e-13; zero-screening
geometry AD remains a different configuration. Both hybrid/RSH runs completed;
the large first nuclear derivative has not yet passed, as detailed below.

## Four-family phenanthrene/def2-TZVP results

Common case: 564 Cartesian AOs, grid level 1, 97,240 points, n006 CPU32,
energy/density/orbital thresholds 1e-12/1e-11/1e-10. Final direct screening
is explicitly 1e-13. SVWN/PBE and the seeded PBE0 experiment include an
independent RI starting-density calculation followed by the original final
direct equations. The seed is a nested part of the reference time. No virtual
orbitals are removed. All entries below passed their forward convergence gate.

| XC / initial density | Direct cycles | Cold reference (s) | New-object warm reference (s) | Total energy (Ha) |
|---|---:|---:|---:|---:|
| SVWN / RI-SVWN | 57 | 843.338791 | 883.486146 | -534.8101845197906 |
| PBE / RI-PBE | 59 | 941.644279 | 943.140717 | -539.0293925209918 |
| PBE0 / original | 114 | 3432.555708 | not measured | -539.0781454737033 |
| PBE0 / RI-PBE0 | 35 | 1140.008336 | 1134.021994 | -539.0781454737527 |
| wb97xd / original | 66 | 2631.957424 | not measured | -539.5311959548168 |

These are one cold and, where available, one warm sample, not three-repeat
medians. They do not establish an XC-family speed ratio: iteration counts,
initialization and required operators differ. The PBE0 initial-density
experiment retains the same final target and differs by 4.94e-11 Ha. Including
the seed, its cold reference time is 0.3321 of the original run. The source
change responsible for native quartet reuse is common to both of these
PBE0 rows; these two gains cannot be multiplied.

For G0W0, CPU weigend RI preparation is 9.978346 s (PBE0) and 9.907378 s
(wb97xd). Explicitly staged NV5090 GW times are 125.739326 s and 66.318211 s;
complete geometry-to-QP workflows are 3582.938822 s and 2722.556378 s. All
47 occupied and 517 virtual orbitals remain in G/W, nw=64; only outputs are
restricted to targets 44--49. Maximum QP residuals are 9.03e-17 and 5.56e-17
Ha, with valid selected branches and positive weights. The seeded PBE0 QPs
differ by at most 1.71e-12 Ha. Whole-process peak RSS is 13.24/13.34 GB in
the unseeded workflows, and 22.28 GB in the seeded one. GPU allocator peak
bytes-in-use is 9,943,922,432 B; reserved pool is a distinct 25,259,999,232 B.
These GPU counters must not be called tensor RSS or whole-process memory.

## Large geometry failure and bounded diagnosis

Geometry AD always uses zero direct screening. The original orbital-minimize
path lowered for 168.72 s and compiled for 3103.55 s, with XLA temporary
estimate 29,423,349,216 B; it reached 6000 s before a first gradient.
The DIIS-only path lowered for 141.00 s and compiled for 2831.35 s, temporary
estimate 28,016,590,448 B, then raised `Derivative vector must be finite` in
the native geometry boundary. Neither run produced an accepted force or FD
result. A successful reference state or compile is not gradient acceptance.

A shared saved-reference primal diagnostic found the saved RKS orbital
residual 3.08e-11 but zero-step response validation residual 9.41e-10, above
the unchanged 1e-10 gate. The re-evaluated energy differs by only 6.14e-12 Ha.
Holding the Fock fixed while repeatedly normalizing the orbitals preserves
the residual near 3.1e-11: simple coordinate normalization alone is therefore
not the explanation. The next diagnostic separately recomputes the physical
Fock at the original and transported densities. The native finite-vector
guard remains in place; this evidence does not implicate an incorrect native
integral derivative.

The isolated Brillouin candidate replaces differentiation of the entire
Cayley energy chart by the real body-frame variational residual, only for
checked, unregularized real stationary states. Clipped/opaque callbacks keep
scalar-density AD; certified classic native direct Fock can be reused without
another density gradient. The accepted Brillouin/variational change is
committed as fb0f84d: 24 complete response tests passed in 1026.89 s, the
17 focused tests (see saved JUnit for run-specific timing), and four basis/force-loss mixed-derivative tests
in 157.10 s. The subsequent closed RKS candidate passed 25 focused tests
(109.28 s) and ten complete geometry/projected-contraction tests (1133.73 s).
Large nuclear-force acceptance remains open. Counts refer to separate
scopes and versions, with overlapping fixtures; they are not additive.

For real spin densities `D_s=C_s N_s C_s.T` and an antisymmetric increment
`K[p,q]=theta[p,q]`, `K[q,p]=-theta[p,q]`, the orbital tangent is
`delta D_s=C_s (K N_s-N_s K) C_s.T`. Contracting it with the variational
`F_s=dE/dD_s` gives
`g[p,q]=2 sum_s (n[s,q]-n[s,p]) (C_s.T F_s C_s)[p,q]` for shared orbitals.
Independent spins retain separate packed blocks. The reported norm is half
the norm of this real parameter gradient, as in the existing API. A smooth
change of tangent frame multiplies this residual by an invertible matrix;
at a stationary root, the terms differentiating that frame multiply zero.
Its local implicit response is therefore unchanged. This argument does not
license substituting a clipped, nonvariational reported potential, nor changing
unchecked or regularized response policies. Those paths retain their scalar
energy derivative and original chart. First/second parameter and metric tests
check the implementation rather than relying on the algebra alone.

Follow-up source-aligned diagnostic: at the same original density without any
normalization, fresh RKS also produces residual 9.31733e-10 (43.29 s), matching
the spin-response value 9.31724e-10 (106.63 s). These are single shared-input
diagnostic calls, not clean repeated speed ratios. Crucially, the saved
reference used cutoff 1e-13 while the re-evaluation uses zero cutoff. This
diagnostic cannot establish an incremental-J/K defect or an XC inconsistency.
A zero-cutoff reconvergence is required before testing its response gate.
At that early stage the speculative reference-centered metric change was
not retained. Its small RED tests remain under `experiments/not_retained/`
as investigation history. The later strict-checkpoint experiment isolated
a separate normalization effect and led to commit 69c8ac7; see the local
stationary-response section below for the final implementation.

## Reference-code findings

See `DFT_SOURCE_COMPARISON.md` and `cp2k_source_manifest.json`. Eight selected
CP2K files from pinned commit d077e66fe06a862d49bdcc2b91c179463e9487eb were
verified by Git blob ID and SHA256 after the bulk checkout failed. No CP2K
production dependency or alternative integral backend was added. Relevant
strategies are selected J/K, memory-accounted integral lifecycle, grid/AO
blocking, incremental density updates and explicit RI preparation. GPW and
ADMM change the representation or approximation; they cannot silently replace
the benchmark target. Dense LDA AO derivative ownership remains source-only
optimization potential; streamed LDA already requests derivative order zero.

## Matched closed-shell Fock optimization

Same saved phenanthrene/564AO PBE0 density, zero integral cutoff, same n006
CPU affinity 133--164, float64, prepared dense grid input. Independent
processes call the compiled production response Fock callback, one first
execution and three warm calls each. This excludes SCF, grid preparation
and nuclear AD. All physical inputs are dynamic executable arguments; each
warm invocation executes native J/K, rather than returning a converged cache.

| Metric | Parent spin-batched callback | Closed RKS callback |
|---|---:|---:|
| Warm samples (s) | 107.045697, 103.862216, 104.481372 | 40.842140, 40.671006, 40.642933 |
| Warm median (s) | 104.481372 | 40.671006 |
| Lowering wall (s) | 1.509638 | 1.555655 |
| Compile wall (s) | 1.512135 | 1.377253 |
| XLA temporary estimate (bytes) | 3,079,663,184 | 2,196,302,024 |
| Whole-process peak ru_maxrss (bytes) | 9,575,153,664 | 9,569,067,008 |

Warm after/before ratio is 0.389266. Full-array validation passed:
identical input densities, max Fock error 4.35652e-13 Ha and total energy
error 2.27374e-12 Ha. Predeclared gates are in `fock_acceptance.json`;
full records in `raw/phi-fock-comparison.json`. Peak process RSS is effectively
unchanged; the lower compiler temporary estimate is a different metric.
Compile wall time is observed in these processes; persistent-cache hit state
was not independently logged, so no cold-compile speed claim is made.

The scalar closed-shell density removes duplicated spin contractions and
uses the existing RKS builder. It also permits 13 private native JK workers
under the same 64 MiB cap, versus six for the duplicate spin batch. Integral
shell generation within a batch was already shared, so this is not a claim
of halving the number of shell quartets. Open/independent spins and opaque
providers retain their prior contracts; pure XC still requests only J.

The candidate initially constructed an unused potential on energy-only
calls. A five-family scalar-energy guard was added; one HF node reproduced
the extra work (RED 5.18 s). The final condition applies RKS reuse only when
a potential is requested, preserving the original scalar energy AD graph.
The same 25 focused tests passed in 108.23 s. The Fock benchmark requests
potential=True, whose numerical program is unchanged by this guard. No
end-to-end 2.57x speedup is inferred from the Fock-only result.

## Strict convergence versus incremental JK roundoff

At an independently zero-screen reconverged density, saved SCF residual is
3.32143e-11. Fresh RKS evaluation at that identical density gives
1.09478e-10, max AO Fock difference 6.59917e-13 Ha; both use zero cutoff.
This isolates an accumulated-Fock versus rebuilt-Fock distinction, separate
from the earlier screened-reference comparison. Subsequent metric transport
can shift the response validation residual further (observed 1.40090e-10).
A process returning exit 0 for this diagnostic does not pass stationarity.

A local RKS candidate now refreshes J/K when the incremental convergence
predicate first succeeds, checks the same energy/density/orbital/finite-value
conditions, and continues the existing DIIS loop if the rebuilt values fail.
It neither refills orbitals nor relaxes tolerances. Native providers reached
through the full facade request the same check; full/DF and custom scalar
SCF loops retain their existing behavior. Five RED tests gave four failures
and one existing pass; all 25 targeted convergence/DIIS tests then passed in
15.58 s. The subsequent combined real-molecule/geometry acceptance passed
ten complete gradient/HVP/projected-basis cases in 1174.93 s.

The final large preflight uses the original 97,240 grid points in blocks of
4096 and strict 1e-12/1e-11/1e-10 SCF gates. Blocking changes evaluation
layout, not quadrature precision. No large force is claimed before its
stationarity, complete gradient and both original FD checks pass.

## Acceptance update and CPU topology correction

Commits c934d74 and 19660d5 are local only. Combined final source passed all
ten complete geometry/HVP/projected-basis tests in 1174.93 s. The strict
large zero-screen SCF also completed: 25 cycles, 1145.453 s, energy
-539.0781454736988 Ha and residual 9.94653e-11. The 1200 s process timeout
occurred during the subsequent response-primal stage. The complete SCF
checkpoint was recovered with explicit stage lineage; this is not a completed
force workflow. The actual sole-child peak ru_maxrss was 3,535,446,016 B.

A live SMT topology check found that the old allocation's 133--164 and
325--356 are sibling threads of 32 physical cores. The small validation
worker shared one physical core with the Fock benchmark. Therefore the
matched Fock numbers above are preliminary timing evidence, with valid full
numerical comparisons, rather than a fully isolated clean timing. A fully
serial repeat is required before final timing acceptance. No end-to-end speed
claim is made from this preliminary ratio. New allocation 113825 retains the
same resource ceiling; the prior allocation was released after its steps ended.


## Local stationary response and compilation boundary

The checked zero-iteration interface now preserves the supplied numerical SCF
origin rather than transporting it twice. For a verified real origin C0/S0,
the local chart uses

\[
G(R)=I+C_0^T[S(R)-S_0]C_0,\qquad
C(R,\theta)=C_0\operatorname{chol}(G(R))^{-T}\operatorname{cay}(K(\theta)).
\]

At the origin G=I and the numerical C0 is unchanged. The current S(R) stays
live, with dG=C0^T(dS)C0, retaining Pulay terms. The input's existing 1e-8
metric-orthonormality contract is checked also under JIT. This is a local
chart about a numerical stationary state; it does not introduce a new
physical frozen-metric approximation. Positive-iteration, complex GKS,
regularized and unchecked policies keep their original charts.

The causal large-case test is now explicit: at the same strict checkpoint,
original orbitals give Brillouin norm 9.88854e-11; one numerical normalization
gives 1.03909e-10, with density change 2.83107e-14 and Fock change
8.16014e-13 Ha. Keeping the origin passes the unchanged 1e-10 gate and agrees
with the strict SCF energy to 2.27374e-12 Ha. The previous "metric candidate
not implemented" note describes the earlier investigation, before this
same-point evidence was available.

Let Ecal(R,theta) be the physical energy in this moving-metric chart and
F=partial_theta Ecal. At a nonsingular stationary state,

\[
\frac{d\theta^*}{dR}=-F_\theta^{-1}F_R,\qquad
\frac{dE}{dR}=\partial_R\mathcal E,
\]
\[
\frac{d^2E}{dR^2}=\mathcal E_{RR}
-\mathcal E_{R\theta}\mathcal E_{\theta\theta}^{-1}
 \mathcal E_{\theta R}.
\]

The first derivative uses the stationary envelope with metric transport;
higher response retains the shared implicit linear solve. A supplied numerical
SCF solution is the origin of that rule, not a frozen-orbital derivative.
The new candidate `forward="reference"` with dynamic `solution` explicitly
separates completed numerical SCF from local response compilation. It checks
the current stationarity and metric, and rejects stale solutions. Finite
differences must solve new SCFs at displaced geometries; evaluating a stale
reference there is not the defined energy curve. Default algorithms and all
three training modes are unchanged.

Candidate acceptance so far: PBE0 and original wb97xd, nonzero-field H2,
complete gradient/HVP versus full SCF, two reconverged FD steps, dynamic
solution/field updates with one executable, and stale-solution rejection all
passed (three nodes, 371.72 s). One unrestricted H2+ empty-beta-spin test
passed in 20.35 s, including a nonzero field and two reconverged FD steps.
This establishes the bounded local-response contract. Large-gradient acceptance remains a separate gate.

## Compiler and metadata diagnoses are separate

A read-only nonblocking py-spy 0.4.1 sample of the owned gradient PID shows
`backend_compile_and_load -> compile()` in JAX. The earlier probe emitted
lowering/compile timing together only after compilation. Therefore an absent
compile row did not mean that lowering was still active. The joint
SCF-plus-gradient experiment was deliberately stopped while the backend
compile was unfinished, after 2882.047 s process wall time, with actual
sole-child peak RSS 63,612,715,008 B. It produced no accepted gradient and
cannot be a valid speed-ratio denominator. Its manual-stop reason, original
records and stack sample are preserved.

The follow-up reference-response experiment is a controlled compilation-boundary
candidate: same converged SCF, physical equations, 564 AOs, all orbitals,
97,240 grid points, block size 4096, zero screening and original tolerances.
Lowering completed in 137.399 s; backend compilation was still unfinished
after about 19 minutes and this owned process was deliberately stopped.
The compilation-boundary change alone did not solve the large graph problem.
It produced no force and supplies no completed speed-ratio denominator.

Separately, shared nested Jaxpr metadata was repeatedly scanned by the memory
estimator. Per-estimate memoization keeps every call-site byte charge and all
opaque scratch multiplicities unchanged. The new tests gave 3 RED failures
and one existing pass, then 29 metadata/budget regressions passed. On a real
nested JIT graph of depth 16, all three estimates return exactly 9,437,112 B:
old times 0.420934/0.419389/0.419383 s; new times
0.0001174/0.00008892/0.00008866 s. This is a metadata-only microbenchmark;
it does not establish any DFT or XLA backend speedup. No physical data are
cached and no memory bound is relaxed.


## Moving Becke grid program: exact scheduling change

The remaining backend compile was traced to the geometry-dependent Becke
partition builder. The traced path expanded the atom-pair updates in Python
for each atomic grid. With 24 atoms it copied 24 * 276 * 2 scatter updates
into the program. This is graph size, not an assertion that all logical
intermediates are simultaneously allocated. The fixed-coordinate NumPy path
did not have this compilation issue.

The production fix stages the same row-major lower-triangular pair sequence
with `jax.lax.scan`. Each update multiplies the same two atom weights in the
same order. Grid level, radial/angular points, pruning, radius adjustment,
normalization and dtype are unchanged. The scan differentiates through every
weight and coordinate; no physical input is stopped or cached.

A four-atom graph regression failed before the change (48 copied scatter
operations, above the declared linear bound). Afterward, 26 grid regressions
passed in 31.495 s on n006: NumPy/PySCF value alignment, empty pair sequence,
mixed atoms, JVP/VJP transpose identity, directional gradients and HVPs with
two FD steps. The tested production SHA256 is
`21cbde9e02ef14d71a8eed7174acd13624dfa7439a2ba69beca8338ea7b9f23a`.

For the unchanged 564-AO PBE0 zero-screening local geometry response, lowering
completed in 19.117219 s and backend compilation in 13.138209 s. XLA reports
3,419,016,864 temporary bytes. Execution of all 72 nuclear derivatives is
still in progress; compiler completion is not a force or FD acceptance.
The former unfinished compile cannot be used to quote an end-to-end speedup.

Local commits `c1e5f1e` (grid scheduling) and `2cccb60` (supplied stationary
solution interface) are separate. Their remote sources and local files have
matching SHA256. Neither was merged or pushed.

A serial successor Slurm job, 113922, depends on completion of 113825 and
requests the same one NV5090, 64 logical CPUs and 256 GiB, for four hours.
It consumes the accepted gradient only if present, runs fresh-SCF FD pairs
at 3e-4 and 1e-4 bohr with frozen sources, then performs the fully serial
Fock comparison. Each individual process retains its 6000 s timeout.
This dependency prevents the campaign from holding two GPUs concurrently.


## Execution-tail observation (diagnostic, not a throughput conclusion)

Two read-only `/proc/<pid>/task/*/stat` samples, 2.006681 s apart during
the first large derivative execution, found two active XLA worker threads
with 2.00 and 1.99 CPU seconds. This establishes low utilization at that
sample, not which numerical kernel caused it. A single bounded py-spy native
stack dump took 3.329971 s and showed the main Python thread awaiting
completed JAX buffers. It did not expose the native-only worker frames, so
it does not establish that direct-integral VJP dominates. No additional
profiler was installed and no second heavy calculation was launched.

The execution timing of this first large acceptance run includes that probe
and is labeled diagnostic. It remains a valid numerical derivative test;
it is not used as a clean throughput benchmark or speed-ratio denominator.
Native coordinate VJP first-shell partitioning and the staged grid transpose
are still distinguishable source hypotheses until an actual phase profile
or controlled operator probe resolves the worker activity.


## Completed large first derivative, pending displaced-SCF acceptance

The 564-AO PBE0 derivative process returned exit 0, with all 24 by 3 entries
finite. Its energy is -539.0781454736968 Ha, differing from the strict saved
reference by 1.9326762412674725e-12 Ha. The sums of gradient components are
[1.7260e-12, -1.5645e-11, -1.0432e-14] Ha/bohr; this is supplementary
translation evidence, rather than a substitute for directional FD.

First execution took 1676.421649 s; whole process wall was 1712.628931 s
and actual sole-child peak RSS 5,750,697,984 B. These are separate from the
compiler's 3,419,016,864-byte temporary estimate. Execution includes the one
3.329971 s diagnostic stack sample, so it is not labeled a clean benchmark.
There is no completed old gradient for a matched force speed ratio.

Slurm 113825 completed after its only numerical step. Job 113922 then
started on n006 with one NV5090, 64 logical CPUs and 256 GiB. The FD
processes explicitly use CPU32 (32 physical cores at affinity 133--164),
full orbitals, unchanged grid level 1/block 4096, zero cutoff and the same
source/library/checkpoint hashes. The first FD pair is in progress. Each
endpoint relaxes a new SCF; no supplied solution is reused off its stationary
geometry. The second pair and serial Fock comparison follow automatically.

Final narrow regression job 113932 requests only one CPU, 16 GiB and no
GPU, and depends on completion of the timing job. Its 30 planned nodes
cover supplied-solution response, projected contractions in both AD modes,
force-loss mixed derivatives, variational/clipped Fock policies and the
updated memory fixture. The force-loss fixture uses a synthetic hcore
parameter with real moving native integrals through the neural-XC training
helper. It is not a pretrained neural-model or NNAO optimizer campaign.
