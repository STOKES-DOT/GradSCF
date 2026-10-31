# Hybrid/RSH KS references and complete molecular geometry response

Baseline f0c9b2b; branch perf/hybrid-gw-geometry-20261031 in the existing isolated
worktree. Preserve the previous branch and main workspace. Defaults: PBE0 and
original wb97xd, Cartesian real closed-shell molecules, float64. All numerical
work remote on one V100/NV5090, <=64 logical CPUs, 256 GiB, allocation <=48 h.
Per-calculation timeout <=6000 s; first validation/performance round <=7200 s.
No grid, orbital-space, XC recipe, or convergence relaxation for speed.

The main capacity/performance cases are angular phenanthrene C14H10,
def2-SVP (260 Cartesian AOs) and def2-TZVP (564 Cartesian AOs), grid level 1.
Keep every occupied and virtual orbital in G/W; G0W0 may report the same six
frontier QP targets as the preceding report. H2 is a response-composition gate,
not the large-system acceptance. Large qsGW/scGW require a preflight of their
actual resident/work arrays; a rejected oversized case is not a successful run.
Static screened calculations and zero-screening geometry AD are distinct
settings and must not be presented as identical mathematical speed comparisons.

- [x] HG-01: Establish exact source/environment and functional baselines.
  - [x] Freeze actual imports/native SHA and hardware/affinity; verify each live allocation.
  - [x] Verify actual PBE0/wb97xd RKS recipes and all four GW forward paths.
  - [x] Time phenanthrene/def2-SVP and def2-TZVP preparation/GW separately, with explicit repeat counts.
- [ ] HG-02: Optimize existing hybrid/RSH DFT geometry execution.
  - [x] Measure native J/K throughput, large SCF iteration counts and derivative compile/workspace costs.
  - [x] Connect existing direct optimizer through MolecularEnergy, propagating budget/grid/CHG.
  - [x] Verify full/direct values, relaxed gradients/HVP, FD and no full/LR ERI production (31 tests, H2).
  - [ ] Close the 564-AO derivative gate: grid-pair scan compiles in 13.14 s; full gradient/FD still pending.
- [x] HG-03: Reuse MolecularGW for hybrid/RSH and matrix-method geometry maps.
  - [x] Retain actual F-h-J subtraction for diagonal GW and proper matrix-method initialization.
  - [x] Handle packed/full/direct sources and distinguish exact-pair reference from native RI.
  - [x] Add qsGW/scGW result contracts without invalid BSE frame or inferred scGW QP poles.
- [ ] HG-04: Validate complete coordinate/parameter dependencies.
  - [x] Test PBE0/wb97xd small-molecule DFT gradients/HVP, including moving grids/LR/CHG.
  - [x] Test four GW methods with JIT JVP/VJP, translation and two reconverged FD steps (two XC each).
  - [x] Use the same declared integral approximation for value/AD/FD; reject unsupported RI geometry.
  - [ ] Complete phenanthrene/def2-TZVP nuclear gradient and its original FD gates.
- [ ] HG-05: Accept only measured local gains and preserve clean handoff.
  - [x] Matched native-kernel timing and final-direct initial-density experiment, actual RSS and QP gates.
  - [x] Repeat the closed RKS Fock comparison fully serially, with three warm samples and full-array gates.
  - [x] Narrow related regressions and existing custom/neural parameter protections.
  - [ ] Local focused commits/report, original workspace verification, release own resources.

- [x] HG-06: Include UKS/GKS execution in the same optimization round (user scope extension).
  - [x] Remove measured redundant spin J/K and unused GKS XC-potential preparation using public operators.
  - [x] Reuse existing UKS DIIS for implicit molecular response; preserve independent spin occupations.
  - [x] Check complex spin mixing, collinear/noncollinear supported XC, geometry/HVP and parameter AD.
  - [x] Measure bounded CPU/GPU before/after cases serially; retain only useful local changes and explicit capability limits.

- [x] HG-07: Balance libcint CPU work using CP2K's resource-management principles (user scope extension).
  - [x] Verify pinned HFX process/thread memory accounting and task-bin scheduling against current native calls.
  - [x] Measure requested/actual workers, private workspace and task imbalance; preserve the allocated CPU/BLAS limits.
  - [x] Implement only supported local scheduling/budget improvements, retaining deterministic reductions and full AD.
  - [x] Validate native/consumer values and first/second products, matched throughput/RSS, and a lower downstream response budget.

Current measured large baselines: phenanthrene/def2-SVP direct RKS on n002,
CPU32 (16 physical cores), cutoff=0, grid=1: PBE0 576.53698 s,
wb97xd 954.37833 s; both converged, whole-process peak RSS 4.585/4.588 GB.
These are one cold run each, not warm medians. The n006 follow-up uses a
different CPU and must establish its own matched before/after measurements.
Direct J/K quartet reuse passed 83 production tests and six diagnostic gates;
same-input warm medians improved 4.721->1.755 s and LR 5.092->1.724 s.
Phenanthrene/564AO PBE0 and wb97xd G0W0 both completed. RI-PBE0 initialization
reduced the same final-direct PBE0 reference from 114 to 35 SCF cycles, including
the seed cost. Large matrix-GW success is not inferred from these G0W0 runs.

The user's later DFT focus includes LDA/GGA/hybrid/RSH. SVWN and PBE use the
same 564-AO case/grid with explicit RI initialization and final direct J/K.
Pure packed J-only contracts passed 26 tests; native/DF/default-RKS pure paths
that already requested J-only were preserved. CP2K commit d077e66f has eight
verified source files; see DFT_SOURCE_COMPARISON.md. The original large
minimize derivative timed out; DIIS-only still compiled for 2831 s and then
raised a nonfinite derivative-vector error. These remain failed gates.

Four methods mean G0W0, evGW, qsGW and finite-T matrix scGW; evGW0 is an
additional useful fixed-W control. qs/sc KS orbitals initialize their full
self-consistency. scGW derivatives target its total energy/density/mu, not
nonexistent continued QP poles. Native RI currently rejects traced geometry;
small exact-pair geometry references must be labeled explicitly and never
silently substitute for an RI gradient. Native full-ERI geometry is CPU-only.
Existing direct supports first/second coordinates at static omega, not a new
arbitrary-order basis AD promise. Geometric targets: DFT/GW first gradients
atol 3e-6 Ha/bohr; DFT directional HVP atol 4e-6 Ha/bohr^2; full/direct
rearrangements much tighter (3e-8 gradients, 3e-7 HVP). QP agreement 1e-8 Ha,
original residual/convergence/branch/particle gates remain mandatory.

The 564-AO PBE0 gradient now returned all 72 finite components: lower 19.117 s,
compile 13.138 s, first diagnostic execution 1676.422 s, peak process RSS
5,750,697,984 B, reference energy error 1.93e-12 Ha. Its two original FD gates
remain unchecked until the freshly relaxed endpoints return. No completed
baseline gradient exists for a force speed ratio. Numerical source is frozen
at production commit 2cccb60; harness commit 2e1e330. Slurm 113825 completed;
113922 performs FD/Fock stages serially; 113932 is a dependent final narrow
CPU-only regression. Native-only worker activity was not resolved by the
single stack sample; low utilization alone did not establish a C-kernel
bottleneck. The subsequent uniform native-VJP diagnostic confirmed first-shell
load imbalance independently. User additions HG-06/07 expand the original scope.

The first 3e-4-bohr FD pair hit its 6000 s process limit without returning a
complete pair. The second was stopped after that failure to diagnose forward
work before another long repeat. Both original FD gates remain incomplete;
do not accept the large gradient solely because its entries are finite.
The replay harness now logs each completed endpoint immediately. This is a
separate instrumentation change; original run-grid.py/results remain frozen.

Latest integrated source is 5fad939 (plus separate thin harness/report work).
CPU allocation 114012 owns 64 logical CPUs / 32 physical cores on n006. The
formal Fock stage uses 32 distinct physical cores; the worker sweep pins all
64 logical CPUs explicitly. Do not call this 64 physical cores. Existing
max_memory now plans native private output work; it is a planning allowance,
not a process RSS hard limit. Its matched 564-AO prepared Fock median is
26.006910 s versus39.965757 s (three warm samples each), with E difference
9.09e-13 Ha and maximum Fock difference3.13e-13 Ha. The integrated 53-test
spin group and six UKS LDA/GGA/hybrid/RSH routing/geometry tests pass.
See CPU_RESOURCE_BALANCE.md and quota-fock-comparison.json. Large geometry
FD, molecular-VJP speed and projected/neural final regression remain open.

HG06/HG07 acceptance: all four matched spin kernels pass full-array and
requested AD gates. GPU energy-only warm time is unchanged; no GPU speedup
is claimed. The CPU coordinate-integral VJP now measures70.6143->22.1305s,
max full-product difference9.33e-12; average CPU-time/wall rises8.92->30.43.
This is fixed-density integral response, not a full molecular force.
Current-source/hash match verified for all16 modified production files.
Projected/neural regression and original564AOFD remain HG05/HG02/HG04 gates.

Final current-source parameter guards:24passed169.48s on114012, including
projected contraction parameters, native mixed force products and tiny
neural-XC explicit/implicit force-loss optimizer steps. These are bounded
regressions, not a pretrained NNAO training campaign.

Downstreamresponsequota a3631c2: actualRED2->GREEN2. The integration
passed64others; two oldhardcodedbudgetfailuresreproducedonfrozenparent,
then correctedresourceassumptions passaffected9tests. Full564AOgradient
nowreturnsin937.921s/RSS6.390GB, old/newmaxgradientdifference2.54e-11.
OriginalFD0.0003/0.0001 remainpending; finisher3372144resumed.
