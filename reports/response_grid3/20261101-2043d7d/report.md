# Grid-3 DFT response workflow campaign

This is an implementation/progress report, not completion of the full matrix.
Source baseline2043d7d; production change2b18b85. Both requested dataflows are
independent public workflows: DFT->TDDFT and DFT->GW->BSE. Physics remains in src.
No main merge or remote push has been performed.

## Inputs and scope

SVWN, PBE, PBE0 and original wB97X-D; RKS/UKS/GKS; G0W0/evGW/qsGW/scGW.
Cartesian phenanthrene/def2-TZVP/grid3 is the largest closed-shell target.
The paired open-shell target is the same-geometry phenanthrene radical cation,
charge+1, spin1. This is a vertical performance fixture, not an optimized cation
or experimental-spectrum reproduction. See OPEN_SHELL.md and cases/sources.json.
All G/W/optical orbital spaces and required QP levels are retained in large inputs.
True auxiliary RI and exact full/direct cases have explicit separate labels.

## Accepted first optimization

The original public UKS solve's optimized HLO retains joint spin XC Hessian
arithmetic inside the Davidson/action grid loops. Evidence is in
raw/uks-public-hlo/module_0337.jit_solve.cpu_after_optimizations.txt: the dynamic
entry includes AO/density/derivative arrays; while.190 reaches region_34.136,
region_35.133, region_43.131, region_44.106, the grid-loop body and
fused_computation.100 with nested jvp(jvp()) Hessian arithmetic. This is the
actual public solve, not a constant-folded fixed-reference action closure.

The new path prepares the known local LDA/GGA joint Hessian once per response,
using bounded full grid blocks and a real-length tail. Actions reuse slices.
There is no global numerical cache, no source mutation and no new stop_gradient.
Same-shape reference updates remain dynamic. Existing planner reservations
already cover two joint Hessians. Opaque/global/neural callbacks retain their
complete-field path; HF returns zero without preparing grid tensors.

Final-source water/UKS/PBE/Cartesian def2-SVP/grid3, CPU8, float64, five TDA roots,
three warm fresh-object calls, identical input/solver controls:

| Stage | Before median (s) | After median (s) |
|---|---:|---:|
| SCF | 1.690457 | 1.700986 |
| Response solve | 5.374807 | 2.139212 |
| Complete DFT->TDA including optical strengths | 7.081291 | 3.854959 |

The complete-flow reduction is45.6%; response reduction60.2%. Maximum excitation
energy difference is7.77e-16Ha. Original-operator residuals remain below1e-8Ha.
See final_comparison.json for unrounded raw samples. These figures are not a
prediction of the large-molecule speedup or of full TDDFT/BSE performance.

## AD and regression evidence

The repeated-preparation regression was RED before the change. An initial
candidate passed28 existing/narrow tests. The final LDA/GGA implementation
passed31 focused tests, including same-shape changing density/AO/derivative/
weights, JVP/VJP and two-step finite differences against an unprepared action.
Existing joint-spin neural/global-kernel tests also pass.

A combined group printed nine PASS nodes, then its final relaxed UKS/full-TDDFT
geometry case aborted during LLVM allocation under CPU64/64GiB. This was not
reported as a passed group. The exact case was isolated in fresh processes at
CPU8/128GiB: frozen parent passed in314.85s; final candidate passed in207.14s.
Both validate newly relaxed SCF density/orbitals, full/direct integrals, excitation
JVP/VJP and reconverged finite differences. Test-run costs are not clean benchmark
speed ratios. No numerical threshold was loosened.

An optional TPSS higher-order probe hit601.07s and40.66GB RSS, without complete
JUnit evidence. Its cause was not established. The new preparation strategy
therefore applies only to the requested LDA/GGA families; MGGA uses its original
block-local HVP. The unchanged slow probe and failure provenance are retained in
benchmarks/probes/response_grid3/mgga_higher_ad_probe.py as an explicit diagnostic.

## Baseline and comparison coverage

Water RKS grid3 TDA: SVWN/PBE/PBE0/wB97X-D cases pass. Water RKS PBE full TDDFT
and closed-shell UKS PBE TDA pass original operator checks. OH/UKS/PBE TDA passes.
OH full TDDFT returns finite values but does not converge within the registered
200 iterations. The five returned values lie above the existing1e-3Ha root
filter, so no conclusion about physical zero modes is made. Original dense
278-dimensional operator and independent PySCF diagnostics are prepared separately.

Independent PySCF water/PBE TDA RKS and UKS reproduce GradSCF excitation energies
to approximately4e-10eV. Actual grids have33698 vs33704 points; conventions and
padding remain recorded rather than inferred from a shared level number. The
PySCF runner has not yet supplied an independent original-operator certificate,
so no certified cross-engine speed ratio is published here.

H2/RKS/PBE G0W0, evGW and qsGW -> static BSE run through the real public chain.
Water/def2-SVP/PBE G0W0->BSE preparation diagnostics pass. Its first reference
preparation costs0.201s and two repeated exports0.00269/0.00225s. The single
workflow performs one MO transform. No new long-lived BSE cache was added.
All BSE forward rows are checked for valid screening, convergence/stability,
finite energies and original residuals. See BSE_CAPABILITIES.md.

GKS spinor response/GW adapters and Matsubara scGW->static BSE are not implemented.
Two actual public-API guard tests pass; they demonstrate rejection, not spectra.
Static Fock eigenvalues were not substituted for scGW QP poles. UKS qsGW/scGW
and conventional-GW BSE export remain explicit source-level capability limits.

## Large stage and resource state

Small true-RI prerequisites for RKS/UKS TD and inherited-RI GW->BSE pass. The
neutral phenanthrene TZVP/grid3 workflow has started; cation TD and neutral
all-QP G0W0->BSE follow serially. Each large case is one cold capacity run first,
not a completed cold+three-warm benchmark. Final dimensions and RSS must be
read from outputs, not from the static memory ledger.

V100/NV5090 were occupied; the previously authorized A800 fallback is on n005.
Allocation114465 was released when a metadata-only launcher used old system
Python3.6 and failed before numerical execution. Corrected control uses the
existing Python3.11 environment and allocation114480, one A800/64 logical CPUs/
256GiB. Cases have a6000s ceiling, remaining-lease bound and200GiB sampled RSS
guard; the controller releases its own allocation in finally. No local numerical
tests have been run. Root/controls, run-time states and unknowns are in STATE.json.

## Remaining work

The full requested matrix, all large workflows, GPU measurements, aligned MOLGW
checks and the OH full-response diagnosis remain incomplete. Further production
changes require measured evidence; the retained first patch is confined to one
response module. Current raw outputs remain in the isolated remote campaign and
locally under raw/ (excluded from Git).
