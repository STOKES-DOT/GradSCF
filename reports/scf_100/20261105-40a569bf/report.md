# SCF-100: accepted RKS AO-reuse result and open matrix gates

| Variant | Cold facade (s) | Warm median (s) | Accepted calls | Whole-worker peak RSS (decimal GB) |
|---|---:|---:|---:|---:|
| GradSCF parent `40a569bf` | 792.491512 | 750.493826 | 4/4 | 70.485 |
| GradSCF call-local AO reuse, exact-buffer v3 | 577.914945 | 535.544680 | 4/4 | 71.205 |
| PySCF certified continuation reference | 212.216112 | 211.686044 | 4/4 | 3.537 |

The matched beta-carotene RKS/PBE/RI-J comparison is complete: one cold call
and three fresh-object warm calls passed for both GradSCF variants. The warm
median fell by **28.64%**, with unchanged recorded energy, final gradient and
23 SCF cycles per call. Whole-worker peak RSS increased by **0.719 GB (1.02%)**.

The **implementation comparison is the two GradSCF rows**: candidate/parent
warm time is 0.713590; cold time decreased by 27.08%. The PySCF row matches the
physical task and accuracy but uses a different atomic guess, stopping workflow
and RI representation. It is not an implementation-only speedup denominator.
Sources: [parent](baseline_rks_pbe_ri.json), [candidate](candidate_rks_pbe_ri.json)
and [certified reference](reference_certified.json).

The subsequent preparation/XC diagnostics and local algebra experiments are
in [FOLLOWUP.md](FOLLOWUP.md). Their kernel timings are separate from the
accepted complete-workflow measurements above.

## Measurement scope

The fixed input is the existing C40H56 conformer: 96 atoms, 296 electrons,
Cartesian def2-SVP, PBE and level-3 quadrature. Measured dimensions are 880 AO,
148 occupied and 732 virtual orbitals, 3112 auxiliary functions and 1,113,968
grid points. These CPU float64 runs use 64 logical CPUs on jy_gpu_2/n006 under
Slurm, the existing Python 3.11 environment and the unchanged libcint library
SHA256 `ce51d885d4b65e753b5f56863470f1f1bb9c41d4b0762e47df5128e0d0b8feb8`.

Facade time includes integral/grid preparation and SCF. Parent warm samples
are 750.493826, 750.598513 and 750.109793 s (sample SD 0.257322 s); candidate
samples are 534.428310, 535.544680 and 537.126375 s (sample SD 1.355704 s).
Both record -1554.7036403036154 Ha and a final MO gradient norm of
3.9285116124375864e-8 Ha. No phase or iteration cost is inferred by subtracting
cold and warm totals. GradSCF RSS peaks cover the complete four-call workers,
sampled every 0.2 s; they are not per-call or isolated warm peaks. This result
establishes a runtime reduction, not a memory reduction.

The candidate's RKS/grid files match `1a575d18`; its unused UKS caller predates
the conservative second-spin exchange-workspace reservation. The exact tested
source and raw-evidence hashes are retained in
[ao_cache_candidate_source.json](ao_cache_candidate_source.json). This is not
a claim that every current-source method has been benchmarked.

## Production change and compatibility

The retained AO patch prepares bounded Gaussian blocks once per builtin SCF
call, then reuses exact AO/derivative planes through bounded gathers. Inputs,
SCF state and known operator workspace are charged before retention. The raw
provider and reference objects are unchanged; new physical inputs rebuild the
local reducer. Explicit streaming, insufficient budget and unsupported
operator/device contexts retain their original paths. Functional, grid,
orbitals, integral engine and numerical gates are unchanged. Details and the
separately charged macro experiment are in [AO_REUSE_PATCH.md](AO_REUSE_PATCH.md).

The earlier consumer work supports occupied-width UKS RI exchange, incremental
and freshly certified UKS direct products, and GHF/GKS spatial RI/direct
operators with all complex spin blocks retained. RSH auxiliary fitting is
explicitly **RI-J/direct-K** via `only_dfj=True`, not RI-JK. HF energy preparation
can omit quadrature; grid-consuming references restore it without rebuilding
spatial integrals or repeating SCF.

Fixed TDD, HF-grid compatibility, three neural training modes, force-loss and
reference-reuse evidence is retained in [validation_summary.json](validation_summary.json).
Original RED runs and fixture failures remain documented in
[TDD_AMENDMENTS.md](TDD_AMENDMENTS.md); assertions and tolerances were not relaxed.
The exact-buffer cache/budget/AD group passed 13 cases, including JVP/VJP,
dense HVP, two-step finite differences, changing inputs and ownership checks.

A further **8-case T4.2 compatibility group passed with zero skips** using a
complete `1a575d18` archive and the same native library: UKS/GKS native geometry
implicit/unrolled cases, methane NNAO direct/RI, native coefficient/exponent
products, and a small two-H2-geometry MACE update/checkpoint fixture. Test time
was 94.350 s; guarded wall time was 100.9668 s, with peak RSS 2,646,773,760 bytes.
Together with the earlier three-mode neural and force-loss gates, this closes
T4.2 within its previously supported scope. It does not establish all-method
large-system performance or new auxiliary-geometry AD. Raw evidence and exact
case names are in [nnao_geometry_validation.json](nnao_geometry_validation.json).
The MACE fixture is not a trained production model. Overlapping groups are not
summed into a new total.

## Independent-reference differences and retained failures

The [original PySCF single-kernel attempt](reference_original_single_kernel.json)
remains unaccepted. Its 24 main iterations passed the checks, but an extra
final diagonalization produced a 2.217132e-7 Ha gradient and revoked convergence;
this was not exhaustion of 200 main iterations. The certified variant resumes
the same object/current density only after that failure, retaining the initial
guess, thresholds, conv_check and total main-iteration budget. It records 28 main
cycles and two extra checks; each restarted kernel has a new local DIIS history.
The single logical-call timer includes the continuation.

The original observer called `DF.get_naoaux()` after timing, which could build
unneeded RI factors for metadata. Its whole-process time/RSS are contaminated;
the timed facade remains an unaccepted original measurement. The repaired
observer reads only existing dimensions/factor shape and leaves unknown rank
null. The certified reference energy differs from GradSCF by approximately
3.015e-10 Ha, with gradient 4.740868375259011e-8 Ha. These workflow/representation
differences remain explicit; no cross-engine implementation acceleration is claimed.

## Remaining acceptance

The original-source RKS/PBE conventional-direct baseline completed all four
calls with 22 cycles each: cold 795.437445 s and warm median 754.542832 s.
[baseline_rks_pbe_direct.json](baseline_rks_pbe_direct.json) retains the exact
samples and raw hashes. Candidate direct repetitions also passed 4/4, with
warm median 544.734456 s and peak RSS 37.304 GB versus the parent's 5.085 GB.
The independent conventional PySCF PBE attempt did not converge under the
registered energy-change gate; it is not an accepted timing denominator.
[CONVENTIONAL.md](CONVENTIONAL.md) records the complete HF/DFT follow-up,
the separately labelled RHF fresh-density reference, and direct J/K evidence.
The 30-configuration matrix per engine remains incomplete. The TDD contract is
[TDD_PLAN.md](TDD_PLAN.md); the full matrix and final delivery remain open.

The fixed-matrix continuation passed a metadata-only dry-run over all 60
registered inputs: two completed engine rows are skipped and 58 were not yet
attempted at that check. It waits for the direct-pair controller to finish,
then runs remaining rows serially in allocation 115927 with complete
`1a575d18` source. Each original time/RSS budget is retained; rows that cannot
fit the remaining lease are recorded as not run. Queueing is not scientific
completion. [commands.txt](commands.txt) and the benchmark README record replay.

Nine small native public-entry smoke workflows passed their recorded gates;
their CAM-B3LYP energies agree with the respective PySCF references within about
1.21e-11 Ha. Tiny PySCF references retained packed exact exchange, so their timings
are not comparable to direct-K ([native_smoke.json](native_smoke.json)). These
checks do not establish large-system acceptance for other methods. RI-J/direct-K
response export remains explicitly rejected; full native auxiliary-geometry AD
and noncollinear GGA are outside the capabilities established here.
