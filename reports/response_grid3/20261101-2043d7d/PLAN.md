# DFT -> TDDFT and DFT -> GW -> BSE execution optimization

Baseline 2043d7d; new worktree perf/bse-tddft-grid3-20261101. Physics only in src.
Four XC: SVWN/PBE/PBE0/original wB97X-D. Three KS: RKS/UKS/GKS.
Four GW: G0W0/evGW/qsGW/scGW; evGW0 only an optional control.
Largest closed shell: phenanthrene/Cartesian def2-TZVP/grid level3, all orbitals.
Open shell: same-geometry phenanthrene radical cation, charge+1/spin1; vertical benchmark.
All production paths must retain current custom/neural parameter dependencies.

- [ ] R1: Freeze exact source, physics and resource contracts.
  - [x] Create isolated managed worktree; preserve main user state; read live cluster manual.
  - [x] Define paired neutral/cation geometry and grid3/all-orbital inputs.
  - [ ] Record source-verified versus runtime-verified capability matrix for both chains.
  - [ ] Freeze run configurations, hashes, resource bounds and tolerances before timing.
- [ ] R2: Establish small and medium independent workflow baselines.
  - [ ] Run tiny smoke then water/OH def2-SVP grid3; all four XC across supported KS.
  - [ ] Run DFT->TDDFT TDA/full, five roots where space permits, original-operator residuals.
  - [ ] Run DFT->GW->BSE TDA/full using actual method outputs and all required QP levels.
  - [ ] Compare matched PySCF/MOLGW tasks or record exact method/capability mismatch.
- [ ] R3: Optimize at most three measured repeated-work or workspace bottlenecks.
  - [ ] Profile preparation/action/solver separately from clean timing; inspect actual batch/work ownership.
  - [ ] Check CP2K/PySCF source strategies against observed costs; avoid duplicating existing caches.
  - [ ] Make independent production patches only after narrow failing tests or equivalent measured gates.
  - [ ] Retain only demonstrated throughput, memory, correctness or maintenance improvements.
- [ ] R4: Verify live data and AD after each production change.
  - [ ] Same-shape geometry/orbital/model updates must change results and agree with fresh computation.
  - [ ] Exercise JVP/VJP and supported geometry/HVP; complete dependencies, no new stop_gradient.
  - [ ] Preserve nonsymmetric densities, independent X/Y, unequal occupied/virtual counts and spin blocks.
  - [ ] Preserve custom/neural XC, implicit and explicit existing response modes and device contracts.
- [ ] R5: Close bounded large-system coverage and report.
  - [ ] Preflight tensor ownership at phenanthrene and cation SVP/TZVP; grid3 never reduced for speed.
  - [ ] Run supported largest workflows serially; retain timeout/OOM/capability records without fake spectra.
  - [ ] Record cold plus three new-object warm timings, residuals, actual RSS and device metrics.
  - [ ] Review and commit locally; preserve source/log lineage and release own resources. No automatic merge/push.

## Registered protocol

CPU float64 common baseline; CPU8 (logical affinity recorded), BLAS/OpenMP1;
libcint workers follow case threads. GPU is a separate group with explicit CPU
integral/preparation and device transfers. Single allocation <=64 logical CPUs,
256GiB, one GPU; no heavy concurrent measurements. Each calculation <=6000s;
first validation round <=7200s, then inspect evidence before selecting large runs.
SCF energy/density/orbital criteria1e-12/1e-11/1e-10. TD/BSE root residual <=1e-7Ha;
TD energy reference tolerance1e-4eV, comparable GW/BSE1e-3eV. Rearrangements use
stricter same-input gates. Array AD transpose1e-9 relative/absolute; geometry
1e-5Ha/bohr (two FD steps). Do not loosen after seeing results.
Full ERIs only small reference; true auxiliary RI and conventional/direct are
labeled separately. BSE requires every QP level in its transition window;
old six-frontier-target G0W0 is not the present all-space BSE task.

## Initial source boundaries, awaiting small runtime gates

GKS spinor TDDFT/BSE and GKS RSH SCF are currently absent. UKS qsGW/scGW absent.
Matsubara scGW has no continued QP pole spectrum and is rejected for static BSE.
RKS qsGW does export a valid rotated-MO BSE reference; do not mark it absent.
Native RI nuclear-coordinate tracing remains unsupported. These are capability
boundaries, never substitute UKS for GKS or fabricate scGW quasiparticle energies.
These restrictions do not suppress independent work on supported combinations.
