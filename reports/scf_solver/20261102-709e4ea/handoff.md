# Shared SCF implementation handoff

Updated2026-11-04. Worktree `/Users/jiaoyuan/.codex/worktrees/1b2c/GradSCF`,
branch `perf/bse-tddft-grid3-20261101`. Implementation commits:
`28edfb22` shared solver/precision and `86e39f2b` optional atomic guess.
No merge/push. Main release HEAD2043d7d and its recorded dirty status unchanged.

## Scope and code

RHF/UHF/GHF and RKS/UKS/GKS use one `solvers/scf/iterate.py` numerical scan
and existing nonlinear DIIS. Physical R/U adapters live in `scf/iteration.py`.
No new stop_gradient; fixed_density/explicit/implicit training and the physical
implicit density response remain. RO algorithms retain their prior physics.

Convergence uses the occupation-weighted MO gradient, including fractional and
complex occupations. Guarded Cholesky orbital frame preserves the clipping
fallback and the separate symmetric DifferentiableSCF frame. Direct-JK fresh
certification remains; nonrefresh steps avoid duplicate assessment. Only scalar
histories are kept for ordinary forward use; explicit AD may still retain tape.

Optional atom guess: native isolated UHF density superposition, not spherical
SAD, bounded immutable LRU128, six named AE families H--Ar, native/raw-shell
cache key. Raw/trainable/ECP/ghost/unsupported families reject explicitly.
Default hcore and model paths unchanged. Performance ablations not executed.

## Verified evidence

Frozen precision candidate archive SHA256:
`91fdbb6eefaac9048fda3132f504a4d85da9ab3e9d555b4f9f00c0a14b986516`.
Linux native unchanged SHA256:
`ce51d885d4b65e753b5f56863470f1f1bb9c41d4b0762e47df5128e0d0b8feb8`.

CPU groups:99core84.36s,20atomic25.74s,11neural/force143.76s,
8NNAO64.26s,1tinyMACE20.25s,12neuralguards47.78s. Do not sum overlapping
iterations of these groups. TinyMACE is a random fixture, not a real trained
checkpoint. Earlier combined NN process aborted at compile; required cases
later passed in isolated groups, not a general compiler fix. GPU-only tests
stopped during CUDA/NUMA initialization; no GPU numerical acceptance.

Strict original phenanthrene/PBE/CartesianTZVP/grid3/trueweigendRI,564AO,
CPU64,n006, hcore,max200, energy/density/gradient1e-12/1e-11/1e-10:
- Cation baseline200nonconverged1299.569267s; candidate69converged468.279903s.
  No valid speed ratio against a failing target.
- Neutral baseline82converged493.779866s; candidate51converged256.654833s;
  energy difference3.61e-11Ha. Single cold observations, not repeated statistics.
- Whole-worker RSS remains about9.7GB; no overall memory reduction claim.

See `before_after.json`, `validation_summary.json` and report.md. Raw logs/JUnit
are local `raw/` (Gitignored) and authoritative remote results/.

## Nov4 correction

Formal matrix first exposed an old GHF/GKS eager Hermiticity gate, identical
at709e4ea: fixed100eps rejects Cl/TZVP finite hcore roundoff1.10e-13Ha at
matrix scale146.9Ha. The corrected gate uses100eps*max(1,maxabs(A)), leaves
arrays and AD unchanged and still rejects materially asymmetric/nonfinite data.
2 RED tests preceded the fix;23 GHF/GKS tests passed16.53s. All five initially
blocked molecular cases passed separate correction attempts.8 metadata tests
passed0.11s. Independent read-only production review found no actionable issue.

Remote `candidate-precision/` remains untouched. `candidate-roundoff-20261104/`
is identical in every src Python file except this gks.py correction.585 source
Python hashes match the local committed production tree exactly; manifest is
`source-roundoff-20261104.json`. Native binary is reused unchanged. Remaining
formal cases use corrected source, with runtime provenance recorded per case.

## Active resources and matrix

Read remote manual before any resource operation:
`~/private/homefile/gradscf-hybrid-gw-geometry-20261031/Delta-Cluster-Manual.md`.
All numerical work Slurm only on jy_gpu_2. Current allocation115757,n006,
NV5090x1/64logicalCPU/256GiB, start2026-11-04T14:15:33+08,
end2026-11-06T14:15:33+08. Old114687/114720 are no longer active. No GPU6 ban.
Scientific workers are serial; do not launch another controller on this output.

Remote root `~/private/homefile/gradscf-scf-solver-20261102/`.
Python `../gradscf-libcint-nnao-20261026/env-compare/bin/python` relative to root.
Python3.11/JAX0.10.0/NumPy2.4.4/PySCF2.13.0; native paths/hashes in runtime.

Frozen plan-v2:60systems,12/basis,5--150atoms,15non-nobleH--Ar elements;
SVWN/PBE/PBE0/CAM-B3LYP15primary systems each. Five ideal coordination inputs
and RAP144+2distantwaters150capacity cluster are unoptimized constructed inputs.
858workers/1380registeredcalls including selected cold+3warm controls; not all
60x5x4 combinations. Expected20R/RKS-open-shell NA and17GradGKS-RSH unsupported.
SmallR/U conventionalpacked, largerR/U trueweigendRI, RSHR/Udirect, Gpacked only.

Controller log `results/resume-matrix-115757.log`.
Controller `harness/frozen-20261104/campaign.py`, source candidate-roundoff;
worker `harness/frozen/runner-140f5dd1e3d1.py` immutable original worker.
Existing process records are reused, orphan logs never overwritten. Remaining
lease insufficient for registered per-worker bound means deferred, not timeout.
Final controller writes release.115757 so the hold job exits. It does not renew.

Original outputs: `results/final-matrix/`. Five correction attempts:
`results/roundoff-retries-20261104/`; original failures remain intact.
`--overlay` summary labels superseded runs. NFS close-to-open delays can make
immediate derived .run metadata temporarily stale; refresh from completed raw
process records without rerunning. `watch_summary.py` does metadata-only refresh
and summary every300s; log `results/summary-watch-115757.log`. It exits once the
allocation disappears. Its dated source/harness paths are specific to this run.

Initial43 runs overlapped some small diagnostics: conservatively excluded from
speed ratios in timing-exclusions-20261104.json, not scientific acceptance.
No diagnostic calculations now overlap clean matrix measurements.
Snapshot2026-11-04T14:37:33+08:91unique attempts,81scientifically accepted,
3unsupported,7nonconverged,35matched energy pairs;8systems have at least one
completed configuration, not all six methods. Static closing-summary-20261104/
may lag remote results/summary-20261104/. Closing matrix is NOT complete.

## Resume and remaining gates

1. Read scheduler and controller/watcher logs, then refresh/sync raw records.
   Never equate an SSH disconnect or old PID with scientific completion/failure.
2. Finish remaining frozen configurations. Preserve failure/resource statuses;
   do not loosen tolerances, change grid or physics to improve pass rates.
3. Analyze by method/basis/XC/approximation and final residuals; keep different
   states and different conventional execution strategies explicit. Ratios are
   GradSCF/PySCF only for accepted, unexcluded comparable runs.
4. Update final report/plan, release resources and make a final local evidence
   commit. No automatic merge/push. Optional atom performance probes are still
   unexecuted. Existing large PBE0 fine nuclear-FD gate and unfinished TDDFT/BSE
   workflows are separate and not closed by this SCF campaign.

Reproduction commands: benchmarks/probes/scf_solver/README.md.
Plan: reports/response_grid3/20261101-2043d7d/SCF_SOLVER_PLAN.md.
