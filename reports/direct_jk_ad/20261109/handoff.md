# Direct J/K AD — accepted local delivery

Implementation worktree: `/Users/jiaoyuan/.codex/worktrees/5fbb/GradSCF`, branch
`perf/scf-100-grid3-20261105`. Base for this round: `96ad38e5`.
Production commits: `9c3c2180` (screened geometry), `d3d23408` (basis/mixed
products and shared schedule), `b9a38f7f` (consumer policy and UHF None-ERI fix).
The latest user instruction authorizes local merge only, without GitHub push.

## Accepted evidence

- Final CPU sweep: 107 passed, one UHF interface failure. After its fix, the
  exact failing case plus 16 direct/DF, six-facade and neural-XC regressions
  passed. Preserve the original failure; do not say the full sweep was rerun.
- Parameter group 86, partial-mask spherical group 4, CUDA-stage group 6,
  NNAO/synthetic training group 3 passed. Groups overlap.
- Diagnostics: 35 passed plus a missing-env skip, then the exact skipped case
  passed with the diagnostic library. One/four workers evaluate 55 quartets
  for eight densities in that fixture.
- 586 production source hashes match remote, local and committed files.
  Final native SHA256 starts `9f8da65e`; baseline starts `8a1f50dc`.
- 96-atom / 880-AO saved-density CPU8 forward: arrays identical. Warm medians
  141.3201 -> 132.0922 s (basis screening), 136.1782 -> 117.1250 s (density
  reference). First compiled calls and preparation are separate. Five before /
  three after repeats were amended before candidate execution. Node drift
  means no stable speedup claim; the 10% regression target is satisfied.
- Eight fixed-density AD/value products pass. They compare zero/positive
  screening within the new code, not old/new total SCF or full training times.

## Boundaries and artifacts

Geometry first/second, raw coefficient/exponent first, and mixed R-c/R-alpha
are supported. Pure/cross basis Hessians and third physical derivatives still
raise explicitly. Positive-cutoff response differentiates a locally fixed
primal mask. GPU integrals use CPU libcint callbacks. Neural-XC force regression
is distinct from full NNAO basis-force training; the latter is not newly claimed.
The public relaxed adapter remains real Cartesian restricted/unrestricted HF/KS.

All scientific execution/builds ran remotely. Main final job 118589 ended;
CPU-only diagnostic supplement 118703 completed. No active final allocations
remain. Builds passed with clock-skew warnings; no macOS native build was run.
The main workspace's unrelated dirty files and the implementation worktree's
older benchmark/report edits are preserved, not swept into this delivery.

Read `report.md`, `acceptance.json`, `manifest.json`, `results/` and `commands.md`.
Raw archive: `jy_gpu_2:/home/user_yuanjiao/private/homefile/gradscf-direct-jk-ad-20261109/`.
Local `raw/` is ignored; large NPZs remain remote. No new numerical runs are
needed merely to update delivery documentation. Older broad HFKS benchmark
gates are not relabelled complete by these two bounded XC/direct-AD rounds.
