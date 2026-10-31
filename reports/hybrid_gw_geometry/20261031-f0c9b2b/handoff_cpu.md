# CPU and spin execution handoff

Worktree: `/Users/jiaoyuan/Documents/GitHub/GradSCF/artifacts/side/g0w0-memory-recovery-20261030`.
Branch: `perf/hybrid-gw-geometry-20261031`. The main release workspace and
its pre-existing changes were preserved. Local merge was authorized and performed
on 2026-11-01; no push was requested.

Accepted production commits:

* `4943d0b`: cyclic canonical shell-pair scheduling for coordinate pullbacks.
* `22387db`: memory-driven native private accumulation; new FFI requires rebuild.
* `5fad939`: spin-batched exchange, demand-driven GKS XC and complex stationary response.
* `a3631c2`: rebind the downstream response quota without mutating its reference.

Harness commits `d7eefc6` and `4a1b314` retain frozen-runner hashes, full-array
comparators, endpoint logs, bounded execution and automatic resource release.
The final 19 production/build files match the remote candidate exactly;
see `integrated_source_verified.json`. Native library SHA256 is
`ce51d885d4b65e753b5f56863470f1f1bb9c41d4b0762e47df5128e0d0b8feb8`.

The resource is Slurm 114012, n006, one NV5090, 64 logical CPUs on 32 physical
cores, 256 GiB, four-hour lease. Numerical tests are remote only. CPU Fock/VJP
measurements pin 32 distinct physical cores; the worker sweep uses all 64
logical CPUs. BLAS/OpenMP thread counts are one. No nested native thread pool.

Accepted matched measurements:

| Scope | Before median, s | After median, s | Numerical gate |
|---|---:|---:|---|
| 564-AO PBE0 prepared Fock | 39.965757 | 26.006910 | full Fock max error 3.13e-13 Ha |
| 260-AO fixed-density coordinate integral VJP | 70.614273 | 22.130468 | all coordinate entries, max error 9.33e-12 |
| Synthetic s8 GKS/PBE0 energy+Fock, 90 AO, CPU | 0.124503 | 0.082783 | full outputs identical |

The first two have three warm repetitions; spin kernels have ten. These are
distinct scopes and cannot be multiplied into a whole-method speedup. The
GPU energy-only probe has no demonstrated warm benefit. Native heap memory,
XLA temporary estimates and whole-process RSS are reported separately.

Validation: native group 88 passed/6 diagnostic skips; spin group 53 passed;
UKS four-XC/routing group 6 passed; parameter guards 24 passed, including
fixed-primitive projected coefficients and tiny neural-XC force training in
explicit/implicit modes. The response integration passed 64 tests plus two
old budget fixtures that also fail on the frozen parent. Correcting their
invalid resource assumptions yields 9/9 affected tests passing. Counts overlap.
This is no pretrained large NNAO training campaign or new direct exponent AD.

The 564-AO current-source zero-screening SCF converged in 13 cycles with
residual 9.52e-11 under the original thresholds. Its complete first gradient
returned 72 finite components in 937.921 s, energy error 2.39e-12 Ha, peak
RSS 6,390,472,704 bytes. Old/new gradients differ by 2.54e-11 Ha/bohr.
**Both original reconverged FD steps remain required before accepting it.**
The older 1676-s parent includes a stack sample and a different certified seed;
`large_gradient_comparison.json` deliberately has no clean speed ratio.

Final state checked on2026-11-01: the3e-4 bohr pair passed with error
5.5066e-8 Ha/bohr. The1e-4 pair timed out after2035.165s under the remaining
lease. Finisher exit124 and automatic resource release are verified.
Allocation114012 is no longer running. The second-step large-gradient gate
remains open; do not present this merge as full large-force acceptance.
See `large_geometry_gate.json`.

Details: `CPU_RESOURCE_BALANCE.md`, `cpu_environment.json`, `before_after.csv`,
`cpu-comparison-114012.json`, `spin-comparison-114012.json`, and the raw records
under local `raw/` (excluded from Git). The complete remote campaign remains at
`jy_gpu_2:~/private/homefile/gradscf-hybrid-gw-geometry-20261031/`.
Commands are in `commands_cpu.txt`; scripts live in
`benchmarks/probes/hybrid_gw_geometry/`. Do not rerun old output paths.

Remaining capability limits: GKS direct/RSH/MGGA and noncollinear GGA were not
added; native RI geometry is unsupported; large matrix qsGW/scGW memory/work
limits remain open. CPU work and the spin changes do not certify these methods.

## Local integration verified on 2026-11-01

Merged into `release/v1.0.0` as `b4b26c5`, preserving the main branch's iKS
changes. The exact merged-source archive passed 81 remote CPU tests, with
no failures or skips, in 366.74 s (Slurm 114180, CPU8, 32 GiB, no GPU).
The unchanged native source tree permitted reuse of the verified Linux
library. Actual import/library paths and the archive hash were checked.

The local arm64 extension was rebuilt against the existing Conda JAX 0.8.1
headers, all 51 exports verified, then atomically installed with a backup.
Its dynamic-loader check passed without calling an integral. No local
numerical test was run. Existing Python sessions must restart to load the
new FFI interface. See `merge_verification.json` and `merge_preservation.json`.
The second-step large-molecule FD limitation remains open. No push was made.
