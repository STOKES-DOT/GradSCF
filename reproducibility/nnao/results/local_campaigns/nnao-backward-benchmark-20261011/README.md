# Single-system NNAO backward and epoch benchmark

**Historical comparison, retired after acceptance.** Old-library switching is
no longer a maintained workflow. Its exact launch script is archived as
`archived-original.sbatch.txt`; result data and checkpoints are retained.
The maintained benchmark runs only the current native backend. The old production
snapshot is preserved while its training process remains active.

## Protocol

- System: GMTKN55 `S22/19a`, benzene C6H6, 12 atoms, neutral singlet.
- Basis: `szp663_direct`, six-primitive fixed core, 84 spherical AOs;
  auxiliary basis `def2-universal-jkfit`, 558 auxiliary AOs.
- Initial state: the frozen `nnao-joint200-20261011/initial.npz` used to start
  production training, retaining all learned MACE coefficients and 14 shared
  exponent scales. Each configuration resets Adam identically.
- Method: the existing stable DF-RHF solver and strict implicit SCF derivative,
  implicit tolerance 1e-8; SCF energy/density/orbital tolerances
  1e-12/1e-10/1e-9, with the existing rescue/stability checks enabled.
- Adam rates: MACE 1e-5, log exponents 1e-3; log-exponent bound 0.25.
- Configurations: old native library / 1 thread, optimized / 8 threads,
  optimized / 16 threads. Identical Python sources, inputs, GPU and allocation.
- Host/GPU work: native integrals and basis pullback on CPU; MACE and implicit
  SCF on GPU. JAX float64, GPU preallocation disabled, BLAS/OpenMP threads 1.

## Timing boundaries

The profile first solves the primal density using the production rescue and
stability routines. It then differentiates the identical implicit density fixed
point and energy expression with that root supplied as the solution, keeping
primal SCF iteration tapes out of the separately timed adjoint. Full backward is
the sum of synchronized implicit SCF pullback,
physical-integral pullback (including GPU-to-CPU transfer, DF whitening and basis
normalization), and MACE pullback (including transfers and gradient retrieval).
The separately profiled gradient is checked against the production combined
`value_and_grad` entry point before training timings are accepted.

Epoch timings come from the unchanged production `run_training` loop. With
exactly one molecule, an epoch means one Adam update followed by one complete
energy/gradient evaluation of the updated parameters, validation of SCF/gradient
diagnostics, and accepted checkpoint/history writes. Its existing timer ends
before the final progress JSON write and print. This is not a 200-system epoch.

First-call compilation and model/metric setup are recorded separately. Profile
timings use three warm repeats at identical initial parameters; epoch timings use
three actual sequential updates, with the same trajectory checked across all
configurations. The optimizer update is compiled using a discarded candidate
before training; accepted state is initialized afresh.

## Execution and evidence

Production task 95681 is not modified. The benchmark uses a separate release
directory. Each binary configuration runs in a new process, and persistent JIT
cache paths are isolated by output directory and native-library SHA256.

The first A800 request (95880) remained pending because n005 had all GPUs allocated;
it was cancelled without computation. V100 attempt 95881 stopped at the isolated
MACE import guard before training. The launch path was corrected to include
`src/gradscf/model/nnao`. Attempt 95882 applied `jax.vjp` outside the whole jitted
SCF value function; this exposed large linearization residual outputs (compiler
estimate about 29 GiB) and OOMed on the 32-GiB V100 before updates. This profiling
failure is not a production-training memory measurement. Task 95883 uses the
compact split described above on n003 with one V100, 32 allocated CPUs, 64 GiB
RAM and a 30-minute limit; together with production this totals 64 CPUs.

Numerical gates compare initial gradients, final gradients, final parameters
and all three epoch energies. Initial checkpoint/source/library hashes and
complete SCF diagnostics are retained in each configuration's result files.

## Results

Task **95883 completed successfully (ExitCode 0:0)** in 5 min 24 s, from
2026-10-11 20:36:55 to 20:42:19 cluster time. Hardware was a Tesla
V100-SXM2-32GB and Intel Xeon Gold 6230 CPUs; JAX 0.10.0, float64. The final
comparison passed remotely and again on the downloaded artifacts locally.

Times below are medians of three warm backward measurements or three real
training epochs. First-call compilation and setup are excluded.

| Configuration | Full backward (s) | Backward speedup | Epoch (s) | Epoch speedup |
| --- | ---: | ---: | ---: | ---: |
| Old native library, 1 thread | 6.2194 | 1.00x | 7.1127 | 1.00x |
| Optimized native, 8 threads | 1.5139 | 4.11x | 2.4208 | 2.94x |
| Optimized native, 16 threads | 1.2345 | 5.04x | 2.1878 | 3.25x |

Component medians (their sum need not equal the median of each full sample):

| Component | Old (s) | 8 threads (s) | 16 threads (s) |
| --- | ---: | ---: | ---: |
| Forward, including SCF | 0.8676 | 0.9030 | 0.9184 |
| Implicit SCF pullback | 0.01137 | 0.01082 | 0.01091 |
| Physical-integral pullback + transfer | 6.2064 | 1.5016 | 1.2163 |
| MACE pullback + gradient retrieval | 0.00185 | 0.00164 | 0.00165 |

Epoch elapsed samples were old `[7.1928, 7.0589, 7.1127]` s, 8-thread
`[2.4208, 2.4078, 2.4509]` s, and 16-thread `[2.1878, 2.1947, 2.1280]` s.
Moving from 8 to 16 threads reduced median epoch time by 9.6%; the serial
forward and other unchanged work limit the end-to-end gain.

### Numerical agreement

- Maximum initial production-gradient difference versus old library: 6.66e-11.
- Maximum final parameter difference after three updates: 3.51e-12.
- Maximum final gradient difference: 2.76e-11.
- Maximum epoch-energy difference: 2.28e-13 Hartree.
- Maximum split-profile versus production-fused gradient difference: 1.09e-10.
- All configurations retained converged, internally stable SCF solutions.

All three trajectories start at approximately -230.559498187727 Hartree and
reach -230.560378424464 Hartree after three epochs: a decrease of 0.880237 mHa.

### Interpretation and limits

16 threads was fastest among the tested configurations for this molecule;
8 threads captured most of the benefit with half the native worker limit.
This supports a production rollout trial, but does not establish the speedup
of the 200-system dataset. The single-system run retains a warm JIT working set
and does not reproduce repeated cross-molecule cache eviction/reload costs.
Thread limits remain per FFI call, and GPU/CPU hardware differs from the ongoing
A100 production allocation.

No production job, model checkpoint, or training code was changed. Task 95681
was still RUNNING at the final check; its native library SHA256 remained
`dd87d8f9113f8e982a76d5f276cdcc5fb6182f9c86fe6430cd265f8d2e5ff584`.

Evidence: `comparison.json`, `job-95883.log`, each configuration's `result.json`,
`metadata.json`, `production-diagnostics.json`, `initial-gradient.npz`, and
`training/{history.json,last.npz}`. The benchmark driver is
`tools/benchmark_nnao_backward.py`; the original Slurm script is archived as
`archived-original.sbatch.txt`.
