# NNAO Stage 2: GPU memory and SCF repair record

This file is an English translation of the imported source record, not a new run or result. Dates, status statements, and numerical values refer to that source record.

Record date: 2026-10-08. This record distinguishes local compilation analysis, remote revalidation of four systems, and the still-pending complete 200/50 validation.

## JAX GPU memory settings

The original stage-2 jobs all set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so startup GPU-memory preallocation was disabled. On-demand allocation was still constrained by the default BFC allocator limit.

Job 93522 ran three independent processes on n003, Tesla V100-SXM2-32GB, JAX/JAXlib 0.10.0, allocating only 2×2 float64 arrays:

| Configuration (preallocation disabled in all cases) | Allocator bytes_limit | GiB |
| --- | ---: | ---: |
| Default memory fraction | 25,554,419,712 | 23.80 |
| `XLA_CLIENT_MEM_FRACTION=.90` | 30,665,302,842 | 28.56 |
| `XLA_PYTHON_CLIENT_ALLOCATOR=platform` | memory_stats not provided | — |

Both the default and .90 processes reported `peak_pool_bytes` of 2 MiB, showing that the startup probes did not preallocate those limits. These probes are not peak-memory measurements of scientific calculations. Subsequent scientific jobs kept the default allocator, disabled preallocation, and explicitly set a 90% usage limit. JAX 0.10 uses `XLA_CLIENT_MEM_FRACTION`; the old alias `XLA_PYTHON_CLIENT_MEM_FRACTION` is not set simultaneously.

## DF backward-memory repair

Previously, `project_factors` saved the expanded primitive matrices of every auxiliary block during reverse AD. For MCONF/21, a single `(24,64,957,957)` cache occupied 10.48 GiB. The `fori_loop` step now uses `jax.checkpoint(step, prevent_cse=False)`, recomputing one block at a time during backward.

The mathematical contraction, dtype, auxiliary basis, and block size are unchanged. No full four-index ERI was added; JVP/VJP/HVP are retained. CPU JAX 0.8.1 abstract compilation analysis of the actual shapes: implicit-method arguments were 5.38 GiB in both cases; temporary memory decreased from 39.61 GiB to 11.62 GiB (a 70.7% reduction). This is a compilation estimate, not GPU peak memory.

## SCF rescue and implicit boundary

The initial hcore guess for S66/33 showed charge oscillation between the pyridine/ethene fragments. The optional `--scf-rescue-level-shift .2` was added: only if the original SCF fails to converge, retry with a 0.2 Ha level shift, then remove the shift and pass a consistent density, MO coefficients, and MO energies to the original SCF for polishing.

The original `h+J−K/2` Aufbau density fixed point and implicit backward remain unchanged; GMRES tolerance remains 1e-8. The final state must converge, with both the raw orbital residual and `||Phi_raw(D)-D||` ≤1e-7. No stationary Lagrangian or zero-gradient fallback was introduced. This check does not prove a global HF minimum or orbital-Hessian stability.

## GPU revalidation of the four failed systems

93522: V100 32GB, 16 CPU, 128 GiB host memory; RHF/DF, `szp663_direct`, core6, `def2-universal-jkfit`, float64, seed0, and mmap of the original cache. SCF max150, with energy/density/gradient thresholds 1e-12/1e-10/1e-9. Each system was checked for finite implicit gradients, fourth-order coefficient FD, MACE-parameter FD, and 5-step Adam (lr1e-4, clip1). FD step1e-4, atol2e-6, rtol2e-5.

| System | Coefficient FD absolute error | MACE FD absolute error | Energy decrease over 5 steps / mHa | Elapsed / s | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| MCONF/21 | 7.78e-9 | 4.53e-9 | 12.340075 | 150.32 | Passed |
| Amino20x4/ARG_xci | 7.68e-10 | 4.70e-9 | 14.302988 | 142.88 | Passed |
| MCONF/44 | 1.83e-9 | 1.04e-9 | 12.367247 | 135.02 | Passed |
| S66/33 | 1.28e-9 | 3.66e-9 | 4.630052 | 78.52 | Passed |

The three OOM cases used rescue parameter 0; S66/33 used .2. The final unshifted polish ran 2 times; raw fixed-point residual 4.37e-10, orbital residual 2.78e-10. Complete results are stored in `probe-93522/{large-results,scf-results}/`. Total numerical revalidation time for the four cases was 506.75s, excluding job startup and allocator probes.

All 60 related local control/numerical regressions passed; the DF repair also covered zero contraction coefficients, dual-input JVP, and HVP. The directional FD error for the new CPU S66 API was 7.17e-9. The original 250 fixed primitive DF caches did not need rebuilding; original failure records were retained.

## Restored job chain

- 93537: V1001, 32CPU, 128GiB, 48h. First use the final training rescue=.2 configuration to revalidate MCONF/21, then rerun complete validation for training200 and heldout50. Failure in any group prevents training from starting.
- 93538: A100_80G1, 32CPU, 128GiB, 48h, `afterok:93537`. Train from scratch for 1000epoch, Adam1e-4→1e-5, full-batch mean; heldout energy evaluation only every 20epoch.
- Original 93256 showed `CANCELLED / DependencyNeverSatisfied`, runtime 0, and produced no formal training result.

Passing these four systems does not replace complete validation of 250 systems or completion of the formal 1000epoch training. New job results are stored separately; the training tool checks consistency of source, structures, caches, and rescue configuration.

The final-configuration precheck of the largest system in 93537 passed: MCONF/21 coefficient FD, MACE-parameter FD, and 5-step Adam all succeeded. rescue=.2 was enabled, but the normal convergence path did not trigger rescue. Evidence is stored separately in `gate-93537/largest/`; complete 200/50 validation remained running, 93538 was still waiting on its dependency, and formal training was at 0epoch.

Scientific source SHA-256: `7f066083658982a552fda47e0144e762796f69788c6a1f1c7e70c77fbfec403a`. The four implementation files were checked individually and had matching local/remote hashes.
