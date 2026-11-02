# ADIM6/AD6 epoch 42 failure investigation

## Preserved original failure

Stage-two job 93538 accepted epoch 41, then stopped during epoch 42 at
`ADIM6/AD6`, after 190 successful training structures. The exception originates
in the coefficient-gradient finite check in `MethaneRHF.evaluate`, after its
forward energy, convergence, overlap and raw density fixed-point checks passed.
The MACE parameter VJP had not yet run for this structure.

`original/` preserves the failed progress/summary, environment metadata and the
accepted checkpoint. Checkpoint SHA-256:
`7e7f66fd900fe50cf9925f6d875fd566a953eb5a67600d67c53c518254084df2`.
Scientific source SHA-256:
`7f066083658982a552fda47e0144e762796f69788c6a1f1c7e70c77fbfec403a`.
No production source, integral cache or accepted checkpoint was changed.

## Exact candidate reconstruction

Each probe restores the saved epoch-41 parameters, full-batch gradient, Adam
moments and cosine-schedule count, then calls the original optimizer update.
The epoch-42 candidate vectors reconstructed on V100 and A100 are bitwise
identical. Controls remain RHF/DF, `szp663_direct`, core6,
`def2-universal-jkfit`, float64, seed0, mmap caches, SCF rescue=.2 Ha, and
implicit tolerance=1e-8, max_iter=100 restart cycles, restart=40.

ADIM6/AD6 is C12H28: 40 atoms, 232 contracted AOs, 876 primitive AOs and
1,404 auxiliary functions. Its raw packed DF cache has shape (1404,384126).
No complete four-index ERI or dense density-response Jacobian is constructed.

## Completed independent probes

| Job | Hardware | Original backward | Observer |
| --- | --- | --- | --- |
| 94793 | V100, n002 | epoch41 and two epoch42 calls passed | initial observer failed because CUDA-only platforms cannot execute host debug callbacks |
| 94794 | A100 80GB, n001 | epoch41 and two epoch42 calls passed | same diagnostic configuration error |
| 94796 | A100 80GB, n001 | epoch41 and 30 epoch42 calls passed | epoch41 and 30 epoch42 calls passed after enabling CPU only for callbacks |

Each job requested 32 CPU cores, 128 GiB host memory and at most 20 minutes.
Probe 94796 keeps the uninstrumented baseline in a CUDA-only process. The
separate observer calls the original residual check once and preserves its
returned norm, validity flag, numerical solution and acceptance tolerance.
Instrumented results are distinct from the original execution graph.

The epoch-42 energy is approximately -470.306424838597 Ha, lower than
epoch41 (-470.305604400861 Ha). SCF takes 18 cycles; minimum overlap eigenvalue
is about .09023 and the raw fixed-point residual about 8.95e-10.

Observed true relative linear residuals range from 1.3575e-9 to 8.7716e-9.
The largest absolute residual is 4.29918e-7 for RHS norm 49.0124; the relative
acceptance term is 4.90124e-7, with an additional production roundoff allowance.
All observed numerical solutions and guards pass. This supports investigating
limited numerical acceptance margin; it does **not** establish the original
failure's hidden adjoint residual, which was not recorded.

## Context probe methods

`diagnose_tail.py` runs the original consecutive evaluators
`MCONF/21 -> Amino20x4/LYS_xas -> BSR36/r16 -> ADIM6/AD6`, then repeats AD6.
It verifies checkpoint/source/structure/cache hashes and the exact reconstructed
candidate. It records the actual prediction only after the original evaluator
returns, without an extra prediction call. If evaluation fails, original
traceback frames supply energy, forward diagnostics, gradient and predicted
outputs; they are saved without rerunning or replacing the failed observation.

94804 is the completed A100 context comparison (up to 100 baseline / 30
observed AD6 calls, stopping each mode at the first failure). Pending V100
alternative 94806 was replaced before running by A800 94808 (up to 50 baseline /
20 observed calls), to use an earlier available allocation. These tests did not
restart training; original 93538's hidden residual is distinct from replay data.

## Reproduced failure and direct cause (2026-10-09)

The consecutive-context probe reproduced the original exception without
instrumenting the production JAX backward graph. On A100, AD6 call 17 failed;
on A800, call 45 failed. Both saved original traceback arrays: the energy and
predicted coefficients are finite, SCF converges in 18 cycles, and all 404
active coefficient-gradient entries become NaN (the 316 padding entries remain
finite). The overlap minimum is .09023 and raw fixed-point residual ~8.95e-10.

The separate observer reproduced the same failure with the actual production
linear guard returning false:

| Hardware / job | AD6 failing observer call | True relative residual | Candidate finite | Final guard |
| --- | ---: | ---: | --- | --- |
| A100 / 94804 | 19 | 1.0814070684709093e-8 | yes | rejected |
| A800 / 94808 | 4 | 1.0006696486704647e-8 | yes | rejected |

Both RHS and numerical adjoint norms are approximately 49.0124; primal
convergence is true. The guard rejects the true residual and intentionally
replaces the adjoint with NaNs, which then propagate to coefficient gradients.
Thus the reproduced failure is a numerical response acceptance failure, rather
than a nonfinite forward energy or an invalid overlap matrix. Original 93538's
hidden adjoint residual was not saved; these numbers belong to the replays.

The solver's upstream target and the final certification both use 1e-8. No
separate numerical safety margin is requested. At fixed weights the 44
successful A800 AD6 predictions have 44 distinct byte hashes, but their maximum
component variation is only 2.7755575615628914e-17. This measured last-bit
variation accompanies residual variability; the specific CUDA reduction that
causes it has not been isolated, and this measurement alone is not a causal
attribution to MACE's segment reductions.

The installed JAX 0.10 GMRES recomputes true residuals at restart boundaries;
its `info` flag does not certify tolerance acceptance. It is inaccurate to
describe the algorithm as checking only the inner estimated residual. GradSCF's
independent guard remains necessary.

Job 94825 is the completed single-variable counter-test: only upstream GMRES targets are
multiplied by .1 (1e-9), for both incremental and batched paths. Final acceptance
remains 1e-8, with the same physical operator, regularization, cache, parameters,
SCF criteria and dtype. This variant is confined to the independent diagnostic
process; no production source or original training identity is rewritten.

## Counter-test completed (2026-10-09 19:27, Shanghai time)

94825 finished with exit code 0 on A100 80GB, n001, 32 CPU cores and 128 GiB
host memory, within its 20-minute allocation. The three preceding structures
passed in both processes. AD6 passed 100/100 uninstrumented full training
evaluations and 30/30 instrumented evaluations. No NaN/Inf occurred.

The observed AD6 true relative residual range is 9.4014e-10 to 4.8798e-9,
strictly within the unchanged 1e-8 final acceptance. Requesting internal 1e-9
does not certify an external 1e-9 residual; the final measured residual and
unchanged independent guard remain the actual acceptance evidence.

The maximum energy difference from the original valid epoch42 evaluation is
1.1369e-12 Ha. Within the 100-call counter-test, maximum parameter-gradient
component variation is 4.8261e-13. Median warm AD6 evaluation time is 1.5294 s.
These are local replay measurements, not a whole-dataset throughput benchmark,
finite-difference gradient audit or completed 1000-epoch training run.

`verification-summary.json` collects the checked counts and metrics; raw data
and logs remain in `job-94825/` and `context-94825.out`. A production change
should keep the NaN guard and final tolerance, introduce internal numerical
margin, and be separately validated before migrating/resuming the epoch41
checkpoint. This investigation has not modified production modules.

## Separate resume-startup issue

See `resume-startup/README.md` and its shell evidence. The cluster Conda wrapper
can recurse in a child shell that inherits an activation flag but not the real
Conda function, producing exit 139. This was reproduced independently. Original
94688's exit code is unavailable, so its correspondence remains conditional on
the inherited submission environment. Diagnostic jobs use the verified absolute
environment Python and install an early exit trap; shared modules/environments
are untouched.
