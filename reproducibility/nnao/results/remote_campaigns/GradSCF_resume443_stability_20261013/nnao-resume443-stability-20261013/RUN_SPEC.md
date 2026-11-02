# Original 200-system training: stability repair and epoch-443 restart

Parent job: 96510. New job: 96984, A800 on Slurm, 32 CPUs, 256 GiB, 72 hours.
Remote checkout: `/home/user_yuanjiao/private/homefile/GradSCF_resume443_stability_20261013`.

The parent accepted epoch 443 with mean energy -383.8195142883713 Eh, then
rejected epoch 444 on ADIM6/AD6 because orbital stability was unresolved.
The accepted last/best checkpoints and the failed candidate remain in the
original checkout at `artifacts/nnao-fd-precision-20261013/training/`.

## Scope and numerical policy

Only `src/gradscf/scf/stability.py` changes relative to the parent's scientific
source. The native library is byte-identical. The original CHNOF 200-system
dataset, `szp663_direct` basis, core6, fixed exponents, seed, Adam learning
rate and implicit SCF implementation are retained. No Ge2H6 response or new
1000-system basis changes are included.

The original 40-vector/100-iteration Davidson stability check remains the
fast path. An unresolved eigensolve retries with 80 vectors/300 iterations,
at the same 1e-7 eigen-residual tolerance. Failed response/stability checks
continue to block updates.

## Required restart gates

1. Run DF-RHF stability and curvature regressions on CPU.
2. Replay the exact saved ADIM6/AD6 epoch-444 candidate with the parent and
   fixed stability implementations on the allocated GPU.
3. Require a stable SCF, finite full MACE gradient and a masked directional
   finite-difference check (step 1e-4, atol 2e-6, rtol 2e-5) with the fix.
4. Migrate accepted last/best epoch-443 checkpoints to the separate output
   directory. Only the source identity changes; parameters, gradient, encoded
   Adam state and history must retain their bytes/content.
5. Verify Adam counters are 443 and its next update reproduces the saved
   rejected epoch-444 parameter vector within 1e-14 absolute error.
6. Replay all 200 accepted energies and the batch gradient, then rerun the
   existing finite-difference gate before continuing epochs 444–1000.

All gates execute in the new Slurm job before training. Submission/running
status alone is not evidence that replay or epoch 444 has completed.

The waiting S-SZ job 96738 now depends on successful completion of 96984;
its existing nine-family validation guard is retained. Other DZ/TZ jobs are
outside this repair's scope.

Evidence files: `submitted.json`, `ad6-baseline.json`, `ad6-fixed.json`,
`migration.json`, `optimizer-check.json`, `restart-gates.json`, and
`training/progress.json` (generated as each stage completes).

## Initial verification results

Job 96984 is RUNNING on n004. Twelve CPU stability/curvature tests passed.
The exact rejected AD6 candidate passed both standalone baseline and repaired
replays; the original failure was not reproduced in the standalone baseline,
consistent with numerical-path sensitivity rather than proof of a negative
curvature. The repair retains the fast path and adds a checked retry when
needed; it does not relax the acceptance criterion.

The repaired candidate had eigen residual 5.10546e-8, finite MACE gradients,
and directional AD/FD values -0.00613985421083 / -0.00613985434939. Both
accepted epoch-443 checkpoints were migrated with unchanged parameters,
gradient, history and Adam bytes; Adam count is 443 and the next update
matches the saved epoch-444 candidate. Full 200-system replay is still
pending completion; epoch 444 is not yet claimed accepted.
