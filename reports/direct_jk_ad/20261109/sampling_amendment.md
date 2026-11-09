# Bounded sampling amendment

On2026-11-09, before the final candidate process started, the controller
reported that8-CPU large-system calls exceeded2minutes; Slurm rejected a
90-to120minute time extension. Preserve all5 existing baseline warm samples,
use3 candidate warm samples (plus the same first compiled execution). Physical
inputs, math, cutoffs, worker counts, quota and complete J/K outputs are unchanged.
Only per-call samples/medians are compared; whole-process wall ratios are invalid.
No numerical or performance tolerance is changed.

Baseline runner SHA256:`fa6764b9341f22c03e889a31d70a259a409dcdbe6793cfbaa2f00fcae7774dd3`
Candidate runner SHA256:`5c03d992be6aaf79d38ad19205a8fd8f0eea9dcaf9b1a6df2e3c97d4695749ad`

The complete harness diff is sampling_amendment.patch.
