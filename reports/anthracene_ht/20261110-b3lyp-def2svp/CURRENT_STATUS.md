# Current status — actual AD-HT running, no final spectrum yet

Live-verified **2026-11-12 11:09:47+08:00**. Job **122119** is running on
jy_gpu_2/n006 with one NV5090, 64 logical CPU and 256 GiB. The native geometry
AD chain explicitly runs on CPU; separate GPU array AD tests do not certify
GPU native geometry execution. No PySCF calculation was resumed.

The full restricted TDDFT facade now supports opt-in
`tda_gradient_mode="implicit_eigenvector"` through the existing structured RPA
solver. Native integrals, moving grid, relaxed SCF, full Fock blocks and X/Y
remain differentiable. The default eigenvalue-only mode remains unchanged.

The remote acceptance receipts contain 54 different CPU test identities and
two independent GPU array reruns. Complete HF/HeH+, PBE/H2 and B3LYP/H2
geometry JVP/VJP agree with reconverged signed finite differences: maximum gap
error `9.85e-9 Ha/bohr`, maximum dipole error `1.40e-9 e`.

The actual anthracene AD run began **10:58:55+08:00**. Its reference SCF converged
in three cycles / 17.006 s at `-539.1704884154773 Ha`, consistent with the saved
ground state. The shared VJP first forward is in progress; **0/32 Cartesian
components saved** at this check. Slurm sampled MaxRSS is `145118036 KiB`
(about 138.4 GiB), not a final process-group peak. The process RSS guard is
220 GiB. No completed AD molecular spectrum or speed claim is made.

After the shared forward, 32 scalar pullbacks supply eight excitation-gap and
24 signed dipole gradients, projected onto all 66 validated normal modes.
Four old signed FD directions validate the result; they never supply the AD
model derivatives. Translation checks and finite pullbacks are mandatory.
The spectrum step runs only after all derivative gates succeed.

The old FD job119234 ended **2026-11-12 04:03:45+08:00** when the RSS monitor's
`/proc` glob raised ESRCH. Its 39 points (19 complete mode pairs plus one
positive point) remain validation archives. The sampler race is reproduced
and fixed. No FD production restart was launched. Ground geometry, all 72
Hessian columns and all 66 positive modes are reused without recomputation.

Implementation worktree: `/Users/jiaoyuan/.codex/worktrees/bd0a/GradSCF`.
Remote: `/home/user_yuanjiao/private/homefile/gradscf-ad-ht-20261112`.
Actual AD output: `ad_eight_states/raw`.
Acceptance and remaining TODOs:
[fixed plan](../../anthracene_ad_ht/20261112-9b087fad/PLAN.md) and
[implementation report](../../anthracene_ad_ht/20261112-9b087fad/implementation_report.md).
