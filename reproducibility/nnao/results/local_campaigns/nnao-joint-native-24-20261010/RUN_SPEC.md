# 24-structure joint variational continuation — 2026-10-10

Continue the original shared MACE checkpoint at epoch 975 (7,171 network
parameters, SHA256 `ba93fb40efca3268962c7501f82627e188ed9509b6a73bd27a2df2170a3495de`).
Use exactly the original 24 GMTKN55 pilot geometries, charge/spin and order;
input SHA256 `cddae74070ee5053656c1e105be45e0b8dbd98ab15ec347d0fc87e4233405899`.
This is training-set optimization, not held-out generalization.

## Objective and parameters

Minimize the unweighted mean converged DF-RHF total energy across all 24
structures. One epoch means one full-batch gradient and one joint Adam update,
followed by validation of the updated parameters on all 24 structures.
Reference energies are used only to verify checkpoint parity at initialization.

- MACE directly predicts contraction coefficients using the unchanged
  `szp663_direct` model and element set H/C/N/O/F.
- Add 14 shared log-exponent scales: H s/p and C/N/O/F s/p/d.
  `alpha = alpha_initial * exp(beta)`; relative primitive ratios remain fixed.
- Keep core6, geometry and auxiliary basis `def2-universal-jkfit` fixed.
- Adam: MACE learning rate `1e-5`, exponent learning rate `1e-3`, combined
  gradient-norm clipping at 1, projected `beta` bounds `[-0.25, 0.25]`.
- Initialize MACE from best975, beta from zero, and reset Adam state.
- Target: 1,000 new joint epochs. Save `last.npz`, `best.npz`, history and
  failed-candidate diagnostics; invalid backward results never replace the
  accepted parameters, gradients or optimizer moments.

## Derivative and compute path

CPU native contracted S/H/packed three-center integrals and analytic
coefficient/exponent VJPs; JAX normalization and fixed-metric whitening.
Transfer contracted physical inputs to GPU, run converged SCF and the shared
strict implicit fixed-point response, return physical cotangents to CPU, then
propagate the contraction-output cotangent through MACE on GPU. No stationary
Lagrangian shortcut, primitive-factor cache or full four-index ERI.

SCF tolerances: energy `1e-12` Ha, density `1e-10`, orbital residual `1e-9`;
maximum 150 cycles, optional level-shift rescue 0.2 Ha, followed by raw fixed-point
validation. Implicit linear-solve tolerance `1e-8`; float64 throughout.

## Gates and provenance

Before any parameter update:

1. Verify 1,255 snapshot files, the original frozen checkout/library and checkpoint.
2. Build the new native library only inside the allocation and isolated source directory.
3. Run native exponent and joint-training regressions on CPU.
4. Check complete GPU/native directional gradients on CH4 and N2 against
   reconverged fourth-order finite differences.
5. Evaluate all 24 energies/gradients and verify per-structure energy parity
   with best975 within `1e-7` Ha, including N2's original solution.

Local targeted regression: **45 passed in 24.41 seconds**.
Remote readiness/running status must be read from job artifacts, not inferred
from submission or this specification.

## Submission

Slurm job **95224**, home partition, one typed V100, 16 CPU cores, 128 GiB RAM,
48-hour limit. GPU preallocation disabled. The unrelated 200-structure job
94863 remains untouched; at submission its 32 CPUs plus this job's 16 total 48.

Formal remote directory:
`/home/user_yuanjiao/private/homefile/GradSCF/artifacts/nnao-joint-native-24-20261010/job-95224/`.

The first submission, 95223, exited during environment setup because its
compute node lacked `/opt/rh/gcc-toolset-12/enable`. No build or training ran.
The replacement records the actual node compiler and uses the existing system
compiler when that optional toolset is absent; the original working native
library was built with GCC 8.5.0.

The source snapshot is separate from the running original checkout. Check
`status.txt`, `build.log`, `regression.log`, `training/gradient-checks.json`,
`training/parent-parity.json`, `training/batch-progress.json` and
`training/progress.json`. `training/metadata.json` records source/native hashes;
`training/runtime.json` and `gpu-inventory.csv` record the execution device.
