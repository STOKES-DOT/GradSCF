# Joint variational training: 24 structures, native analytic gradients

Continuation of the original 24-structure best975 shared MACE checkpoint.
The scientific objective and settings are unchanged from
`../nnao-joint-native-24-20261010/RUN_SPEC.md`: 1,000 new full-batch epochs,
DF-RHF mean total energy, MACE learning rate `1e-5`, exponent learning rate
`1e-3`, global clipping at 1, and 14 element/angular log-exponent scales
bounded to `[-0.25, 0.25]`. Core6, geometry and auxiliary basis remain fixed.

## Why a new source snapshot

The first compute job (95223) exited before calculation due to an unavailable
optional compiler activation script. Job 95224 built successfully and passed
28 CPU regressions but was stopped by the full-network directional gradient
gate before any training update.

Read-only diagnosis 95225 established:

- CH4 exponent-only direction: analytic `0.000694142880052` versus
  finite difference `0.000694142871656` at step `1e-2`; native physical-input
  JVP and VJP contractions agree. Smaller steps also agree within numerical noise.
- The discrepancy is confined to network directions. The vendored MACE stores
  fixed activation normalization scales as `nnx.Param(is_mutable=False)` and
  explicitly applies `stop_gradient` to them. The old random finite-difference
  direction changed these fixed constants, outside the training domain.

Version 2 constructs a mutability mask in flattened parameter order. Both the
gradient check and Adam explicitly freeze those constants, while preserving
the original checkpoint vector and unchanged initial forward model. No
numerical tolerance was loosened and no sample was removed. Original runs,
source snapshots and the best975 checkpoint remain unchanged.

## Verification and execution

Local regression after the fix: **46 passed in 22.51 seconds**. This includes
a reproducer for frozen MACE constants, optimizer moments/parameter rollback,
native exponent/coefficient pullback finite differences and implicit SCF tests.
The NNX State flattening method emits a deprecation warning on the local Flax
version; the returned mask is checked against the actual ravel order.

Remote gates are repeated on the new immutable source snapshot: build, CPU
regression, joint GPU/native finite differences for CH4/N2, then 24/24 parent
energy parity and finite backward gradients. Only then may epoch 1 be accepted.

Resources: home partition, V100 ×1, CPU cores ×16, 128 GiB host RAM, 48-hour
walltime. MACE and implicit SCF run on GPU; native integrals/pullbacks on CPU.
GPU preallocation is disabled. The independent 200-structure A100 run remains
untouched. Submitted as Slurm **95226**, running on **n002** with a Tesla
V100-SXM2-32GB. Formal run directory:
`/home/user_yuanjiao/private/homefile/GradSCF/artifacts/nnao-joint-native-24-20261010-v2/job-95226/`.
Submission and progress are recorded separately in job artifacts.

Remote startup evidence for 95226:

- New native library built with GCC 8.5.0 and the environment's MKL LP64 runtime.
- CPU regression: **29 passed in 44.54 seconds**, one NNX deprecation warning.
- Real MACE mask: 7,167 trainable parameters and 4 fixed normalization constants
  (flat indices 5, 262, 2631, 2888), plus 14 trainable exponent scales.
- Complete GPU/native directional finite differences at step `1e-4`:
  CH4 absolute error **1.51e-10**, N2 **2.48e-10**.
- Initial N2 energy `-108.8037455630876` Ha matches the best975 reference
  `-108.80374556308769` Ha. Both gradient checks reconverged SCF without rescue.

The full 24/24 initialization and accepted epoch count are recorded in
`training/parent-parity.json` and `training/progress.json`; passing the two
directional checks alone is not completion of either condition.
