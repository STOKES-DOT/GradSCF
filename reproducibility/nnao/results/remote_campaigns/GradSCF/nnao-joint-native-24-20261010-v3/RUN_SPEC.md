# Joint variational training, 24 structures — final startup revision

Use the original 24 GMTKN55 pilot structures and best975 shared MACE weights.
Objective: mean converged DF-RHF total energy. Target: **1,000 new full-batch
epochs**, with one accepted joint update per epoch after all 24 backward
evaluations are validated.

| Setting | Value |
|---|---|
| Basis | all-electron `szp663_direct`, fixed core6 |
| Coefficients | 7,167 trainable MACE parameters; 4 normalization constants frozen |
| Exponents | 14 shared H s/p and C/N/O/F s/p/d log scales; primitive ratios fixed |
| Initial exponents | original pool, beta=0 |
| Adam rates | MACE `1e-5`, exponent scales `1e-3` |
| Gradient clipping / bounds | joint norm 1; beta in `[-0.25, 0.25]` |
| Auxiliary basis | fixed `def2-universal-jkfit` |
| SCF | energy/density/orbital tolerances `1e-12/1e-10/1e-9`, 150 cycles, rescue shift 0.2 Ha |
| Backward | strict implicit SCF, linear-solve tolerance `1e-8` |
| Arithmetic | float64; no full ERI or primitive-factor cache |
| Resources | V100 ×1, 16 CPU cores, 128 GiB RAM, 48-hour limit |

MACE and physical-input SCF/implicit response execute on GPU. Native contracted
integrals and their analytic coefficient/exponent VJPs execute on CPU with
explicit transfer of physical inputs and cotangents. Both parameter groups are
updated by Adam; no Powell search or stationary-Lagrangian shortcut is used.

## Startup corrections and evidence

All original failed attempts and diagnostics are retained in the preceding
`nnao-joint-native-24-20261010` and `...-v2` directories.

- Job 95223 exited before calculation because a compiler activation script was
  absent on its compute node. The script now detects that optional toolset.
- Diagnostic 95225 localized a directional-check discrepancy to perturbing
  immutable MACE activation constants. Version 2 explicitly masks them in both
  checks and updates; CH4/N2 full GPU/native gradient errors then fell to
  `1.51e-10` and `2.48e-10` respectively at finite-difference step `1e-4`.
- Version 2 job 95226 passed **24/24** initial SCF/gradient checks and parent
  energy parity, and saved epoch 0. Its checkpoint readback then failed because
  Python tuple metadata became a JSON list. No optimizer update occurred.
- Version 3 JSON-normalizes model metadata before saving/comparing identity.
  Source/data/configuration checks remain strict. The new regression exercises
  an actual save/read/update cycle; all **5 focused tests passed in 3.16 s**.
  This is a serialization-only correction; integral/SCF/model mathematics are
  unchanged from the validated version 2.

The complete immutable snapshot is rebuilt and all remote gates are repeated.
Status, gradient checks, parent parity and accepted epochs are recorded in
the job directory. `last.npz` and `best.npz` contain the flat MACE vector followed
by 14 beta values, gradients, Adam state and metadata. Failed candidates are
diagnostic artifacts only. Existing unrelated 200-structure training and the
original best975 checkpoint are unchanged.

This is optimization on the original training set, not held-out validation.

## Live submission record

Job **95230** is running on **n002**, Tesla V100-SXM2-32GB, under the allocation
above. Remote source verification and native build succeeded. The 5 focused
startup tests passed in 7.84 seconds (one NNX deprecation warning).
The full GPU/native gradient checks passed again: CH4 error **8.56e-10**, N2
error **3.20e-10**. The current job then entered its initial full-batch evaluation;
at the captured startup snapshot **3/24** structures had completed, with no
accepted optimizer update yet. The earlier v2 all-structure parity check had a
maximum absolute energy difference of **5.12e-13 Ha**.

Formal remote run:
`/home/user_yuanjiao/private/homefile/GradSCF/artifacts/nnao-joint-native-24-20261010-v3/job-95230/`.
Read `training/batch-progress.json` for the current structure, and
`training/progress.json` for accepted epochs after initialization. These live
files supersede the dated startup observation above.
