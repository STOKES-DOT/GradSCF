# NNAO fixed-exponent restart: parameter-mask repair

Date: 2026-10-13. Stage C: fixed primitive exponents, shared MACE contraction training.

## Root cause

Restart 96305 reproduced the accepted epoch-22 energies and gradients, but failed 187/200 parameter directional finite-difference checks. The fixed-pool workflow perturbed all 7171 flattened values, including four MACE activation normalization constants marked `is_mutable=False` and read through `stop_gradient`. These constants occupy indices 5, 262, 2631, and 2888. Only 7167 values are trainable.

All 29 parameter leaves are float64. Explicit float64 promotion did not change the error, and a step-size sweep did not remove it. Independent SCF coefficient derivatives agreed with finite differences to approximately 1e-9. Perturbing only the immutable constants reproduced the discrepancy. At the zero-initialized output head these constants do not affect predictions, explaining why initialization checks passed.

## Repair

Reuse one metadata-derived trainable mask in the parameter FD gate and Adam. Normalize the FD direction after masking. Keep the default predictor API unchanged; expose the mask through an optional return value. Preserve the existing Optax state layout, parameter vector, MACE forward function, native integral path, and SCF thresholds. The joint-training path imports the same mask helper.

Only three scientific modules differ from the validated performance snapshot: `training/alternating.py`, `training/trainer.py`, and `workflows/fixed_pool.py`, under `src/gradscf/model/nnao/`.

## Numerical verification

Three-system isolation ran on n005, NVIDIA A100 40 GB, float64, epoch-22 parameters. Central FD step remains 1e-4; tolerances remain atol=2e-6, rtol=2e-5. Errors are absolute directional energy-derivative discrepancies.

| Structure | All-value direction | Trainable-only direction |
|---|---:|---:|
| INV24/Ether | 5.07947e-6 | 3.45595e-10 |
| PX13/hf_4_ts | 4.48150e-5 | 2.43265e-10 |
| S22/08a | 1.59209e-6 | 8.08560e-10 |

Evidence: `diagnosis.json`, `mask-diagnosis.json`, and `mask-96502.log`. Commands are recorded in the accompanying diagnostic Slurm scripts. Remote regression job 96508 passed 39 tests in 48.37 s, including real MACE training/restoration and the existing joint path. Local expanded regression passed 46 tests with one missing-MACE dependency skip; migration-specific checks passed four tests. See test logs and `validate.slurm`. No memory or end-to-end speed claim is made by this repair check.

## Restart provenance and current boundary

Replacement job **96510** started on 2026-10-13 at 10:13:17 +08 on n004: one A800, 32 CPUs, 256 GiB host memory, 72 h allocation. Output directory is the remote counterpart of this directory, with checkpoints in `training/`.

The accepted epoch remains **22**, mean energy **-383.3513154811086 Eh**. Parameters, accepted gradient, serialized Adam state, and optimizer count 22 are preserved; Adam is not reset. The original archive remains intact.

Target scientific-source SHA256: `ec3b63f54f67fd89069bcd1a6992dc8732d5bb3380a4ea49b71818d0e39c7ff6`.

At the observed job elapsed time 9:44, 8/200 systems had completed the first full replay including compilation. No restart failure record was present. The full 200-system replay and masked FD gate must pass before epoch 23 updates. This report does not establish their completion; consult live `training/phase.json`, `training/batch_progress.json`, and `restart-gates.json`.
