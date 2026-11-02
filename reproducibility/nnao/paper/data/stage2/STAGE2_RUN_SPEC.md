# NNAO second stage: 200 training structures and 50 independent validation structures

This file is an English translation of the imported source record, not a new run or result. Dates, status statements, and numerical values refer to that source record.

Date: 2026-10-07 (Beijing time). The user confirmed fresh model initialization, retaining the HF variational-energy objective and implicit backward without loading first-round weights.

## Data and training

Fixed GMTKN55 v2 snapshot, CHNOF, neutral, spin=0, even electrons, and 2–40 atoms. 200 train / 50 validation, grouped by chemical formula with no formula crossing training and validation; validation excludes formulas seen in the first round. At most two geometries per formula, deduplicated by coordinate hash. This grouping is not a full reaction-connectivity split and does not establish full GMTKN55 reaction-energy benchmark performance.

Spherical all-electron `szp663_direct/core6`; fixed geometry, primitive exponents, and inner-shell parameters. MACE predicts full contraction coefficients for valence and multi-primitive polarization shells. One shared model, CHNOF vocabulary, channels=8, 2 interactions, max_ell=1, correlation=2, seed=0, fresh reference-bias initialization.

Objective: the equal-weight mean of 200 converged RHF/DF total energies. 1000 epochs; each epoch traverses all 200 training structures and performs one Adam update, with global gradient clip=1. Cosine learning rate decreases from 1e-4 to 1e-5. The 50 validation structures do not contribute gradients; their converged energies are recorded every 20 epochs and at the end, with a separate `best_validation.npz`.

Forward tolerances remain: energy 1e-12 Ha, density 1e-10, orbital gradient 1e-9, max_cycle=150. The shared implicit module uses linear-solve tolerance 1e-8, max_iter=100, restart=40, and no additional regularization.

## Memory and admission

DF auxiliary basis `def2-universal-jkfit`. Estimated raw cache total 304.810 GiB, largest individual structure 5.961 GiB, including the full-rank upper bound of the auxiliary index but excluding construction, SCF, or backward temporary memory. Metric truncation may reduce the actual rank.

Fixed integrals are stored in the existing uncompressed NPZ files. Numeric NPY members are mapped read-only from the original files with memmap; the OS can reclaim file pages, and each structure is passed to the GPU as JAX inputs. The 305 GiB arrays are neither loaded into RAM as a whole nor duplicated in another stored copy. Peak GPU memory per structure still requires measurement. No full four-index ERI is stored.

All 250 structures are required to complete two directional FD checks, finite coefficient/parameter implicit-gradient checks, and 5-step Adam admission individually. FD perturbations evaluate only reconverged energies with the same SCF quality checks; initial AD and Adam steps still execute the actual backward path. Any failure is recorded and blocks formal training; structures are not removed to recompute the mean. spin=0 and even electron count do not establish orbital stability; no stability-Hessian certification is performed in this round.

## Jobs and dependencies

| Job | Task | Resources / time limit | Dependency |
|---|---|---|---|
| 93225 | V100 check of the two largest structures | 1 V100, 32 CPU, 128 GiB / 2 h | Independent check; currently awaiting a GPU |
| 93243 | Preparation of fixed DF caches for 250 structures | 32 CPU, 128 GiB / 24 h | Started on n001 |
| 93255 | Complete numerical admission of 200 train and 50 heldout structures | 1 A100_80G, 32 CPU, 128 GiB / 48 h | afterok:93243 |
| 93256 | Formal shared training of 200 structures for 1000 epochs | 1 A100_80G, 32 CPU, 128 GiB / 48 h | afterok:93255 |

CPU preparation, validation, and training are serialized by dependencies; if the independent probe runs concurrently, the total CPU request does not exceed 64. GPU stages had not started; queued jobs or completed cache preparation cannot be treated as successful SCF/training numerics.

Code/logs: `~/private/homefile/GradSCF/artifacts/nnao-stage2-20261007/`. Formal cache: `~/private/datafile/GradSCF_GMTKN55/nnao-stage2-20261007/cache/`. The approximately 305 GiB cache remains remote; large caches are not downloaded into the TF result archive.

Local combined verification at this point: 56 passed in23.76s; script compilation and `git diff --check` passed. Source selection inventories, CLI, Slurm scripts, and actual results record data/code/cache hashes.

Early check job 93221 failed at startup when calling the nonexistent `/usr/bin/time` (exit127), producing no integral or GPU result. 93225 was changed to use the shell's built-in time; the old log is retained only as a diagnostic archive.
