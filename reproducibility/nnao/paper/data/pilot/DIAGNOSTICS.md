# NNAO pilot execution record and unresolved issues

This file is an English translation of the imported source record, not a new run or result. Dates, status statements, and numerical values refer to that source record.

Date: 2026-10-07. The formal results come from `job-93084/validation/summary.json`: 24 records, 23 passed and 1 failed. `README.md`, `metrics.csv`, and `verification.json` do not count diagnostic runs or experimental patches as formal passes.

## Execution and provenance

| Job | Scope | Outcome and rules for use |
|---|---|---|
| 93079 | Native build preparation | The node lacked the GCC12 path listed in the manual; startup failed and produced no scientific result |
| 93081 | Environment and minimal validation | Imported GradSCF from the old setup directory; stopped and excluded from scientific conclusions |
| 93084 | CPU native build, DF cache, and GPU validation of 24 structures | V100-SXM2-32GB, n002, 16 CPU, 64 GiB; runtime 13 minutes 40 seconds, exit 1; 23 passed and S22/08 failed |
| 93092 | Supplementary checks dependent on the main job | The main-job failure made the dependency unsatisfiable; cancelled |
| 93093 | Final-parameter FD and exact shell-direct comparison for three structures | CPU, 4 cores, 16 GiB, exit 0; see `job-93084/final_and_direct.json` |
| 93094 | GPU diagnostics with debug callbacks | The CUDA-only environment had no registered CPU callback device, so the diagnostic failed; this is not root-cause evidence for the original gradient failure |
| 93096 | Independent GPU response-operator diagnostics | Forward values and JVP/VJP were finite; the wrapper's true residual did not meet the requirement; see `response-gpu-93096.json` |
| 93098 | Trial with tighter internal stopping criteria | The initial S22/08 gradient still failed; no regression on the other structures was reached |
| 93099 | Residual-correction trial with debug output | The gradient was finite, but instrumentation was a confounder; this does not establish a formal fix |
| 93100 | One residual-correction trial without debug output | The initial S22/08 gradient still failed; no regression on the other structures was reached |
| 93103 | Unwrapped comparison of cuda and cuda,cpu configurations | Both runs evaluated energy, integrals, and gradients on cuda:0, and both had 94 nonfinite gradient components |

Resetting the source path after activating Conda and asserting the actual module origin resolved the old-source import in 93081. Each structure's result stores the formal-run source and native-library hashes. The native build succeeded with the node's available GCC/G++8.5 and the environment's MKL; no new dependencies were installed.

## S22/08: methane dimer

The formal GPU forward energy was approximately **−80.3194689323405 Ha**; 12 SCF cycles, orbital residual approximately **3.37×10⁻¹¹**, minimum overlap eigenvalue approximately **0.0873954**, and HOMO–LUMO gap approximately **1.14939 Ha**. This was neither an unconverged forward SCF state nor a closed occupied–virtual gap.

Independent density-response JVP/VJP values were finite. The original GPU GMRES wrapper candidate had a true residual of **2.4768×10⁻⁸**, RHS norm **20.8948**, and relative residual **1.18536×10⁻⁹**, exceeding the original acceptance requirement **10⁻⁹**. The validity check rejected that candidate and propagated NaN as designed. Increasing restart/maxiter did not improve this candidate.

Independent calls to the upstream incremental/batched solver gave a relative residual of approximately **3×10⁻¹⁰**; the full gradient for the same structure on local CPU/JAX0.8.1 was also finite. These observations help locate a linear-solve/compilation-composition issue, but do not alone establish a deeper cause. Internal occupied/virtual degeneracy cannot be treated as a confirmed root cause.

## Trials and final code state

The trials included tighter batched internal stopping criteria and one `Aδ=b−Ax` residual correction. The original true-residual acceptance threshold and NaN-on-failure policy were retained throughout. The experimental code passed 28 targeted CPU regressions, but did not restore the gradient in the original GPU case without a diagnostic wrapper.

The trial was therefore retained only as `experimental_gmres_response.patch`. Local and remote `gmres.py` were restored to the formally measured baseline version, SHA-256:

```text
024432754b5753e3cd2b67eb1f1be3ac5c833f89ec9823071cd194d1ed2f6689
```

The final workspace added only the pilot entry points and 9 orchestration/directional-check regressions; existing user changes were preserved. No solver-algorithm change that left the original GPU issue unresolved was retained.

The final source-file inventory comparison confirmed matching hashes for all Python/basis JSON/DAT files shared by the local and remote copies. The overall remote hash still matched the 93084 measurement record. The local copy contained one additional generated CMake `InstallScripts.json`, which is not scientific source; see `source_inventory.json`.

## Final local verification

After restoring the baseline, **32 passed in 26.86 s** (CPU, JAX0.8.1, float64), and `git diff --check` passed. Exact command:

```sh
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q \
  tests/nnao/test_pilot_validation.py \
  tests/solvers/test_gmres_accuracy.py \
  tests/solvers/test_block_linear.py \
  tests/solvers/test_shared.py::test_linear_nonsymmetric_jit_transpose_and_second_derivative \
  tests/solvers/test_shared.py::test_linear_zero_tiny_rhs_and_failed_solve \
  tests/solvers/test_nonlinear.py \
  tests/test_scf_higher_order.py \
  tests/test_scf_autodiff.py::test_failed_adjoint_rejects_approximate_gradient \
  tests/test_scf_autodiff.py::test_shared_implicit_nonlinear_root_and_density_loss_pytree_jit \
  tests/test_scf_autodiff.py::test_fixed_point_argument_gradients_are_preserved
```

The full test suite was not run. The conclusion at this point was that the first validation round had finished but overall admission had failed; S22/08 GPU implicit backward still required repair. Each structure was optimized independently for 5 steps in this round; shared-model training had not started.

## Data and subsequent reproduction

The original TF selection and data were unchanged; these results were stored separately in `results/hf_pilot_20261007/`. The result copy contains raw logs, per-structure JSON, 5-step trajectories, checkpoints, diagnostics, and verification inventories. Large DF integral caches and native build directories remain remote and are absent from the local/TF result copies.

The next diagnostic should isolate compilation differences between the actual custom-root backward operator, numerical solve, and true-residual check; record HLO/array provenance without instrumentation; and use reconverged FD and 5-step Adam on the original GPU case as the repair acceptance test. Stability Hessians, shared training, ions/UHF, DFT, and force validation had not been performed.
