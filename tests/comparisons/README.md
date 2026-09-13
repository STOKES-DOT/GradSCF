# Independent comparisons and reproducibility workflows

These scripts depend on optional PySCF (and, for GPU reference runs, gpu4pyscf).
They live under `tests/` so the installed GradSCF runtime and native tools do not
import either package. Run them from the repository root with the same arguments
as their former `tools/` or `examples/` locations. Their output paths are unchanged.
For example:

```sh
PYTHONPATH=src python tests/comparisons/compare_pyscf_vs_jax_tddft_no_neural.py --help
PYTHONPATH=src python tests/comparisons/closed_shell_s1_self_consistent_train.py --help
```

The native experiment scripts stay in `tools/` and retain their own AD/finite-
difference, SCF convergence, and invariance checks. Independent reference checks
can be run against their saved output without adding PySCF to those tools:

```sh
python tests/comparisons/native_experiment_reference.py geometry artifacts/geometry-gradients/h2-native.json
python tests/comparisons/native_experiment_reference.py h2-contractions artifacts/optimized-321g-contractions/summary.json
python tests/comparisons/native_experiment_reference.py contractions artifacts/benzene-321g-contractions/summary.json
```

The corresponding automated checks are in `tests/test_geometry_gradient_check.py`,
`tests/test_optimize_contraction_coefficients.py`, and
`tests/test_benzene_contraction_gradient.py`. PySCF comparisons skip when PySCF is
unavailable; native numerical checks still run.
