# Shared solver migration validation — 2026-09-21

Environment: local arm64 CPU, JAX 0.8.1, float64 enabled. Commands were run
inside the isolated `refactor/shared-solvers` worktree with `PYTHONPATH=src`,
`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, and `OMP_NUM_THREADS=1`.

| Test group | Result | Elapsed |
| --- | --- | --- |
| `tests/solvers` | 17 passed | 23.84 s |
| CI, TD-SCF eigen/API, SCF implicit/higher-order, DIIS state/scaling | 95 passed | 81.72 s |
| Orbital optimization/gradients, UHF/UKS stability | 21 passed, 3 skipped | 75.96 s |
| scGW response and periodic TD-SCF | 26 passed, 8 skipped | 144.20 s |

Total: **159 passed, 11 skipped**. Skips require the unavailable optional
`jax_xc` package (three UKS and eight periodic DFT cases). The scGW tests emit
JAX complex-to-real cotangent warnings also present before this migration.
This is targeted migration validation, not the complete repository test suite.
GPU execution and generic non-Hermitian EOM-CC are not validated here.

Exact test selections:

```bash
python -m pytest -q tests/solvers

python -m pytest -q tests/ci tests/test_tddft_eigensolvers.py \
  tests/test_pyscf_style_excited_state_api.py \
  tests/test_scf_implicit_and_xc_energy.py tests/test_scf_higher_order.py \
  tests/test_scf_diis_state.py tests/test_scf_diis_scaling.py

python -m pytest -q tests/test_scf_orbital_optimization.py \
  tests/test_scf_orbital_gradients.py tests/test_uhf_stability.py \
  tests/test_uks_stability.py

python -m pytest -q -rs tests/gw/test_gw_scgw_response.py tests/pbc/test_tdscf.py
```

`examples/shared_solvers.py` also ran successfully. Its matrix-free
eigenvector-dependent observable gave derivative -0.23383077651091166 versus
central finite difference -0.2338307764748748 (step 1e-5); the nonsymmetric
linear solve had residual 1.11e-16.

Additional checks:

- Architecture regression: historical solver files contain exports only;
  shared modules do not import electronic-structure modules or JAX private APIs.
- DIIS and the callback minimizer were relocated without changing their source;
  the spectral forward/JVP implementations retain identical ASTs.
- `git diff --check` passed.
- Original-workspace CI baseline files retain their recorded SHA-256 hashes.

Migration boundaries: domain-specific residuals, Hamiltonian construction,
physical root selection and charge-resolution criteria remain in method modules.
RPA exposes eigenvalue response only; its X/Y derivatives are not supplied.
Davidson eigenvector response assumes isolated converged roots and is first-order.
SCF's existing tested higher-order root and matrix-function response is retained.
