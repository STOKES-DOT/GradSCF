# Shared solver migration validation — 2026-09-21

The first section records the initial migration, before removal of the
compatibility export files. The removal is validated separately below.
See `README.md` for current canonical import paths.

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

## Compatibility export removal

After removing the six forwarding files and the fixed-point solver exports
from `gradscf.scf`, all repository clients were migrated to `gradscf.solvers`.
TD-SCF numerical defaults retain their values in `tddft/defaults.py`.
The numerical implementations were not changed in this cleanup.

With the same CPU/float64 environment:

```bash
python -m pytest -q tests/solvers tests/ci tests/test_tddft_eigensolvers.py \
  tests/test_pyscf_style_excited_state_api.py \
  tests/test_scf_implicit_and_xc_energy.py tests/test_scf_higher_order.py \
  tests/test_scf_autodiff.py tests/test_scf_diis_state.py \
  tests/test_scf_diis_scaling.py
# 128 passed, 2 PySCF OpenMP warnings, 105.80 s

python -m pytest -q tests/pbc/test_tdscf.py tests/test_scf_orbital_optimization.py \
  -k 'cached_response or lowers_energy'
# 3 passed, 22 deselected, 12.23 s
```

Architecture tests now require the old files to be absent and reject imports
of removed paths across source, tests, tools and examples. The two pre-existing
dirty files touched by the migration were checked against snapshots: only their
intended import lines changed. `git diff --check` also passed.

## Isolated degenerate spectral subspaces (2026-09-21)

`solve_spectral_projector` was validated on real symmetric toy matrices on
arm64 CPU (`TFRT_CPU_0`), JAX 0.8.1, float64, `OMP_NUM_THREADS=1`.
The reference and mathematical contract are in [DEGENERACY.md](DEGENERACY.md).
No generalized metric, non-Hermitian/RPA, GPU or higher-order subspace-response
claim is made by these tests.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/solvers/test_subspace.py
```

Focused result: **21 passed**, 40.11 s. The coverage includes:

- Exact internal degeneracy, splitting perturbations, and small/nonzero internal
  gaps; projector and trace derivatives against analytic small-matrix values
  and independent NumPy eigendecomposition finite differences.
- JIT, vmap, vector/block probes and probe derivatives; JVP/VJP agreement.
- Frame rotations within the selected span, including a nondiagonal Ritz block;
  projector symmetry/idempotence, `P dP P = 0` and `dP P + P dP = dP`.
- A 24-dimensional matrix-free operator with an eight-column Davidson bound;
  no full operator block is requested in forward or backward evaluation.
- A cut degenerate boundary, unresolved small boundary gap, unconverged guard
  root, failed primal solve and failed adjoint solve; finite primal diagnostics
  but invalid derivatives where required, including a direct response solver.
- The full-space identity projector; reduction to the existing isolated-root
  response for a nondegenerate rank-one subspace.
- Bounded restart capacity, arbitrary starting vectors for a diagonal operator,
  and disconnected invariant sectors whose low eigenvalues are missed by
  diagonal-only guesses (dimensions 6 and 20).

The shared Davidson forward fixes were independently reviewed. Accepted append
counts are capped by remaining capacity, so the restarted basis cannot write
past its last column. Vanishing projected preconditioner directions use the
projected original residual rather than a normalized roundoff direction.

The final shared-solver and client regression command was:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/solvers tests/ci tests/test_tddft_eigensolvers.py \
  tests/test_scf_higher_order.py
```

Result: **115 passed**, no skips, 166.03 s. Two PySCF warnings state that
OpenMP is unavailable. This covers the new subspace suite and existing
isolated-root, CI, TDDFT and higher-order SCF regression tests. The latter
validate preservation of their existing contracts, not higher-order projector AD.

### Executable example

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/degenerate_subspace.py
```

The dimensionless four-state example selects the twofold eigenvalue 1, with
excluded eigenvalues 3 and 5. A perturbation mixes/splits the degenerate pair
and couples it to the excluded states. For `loss = probe.T @ P @ probe`:

| Quantity | Result |
| --- | ---: |
| Loss | 0.2500000000000002 |
| JVP | -0.11499999999999991 |
| VJP | -0.11500000000000000 |
| Central finite difference, step 1e-4 | -0.11499999993655474 |
| Absolute derivative error | 6.35e-11 |
| Boundary gap | 1.9999999999999973 |
| Maximum Ritz residual | 1.82e-15 |
| Cut-cluster status with one selected root | 2 (invalid response) |
| Elapsed time including compilation | 6.15 s |
