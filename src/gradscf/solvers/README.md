# Shared numerical solvers

`gradscf.solvers` owns numerical forward algorithms, implicit derivatives and
their convergence checks. Electronic-structure modules construct Hamiltonian
actions, residuals, orbital maps and physical selection criteria. They do not
maintain copies of these solvers. Clients import the shared modules directly;
historical solver forwarding modules have been removed.

## Ownership

| Implementation | Shared owner | Clients |
| --- | --- | --- |
| Symmetric/Hermitian Davidson | `eigen/davidson.py` | TDA, CIS/CI, stability, periodic TDA |
| Real isolated-root eigenvector response | `eigen/response.py` | TDA and CI coefficient-dependent objectives |
| Real RPA Davidson and metric energy derivatives | `eigen/rpa.py` | Full TDHF/TDDFT, periodic Gamma response |
| Bounded complex dense RPA and metric energy derivatives | `eigen/rpa.py` | Small non-Gamma periodic response |
| Regularized full-spectrum and inverse-square-root derivatives | `eigen/spectral.py` | SCF orbital diagonalization and overlap response |
| Checked GMRES/direct solves and transpose response | `linear/` | Eigenvector response, SCF/GW implicit derivatives |
| Scalar-bordered Schur solve | `linear/schur.py` | scGW charge-constrained response |
| Fixed-point/root implicit differentiation | `nonlinear/` | SCF and scGW |
| DIIS and callback-based orbital minimization | `nonlinear/` | SCF families |

`tddft/eigensolvers.py`, `tddft/eigenvector_differentiation.py`, `scf/implicit.py`,
`scf/diis.py`, `scf/_orbital_solver.py`, and `ci/response.py` have been removed.
Architecture tests prevent their return and reject imports of those paths.
The shared layer has no dependency on CI, TDDFT, SCF, GW or PBC modules.

Import migration:

| Removed path | Canonical import |
| --- | --- |
| `gradscf.tddft.eigensolvers` | `gradscf.solvers.eigen.davidson` or `.rpa` |
| `gradscf.tddft.eigenvector_differentiation` | `gradscf.solvers.eigen.response` |
| `gradscf.scf.implicit` | `gradscf.solvers.nonlinear` and `gradscf.solvers.linear` |
| `gradscf.scf.diis` | `gradscf.solvers.nonlinear.diis` |
| `gradscf.scf._orbital_solver` | `gradscf.solvers.nonlinear.minimize` |
| `gradscf.ci.response` | `gradscf.solvers.diagnostics` |

`ImplicitFixedPointConfig` and `implicit_fixed_point_solution` are no longer
re-exported from `gradscf.scf`; import them from `gradscf.solvers.nonlinear`.
Use `EigenGradientMode` from `gradscf.solvers.eigen.response` for solver mode
annotations. TD-SCF tolerances and positive-frequency defaults remain in
`gradscf.tddft.defaults`, which defines method configuration, not solver aliases.

Physical construction remains in method modules: TD response matrices and
amplitude conventions, CI spaces, orbital parameterizations, scGW equations,
and the beta-dependent charge-resolution threshold. Calls to JAX's elementary
matrix operations inside physical expressions are not separate solver engines.

## Public matrix-free API

```python
import jax
import jax.numpy as jnp
from gradscf.solvers import (
    LinearOperator, EigenSolverConfig, LinearSolverConfig,
    solve_hermitian, solve_linear,
)

def energy(matrix):
    op = LinearOperator(
        matrix.shape, matrix.dtype,
        matvec=lambda v: matrix @ v,
        matmat=lambda x: matrix @ x,
        diagonal=jnp.diag(matrix),
    )
    return solve_hermitian(op, config=EigenSolverConfig(nroots=1)).values[0]

a = jnp.array([[1.0, 0.1], [0.1, 2.0]])
gradient = jax.jit(jax.grad(energy))(a)
response = solve_linear(a, jnp.ones(2), config=LinearSolverConfig(rtol=1e-10))
```

Vectors have shape `(n,)`; block vectors have shape `(n, nvec)`. `matmat` is
optional: its default applies `matvec` over columns with `vmap`. A supplied
`transpose_matvec` must implement the algebraic transpose; otherwise `.T`
constructs the transpose through JAX. It is not a conjugate-transpose alias.
Construct `LinearOperator` inside a transformed function, or close over it;
the callable container itself is not a dynamic JIT argument. Captured numerical
parameters remain differentiable. Multiple linear RHS can use caller-side vmap.

The new `solve_hermitian` and `solve_linear` APIs accept real floating-point
data. Dense symmetric inputs are checked rather than silently repaired;
matrix-free callers assert symmetry. Existing low-level complex forward paths
remain available for periodic calculations, without widening the real API's
derivative contract. Dense materialization has a configurable size limit.
Davidson's legacy default subspace can reach the full dimension; specify
`max_subspace` to bound basis storage for larger problems.

Results are JAX-compatible named tuples, containing the solution/eigenpairs,
true residual norms, convergence flags and integer status (`0` success,
`1` not converged/invalid). Eigen diagnostics are per root. Iteration counts
are not reported because the inherited kernels do not expose reliable counts.
Linear `maxiter` counts GMRES restart cycles, not individual Krylov steps.

## Derivative contracts

- Linear systems use `custom_linear_solve`: both operator and RHS derivatives
  are retained, including transpose solves and higher derivatives. Zero and
  tiny RHS are handled by scaling inside the opaque numerical solve. Forward
  and transposed solves are independently checked using the true residual.
- Eigen `gradient_mode='eigenvalue_only'` differentiates the Rayleigh energy
  at a converged isolated root. Vectors are stopped. The
  `implicit_eigenvector` mode also solves the constrained eigenvector response.
  These Davidson rules promise first-order response only.
- Failed eigenpairs remain inspectable; their derivatives are NaN. Failed
  linear or adjoint solves return NaNs. Residual predicates stay inside opaque
  linear solves so that the tangent map can still be transposed correctly.
- RPA energy response uses the indefinite metric, not a Hermitian problem
  obtained by symmetrizing the RPA matrix. The real Davidson and complex dense
  paths expose eigenvalue-only response; X/Y response is not implemented.
  RPA roots must be isolated, real-frequency and of positive metric norm.
- `regularized_eigh` preserves the existing SCF gap-broadening policy near
  degeneracy. `inverse_sqrt` differentiates the matrix function through a
  Sylvester equation at repeated positive eigenvalues. Existing SCF second,
  third and mixed derivative contracts are retained.
- `attach_root` and `implicit_fixed_point_solution` attach the residual-based
  derivative to a supplied converged solution. The callback minimizer retains
  its actual unrolled trajectory; domain-level policy chooses between implicit
  and unrolled modes. Regularization changes the response problem and remains
  explicit rather than a silent numerical fallback.

TD-SCF calculation signatures, thresholds, positive-root selection, tuple layouts
and successful numerical behavior are preserved. Invalid symmetric/RPA
eigenvalue derivatives now follow the shared NaN policy. The generic public
Hermitian API does not filter negative eigenvalues: stability analysis and CI
must retain those roots.

## Validation and further work

Tests cover nonsymmetric linear systems and transpose derivatives; zero/tiny
RHS; JIT/vmap; isolated-root energies and vector response; unconverged states;
real/complex structured RPA energy derivatives; SCF higher-order response;
and preservation of CI, TD-SCF, DIIS and physical stability calculations.
Run with CPU and float64:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
python -m pytest -q tests/solvers tests/ci tests/test_tddft_eigensolvers.py \
  tests/test_scf_higher_order.py tests/test_scf_implicit_and_xc_energy.py
```

Generic non-Hermitian EOM-CC Davidson/Arnoldi, biorthogonal vector response,
degenerate-subspace derivatives, and CC amplitude equations are future work.
The present RPA implementation is not a generic non-Hermitian EOM solver.
GPU behavior has not been validated in this migration.
