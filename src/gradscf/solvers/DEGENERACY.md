# Differentiating isolated spectral subspaces

`solve_spectral_projector` supports first-order JVPs and VJPs of the projector
onto the lowest `nroots` states of a real symmetric operator, including exact
internal degeneracy. It returns its action on supplied probes and the sum of
the selected eigenvalues. It does not expose individual eigenvector derivatives.

## Mathematical scope and reference

Reference: M. F. Kasim, *Derivatives of partial eigendecomposition of a real
symmetric matrix for degenerate cases*, arXiv:2011.04366v1 (2020),
[paper](https://arxiv.org/abs/2011.04366), Sections 4–5.
The paper identifies compatibility conditions for finite eigenvector response
and projects out the full degenerate eigenspace. Its formulas do not make an
arbitrary labeled degenerate eigenvector a differentiable function of every
matrix perturbation. No source code was copied from an external implementation.

GradSCF's initial API instead targets an isolated spectral subspace. If X is
an orthonormal basis, P=XX^T is unchanged under X -> XU for orthogonal U.
Internal eigenvalue crossings and splitting perturbations can leave P smooth
when the selected spectrum remains separated from the excluded spectrum.

Differentiating AX=XH with H=X^TAX, and choosing the local frame condition
X^T dX=0, gives the complement-space Sylvester equation

```text
P = X X^T, Q = I - P
Q (A Z - Z H) = -Q (dA) X,  X^T Z = 0
dP = Z X^T + X Z^T
d tr(X^T A X) = tr(X^T (dA) X)
```

This derivation uses the whole selected subspace, even if it contains distinct
or nearly equal eigenvalues. It does not divide by internal eigenvalue gaps.
For a concrete matrix, parameters and perturbations must preserve symmetry;
for example, construct A=(B+B^T)/2 from unconstrained trainable parameters B.
The public operator interface continues to require symmetry from matrix-free
callers. The generic RPA and non-Hermitian EOM problems are outside this scope,
as is a variable metric M in AX=MX Lambda.

## Public interface

```python
import jax
import jax.numpy as jnp
from gradscf.solvers import EigenSolverConfig, solve_spectral_projector

jax.config.update('jax_enable_x64', True)
a = jnp.diag(jnp.array([1., 1., 3., 5.]))
probe = jnp.array([.3, .4, .5, .2])
config = EigenSolverConfig(nroots=2, adjoint_tol=1e-11)
out = solve_spectral_projector(a, probe, config=config)
assert out.response_valid
print(out.projection, out.eigenvalue_sum)
```

The first argument also accepts `LinearOperator`. Probes have shape `(n,)` or
`(n,m)` and may themselves depend on the differentiated parameters. The returned
`projection` has the same shape. Passing `jnp.eye(n)` explicitly obtains a
dense projector for small tests; a few probes avoid storing that matrix.
For example, an unweighted sum of transition strengths over a complete cluster
has the form d^T P d and can be evaluated from `projection`.

`EigenSolverConfig` supplies forward and default adjoint controls. This interface
always uses subspace response, independently of that config's `gradient_mode`.
An optional `LinearSolverConfig` controls the Sylvester response solve; the
default is shared checked GMRES. `method='dense'` selects the bounded dense
forward oracle while retaining the same subspace derivative rule.

The result contains:

- `projection`: P times the supplied probes, with first-order response.
- `eigenvalue_sum`: the trace over the selected subspace, with first-order response.
- `residual_norms`: stopped per-root diagnostics, including the boundary root.
- `boundary_gap`: stopped lambda[k]-lambda[k-1], or infinity for the full space.
- `converged`: convergence and orthogonality of the required Ritz roots.
- `response_valid`: those primal checks plus a resolved boundary separation.
- `status`: 0 for a usable subspace, 1 for an invalid primal eigenspace,
  2 for an unresolved boundary.

`response_valid` cannot certify a future loss-dependent adjoint solve. Failed
forward or transposed linear solves invalidate derivatives via the shared
NaN policy. Diagnostic fields are not differentiation targets.

## Boundary selection and failure policy

For k < n, the forward solver obtains k+1 roots. The extra root is a guard for
the selected upper boundary, not an additional differentiated state. Its
convergence is required. With default `gap_atol=gap_rtol=1e-8`, response requires

```text
lambda[k] - lambda[k-1]
    > gap_atol + gap_rtol * max(abs(lambda[k]), abs(lambda[k-1]))
      + residual_norm[k] + residual_norm[k-1]
```

The new interface starts Davidson from lowest-diagonal guesses mixed with
seed-0 Gaussian full-support columns, then orthonormalizes the block. Its width
is at most `2*(k+1)` and is bounded by the configured basis capacity. Pure
diagonal guesses can otherwise converge with zero residual inside an incorrect
invariant sector, missing lower roots in another block. Callers can override
the starting block with `initial_vectors`; guesses are stopped numerical
controls, not physical differentiation inputs. Ordinary `solve_hermitian`
retains its existing default guesses and accepts the same optional starting block.

The residual margin accounts for unresolved Ritz values. A gap among computed
roots is conditional on the forward solver finding the intended lowest roots;
it is not a proof that an iterative eigensolver cannot miss an invariant sector.
As usual for partial eigensolves, physically appropriate starting spaces and
comparison tests matter.

If k cuts a degenerate cluster, the numerical primal is retained for inspection,
`response_valid` is false, and derivatives are nonfinite. Increase `nroots`
until the complete cluster is included. A fixed count preserves JIT shapes;
there is no silent root-count expansion. For Davidson, an explicit
`max_subspace` must be at least `min(n,k+3)`: the guard root plus expansion room.
Internal small gaps are neither averaged nor broadened.
The tolerances classify boundary resolution; they do not regularize the
Sylvester equation. A small accepted external gap can still be ill-conditioned.

## Numerical ownership and derivative order

The new code lives in `solvers/eigen/subspace.py`. Forward eigenpairs come from
the existing shared dense/Davidson implementations. Response uses
`solvers.linear.checked_linear_solve` on flattened n-by-k blocks, including a
checked transpose solve. No method-local eigensolver or GMRES copy is added.
The augmented operator is

```text
K(Z) = Q [A (QZ) - (QZ) H] + PZ.
```

The PZ term enforces the frame constraint in the extension outside Q-space;
it does not shift the physical eigenvalues or the complement-space response.
Projector actions require O(nk) frame storage rather than an explicit n-by-n
projector. Davidson basis and GMRES workspaces additionally depend on their
configured subspace/restart sizes. Set `max_subspace` for bounded Davidson
storage; its legacy default can reach the full matrix dimension.

At an invalid boundary, the auxiliary zero-RHS primal response solve uses an
identity operator solely to preserve the inspectable Ritz-projector value.
Its derivatives remain invalid. This is not a regularized physical response.

Shared Davidson restart retention and accepted correction counts are bounded
by actual buffer capacity. When preconditioning produces a direction parallel
to the existing basis, the forward algorithm uses the projected original
residual instead. The physical eigenproblem and derivative equations are unchanged.
The generic real API scales its orthogonalization cutoff with the requested
residual tolerance and diagonal magnitude to avoid premature rejection of small
corrections. Legacy low-level calls retain their explicit cutoff controls.

The derivative attachment uses stopped forward Ritz vectors and a zero-valued
RHS with a live tangent, as in the existing isolated-vector implementation.
The contract is **first-order only**. Nested differentiation is not validated
and is not a subspace Hessian implementation. Existing `solve_hermitian` and
legacy isolated-root response APIs retain their previous derivative contracts; individual
state observables must not be routed through a projector by assumption.

See [the executable example](../../../examples/degenerate_subspace.py) and
[validation](VALIDATION.md) for the test coverage and numerical evidence.
