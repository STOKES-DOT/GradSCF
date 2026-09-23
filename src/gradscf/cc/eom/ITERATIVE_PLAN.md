# Shared iterative non-Hermitian EOM solve

Approved direction: replace the small-system diagonalization bottleneck with a
shared non-Hermitian Davidson solver, retaining the dense oracle. Continue on
feat/eom-cc from 5a4fade; no integration/push in this increment.

Use a restarted real orthonormal search space enriched by both right and left
Ritz residuals. Project A without symmetrization. Complex conjugate Ritz pairs
contribute real and imaginary directions; selected complex roots remain invalid
real physical outputs. Independent small full-support perturbations prevent exact-invariant default
diagonal guesses from terminating immediately; seeded exploration supplements
them. Explicit user guesses are preserved. Stalled preconditioned directions
fall back to raw residual expansion.
Optional diagonal preconditioning is shared numerical logic, not EOM physics.
A plain unpreconditioned residual expansion covers operators without a diagonal.
Separate right/left iterative solves were considered but complicate root pairing;
a bi-Lanczos recurrence was rejected for this stage because of breakdown risk.

Keep dense as the default oracle; select method='davidson' explicitly. Bound
storage by n*max_space + max_space**2. Never assemble the n*n physical matrix or
fall back to dense. Expose maxiter, max_space, guard_roots and seed. JAX loops use
fixed shapes; numerical iteration is stopped for AD. The existing first-order
energy rule L.T(dA)R remains shared and retains outer CC amplitude response.

Convergence requires true right/left residuals and converged guard Ritz roots.
Report spectrum_complete, iteration count, subspace dimension, spectral gaps
and guard residuals. Iterative gaps are local Ritz estimates, not a certificate
of global root ordering or isolation: no general matrix-free algorithm can
exclude an unseen invariant sector from finite sampling. Reject observed
unresolved degeneracies, complex modes and bad conditioning. Eigenvectors,
cluster response and higher derivatives remain outside the energy-only contract.

EOM owns EE/IP/EA actions and their JAX transposes. Remove the dense dimension
cap only when solver='davidson'; keep integral capacity checks. No diagonal
probing that materializes a physical identity; the initial EOM path may use
unpreconditioned residual expansion. All three sectors use the same solver.

Acceptance: bounded-memory operator and JAXPR checks; JIT/JVP/VJP vs dense and
finite differences; restarts and failures; complex/degenerate cases; full-support
exploration; EE/IP/EA actions, native examples and PySCF energies; focused and
existing numerical regressions. Document incomplete spectral certification.
