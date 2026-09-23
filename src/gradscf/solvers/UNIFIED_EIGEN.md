# Unified Hermitian spectral response: P1–P3

Approved plan, implemented from `bef1be3` in the isolated
`feat/unified-eigen-response` worktree, 2026-09-23.

## Ownership and contract

- `eigen/primal.py`: one stopped numerical Ritz solve, guard roots, initial
  vectors, residuals, orthogonality/symmetry and explicit interval selection.
- `eigen/response.py`: one constrained Sylvester response. Rank one supplies
  an isolated eigenvector; higher rank supplies a private horizontal frame for
  invariant outputs. It is not a collection of individual eigenvector derivatives.
- `eigen/api.py`: `solve_hermitian` selects a differentiated observable through
  `EigenResponseConfig`, assembles outputs and checks spectral prerequisites.
- `linear/`: owns all forward/transposed checked response linear solves.
- CI, restricted/unrestricted TDA and TDA-BSE: construct physics and physical
  thresholds, then call the common interface; no local eigen-response engine.

```python
solver = EigenSolverConfig(nroots=2, method="davidson", max_subspace=12)
response = EigenResponseConfig(target="subspace",
    linear_config=LinearSolverConfig(rtol=1e-11))
out = solve_hermitian(operator, config=solver, response=response, probes=d)
loss = d @ out.projection + out.eigenvalue_sum
```

Changing target to `eigenpairs` requires isolated roots and exposes differentiable
`values`/`vectors`. `eigenvalues` avoids the vector-response solve. Subspace mode
sets both fields to None and exposes detached Ritz data as `raw_*` diagnostics.
EigenResult has additional diagnostics and optional outputs; access named
attributes rather than positional tuple unpacking when migrating clients.
It never constructs an invalid individual-state AD branch whose zero cotangent
could contaminate a valid subspace loss. Target and rank are static under JIT;
there is no gap-triggered change of differentiated quantity or root count.

## Implementation sequence and acceptance

1. Establish shared numerical Ritz diagnostics and response/result types.
2. Replace the bordered rank-one attachment with the rank-one case of the
   projected Sylvester equation already used for invariant subspaces.
3. Migrate the projector tests/example and method clients. Remove
   `solve_spectral_projector`, `SpectralProjectorResult`, `eigen/subspace.py`,
   and both standalone implicit Hermitian Davidson AD wrappers.
4. Check migration boundaries: TDA positive-root selection, false-convergence
   fixtures that now span a larger initial space, hidden invariant sectors,
   and RPA's squared-frequency versus physical-frequency gap units.

No compatibility forwarding files remain. EigenSolverConfig's AD fields are
still consumed by existing RPA facades and supported on Hermitian calls that omit
`response`; an explicit response cannot silently override nondefault settings.
This is configuration compatibility, not a second differentiation algorithm.

TDA's lower spectral cutoff is explicit (`value_min`), with numerical checks
on both sides. An absent upper guard only becomes a certified end of the
interval after full-space coverage. In dense RPA the inner squared-frequency
solve tests relative numerical resolution; the RPA layer retains its absolute
physical frequency-gap criterion. RPA rejects the Hermitian value_min option
until an RPA frequency-window contract is defined. RPA reads detached guard data rather than
requesting a differentiated guard eigenvector.

## Verification and recovery

Test rank-one equivalence, exact/near degeneracies and splitting perturbations,
rotated subspace bases, variable probes, cut clusters, absent guards, interval
cutoffs, unconverged primal/adjoint solves, one forward solve, hidden low modes,
and bounded matrix-free JIT/JVP/VJP/vmap execution. Preserve independent dense,
PySCF and stored QuAcK references in migrated tests. See VALIDATION.md for actual
commands/results; planned checks are not reported as passing before execution.

Keep the refactor on this isolated branch until these gates pass. If necessary,
revert the refactor commit; do not rewrite release history or modify unrelated
working-tree files. P4 aggregate optical APIs (including energy-weighted PAPd)
and P5 indefinite-metric degenerate RPA subspaces remain separate increments.
No new SCF gap regularization, non-Hermitian EOM response or higher derivatives
are introduced by this change.
