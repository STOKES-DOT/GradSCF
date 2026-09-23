# Shared real-orbital stability and stationarity diagnostics

`mf.diagnostics()` reports the stored restricted SCF state without rebuilding a
response reference or resolving SCF. `mf.stability()` analyzes its real,
restricted-internal orbital Hessian. Configuration changes or public MO arrays
that differ from the stored result are rejected as stale.

```python
stationarity = mf.diagnostics(gradient_tol=1e-11)
print(stationarity.gradient_norm, stationarity.stationary)
curvature = mf.stability()  # public shared Davidson solver by default
print(curvature.minimum_curvature, curvature.stable)
```

The SCF diagnostic gradient is the existing RKS residual `||2 F_vo||`, half the
Euclidean norm of the real occupied-virtual angle gradient. The report also
contains MO metric-orthogonality and electron-count errors, the SCF convergence
flag, and the requested gradient threshold. `stationary` requires convergence,
a finite sufficiently small gradient and consistent metric/electron checks.
Final energy/density changes are not invented: the SCF result does not retain
them. This report is not a bound on an excitation or ionization energy error.

## One curvature core, existing energy functions

RKS and UKS share occupation-based Cayley rotations, gradient/HVP construction,
the public `solve_hermitian` call, and true eigen-residual decisions. The energy
functions remain their original RKS and UKS implementations, including their
XC configurations. Native packed AO ERIs stay packed; no full four-index AO
array or physical orbital Hessian is required for Davidson. RKS currently
requires real integer 0/2 occupations, full/packed JK and unclipped HF/LDA/GGA
or global-hybrid potentials. DF/direct JK, meta-GGA and spin/complex restricted
instabilities are outside this first restricted adapter.

`OrbitalStabilityResult` reports:

- `stable`: False for a resolved negative curvature, True for a nonnegative
  lowest computed curvature, None for an unconverged/nonstationary/unresolved case;
- `minimum_curvature`, selected `eigenvalues` and true `residual_norms` in Ha
  per squared normalized orbital angle;
- `gradient_norm`: half the real angle-gradient norm used to check stationarity;
- `stationary`, `eigensolver_converged`, and `spectrum_certified`;
- `mo_coeff`: an existing-style unit Cayley displacement along a negative mode,
  or the original orbitals when no valid negative mode was found.

A successful dense solve (`solver="dense"`, shared capacity limit 2048)
certifies the finite Hessian spectrum; a failed residual check withdraws the
certificate. Davidson (`max_space=40`, `max_cycle=100` by default) reports
Ritz estimates and `spectrum_certified=False`, even when its residuals pass.
Positive partial Ritz values are not a global spectral certificate. Negative
curvature supplies a descent direction, but a stable internal RKS Hessian does
not test spin symmetry breaking or complex orbitals, or prove the global SCF
minimum. Primal curvature validity does not require an isolated-eigenvalue AD
flag: degenerate curvatures can be valid forward diagnostics.

The old unrestricted result names remain aliases of the same result type;
existing UHF/UKS stabilization retains its bounded escape loop and now calls the
same curvature core. The common rotation geometry is extracted from the existing
orbital optimization code, not independently reimplemented.

See [multistart filtering](MULTISTART.md) and
[the layered post-HF example](../../../examples/cc/precision_and_branch_checks.py).
