# Shared real-orbital stability and stationarity diagnostics

`mf.diagnostics()` reports the stored restricted SCF state without rebuilding a
response reference or resolving SCF. `mf.stability()` analyzes its real,
restricted-internal orbital Hessian by default. `channel="spin"` checks opposite
alpha/beta rotations of a restricted state. Configuration, public MO or energy
values that differ from the stored result are rejected as stale.

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
or global-hybrid potentials. DF/direct JK, meta-GGA and complex/generalized instabilities remain outside
these adapters.

`OrbitalStabilityResult` reports:

- `stable`: False for a resolved negative curvature, True for a nonnegative
  lowest computed curvature, None for an unconverged/nonstationary/unresolved case;
- `minimum_curvature`, selected `eigenvalues` and true `residual_norms` in Ha
  per squared normalized orbital angle;
- `gradient_norm`: half the real angle-gradient norm used to check stationarity;
- `stationary`, `eigensolver_converged`, and `spectrum_certified`;
- `channel` and `direction`: the requested channel and a resolved negative
  orbital-angle direction, or None if no valid negative direction exists;
- `mo_coeff`: a unit Cayley displacement along that mode, or the original
  orbitals when none was found. Restricted spin-channel output has alpha/beta
  coefficient matrices; internal restricted output remains a single matrix.

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
existing UHF/UKS stabilization and the new facade entry points use one shared
bounded negative-mode following loop and the same curvature core. The common rotation geometry is extracted from the existing
orbital optimization code, not independently reimplemented.

See [multistart filtering](MULTISTART.md) and
[the layered post-HF example](../../../examples/cc/precision_and_branch_checks.py).

## Restricted spin channel

```python
internal = mf.stability()
spin = mf.stability(channel="spin")
```

Spin coordinates use `(theta_alpha, theta_beta)=(theta,-theta)`. Their energy is
the existing unrestricted functional at the restricted reference, with matched
configuration and packed integrals. Full unrestricted stationarity is checked:
a vanishing spin-projected gradient alone cannot qualify a nonstationary charge
state. The reference gradient norm is scaled back to the restricted half-gradient
convention. The two models' reference energies must agree before comparison.

In these coordinates the physical Hessian is four times PySCF's RHF-to-UHF
response action; tests verify the factor on H2 and H2O. No PySCF numerical solver
is called by production code. This covers real collinear spin breaking at fixed
alpha/beta particle counts, not spin-flip GHF or complex instabilities.

## Explicit bounded stabilization

```python
out = mf.stabilize(channel="internal", max_restarts=5, step_sizes=(0.5, 1.0))
print(out.status, out.energy_history, out.attempts)

# Explicitly permit RKS/RHF -> UKS/UHF only when a spin instability is found.
spin_out = mf.stabilize(channel="spin", max_restarts=5)
```

A fresh solved source is copied and analyzed without repeating its SCF solve.
An unsolved source is solved on a new object first. The original is preserved.
Common input controls are copied across a requested RKS->UKS transition; solved
payloads/caches are not. A stable restricted spin check can return an unchanged
restricted model. After an accepted spin-breaking trial, the full real UKS
space is checked, not only the original opposite-spin direction.

One engine tries both signs of each supplied positive step magnitude. It accepts
only finite, converged SCF candidates lower by more than `energy_tol` (default
1e-10 Ha), then chooses the lowest candidate from that round. Every trial is
recorded with round, signed step, energy, convergence and a chosen/accepted flag.
`restarts` counts attempted rounds; `energy_history` contains only the initial
and accepted states and is strictly decreasing. Accepted=False can mean a valid
lower candidate lost to a still lower one, not necessarily that SCF failed.

Termination status is `stable`, `unresolved`, `max_restarts`, or
`no_lower_converged_candidate`. Unsuccessful trials never overwrite the current
state. Step sizes and round counts are finite, explicit budgets; no automatic
random retries or relaxed thresholds are introduced. Configuration/backend
exceptions propagate. Existing low-level UHF/UKS integral APIs use this same
engine. Negative-mode following is a discrete host workflow, not an AD rule or
a global-minimum guarantee.

A spin-broken reference is not silently substituted into closed-shell EOM-CCSD;
that interface continues to reject unrestricted inputs. A mean-field spin
instability also does not by itself invalidate a deliberately chosen restricted
post-HF calculation. See [the native example](../../../examples/cc/stability_following.py)
and [validation](STABILITY_FOLLOW_VALIDATION.md).
