# Full static BSE reference and GW screening provenance

Continuation of the molecular closed-shell GW+BSE plan, 2026-09-22, based on
`b2568aa`. This increment completes a bounded full-BSE reference with amplitude
response, not the scalable structured-Davidson part of the original P2 stage.

## Chosen implementation

Add the static coupling action
`B[ia,jb] = kappa(ia|jb) - W[ib,aj]` in the BSE physics module, reusing the same
full W and screening/QP windows as the resonant block. The shared numerical
solver owns forward and backward response. The existing legacy RPA energy-only
entry points remain available to their callers with unchanged behavior.

Three numerical paths were considered: directly differentiating the doubled
nonsymmetric eigensystem; extending structured Davidson with an indefinite-metric
amplitude adjoint; and a stable Cholesky-Hermitian reference. The last is selected
for this increment because it supplies a small independently testable reference,
certifies stability, and reuses the common Hermitian vector-response contract.
The structured extension remains the next scaling task. Full BSE requires
`tda=False, solver="dense"`, with an explicit transition-space cap.

Let `M=A-B`, `N=A+B`. For positive-definite M and N, set `M=L L.T` and solve

```text
C = L.T N L
C z = omega**2 z,       z.T z = I
X+Y = L z / sqrt(omega)
X-Y = solve(L.T,z) sqrt(omega)
X.T X - Y.T Y = I
```

AD includes Cholesky, both matrix products, eigenvector response, square-root
frequency conversion and triangular reconstruction. Repeated eigenvalues of M
alone do not imply a singular Cholesky derivative. The returned physical roots
must be isolated. Both reduced and doubled physical residuals are checked;
stability minima and metric normalization are separate diagnostics.

The two minimum eigenvalues of A-B/A+B certify the *specified static BSE
problem*, not a separate SCF stability analysis. Nonpositive/unresolved margins
are invalid; there is no absolute-value square root or filtered unstable mode.
If a requested root is non-isolated, the whole requested set has invalid
first-order response, avoiding inconsistent JVP/VJP validity flags in a vector
containing NaN derivative guards. A smaller isolated prefix remains usable.

## GW data flow

CD GW results record the actual pole spectrum used in W as `screening_energy`.
G0W0 records its starting poles; converged evGW records the final fixed-point
spectrum. Uncomputed QP levels retain explicit false coverage masks. A new
`BSEReference.from_gw_result` constructor requires that provenance and caller-
supplied factors/dipoles in the returned MO frame. It does not infer an scGW QP
spectrum or add evGW/qsGW outer differentiation.

## Acceptance and deferred work

Validate A/B actions against independent scalar-loop formulas and pinned QuAcK
Fortran; full roots against the independently diagonalized doubled matrix;
B=0/one-mode/independent-particle limits; HF+W=v against PySCF TDHF; metric
normalization; isolated-state energy and oscillator-strength JVP/VJP against
reconverged finite differences; instability, invalid-input, degeneracy and
capacity diagnostics; G0W0/evGW provenance; and existing TDA/GW/solver regressions.

Deferred: matrix-free full-BSE amplitude adjoints, complete degenerate-cluster
observables, complex/open-shell/periodic kernels, large-auxiliary screening,
full outer GW response, higher derivatives, and nuclear/basis differentiation.
Actual measurements are reported in [VALIDATION.md](VALIDATION.md).
