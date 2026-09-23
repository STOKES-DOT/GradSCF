# Iterative non-Hermitian EOM validation

2026-09-23, macOS arm64 CPU (`TFRT_CPU_0`), Python 3.12.2, JAX 0.8.1,
PySCF 2.9.0, float64, `OMP_NUM_THREADS=1`. Timings include compilation and
concurrent local validation work; they are not performance benchmarks.

## Native water/6-31G and independent reference spectra

Geometry (Angstrom): `O 0 0 0; H 0 -.757 .587; H 0 .757 .587`.
Each package runs its own RHF and CCSD. GradSCF uses native integrals;
HF tolerance 1e-12 Ha, CC energy 1e-12 Ha and residual 1e-11 Ha. PySCF uses
HF/CC energy tolerance 1e-13 Ha and CC amplitude tolerance 1e-12. No frozen
orbitals. EOM requests two roots plus one guard, `max_space=40`, `max_cycle=150`,
residual tolerance 1e-9 Ha, seed 0, orbital-difference preconditioning.

The comparison explicitly assembles each **PySCF reference** action matrix and
diagonalizes it with NumPy to establish the lowest roots. GradSCF uses only
iterative actions and has no physical dense matrix fallback.

| Sector | Physical dimension | Final search dimension | Iterations / restarts | Lowest two energies / Ha | Maximum reference difference / Ha |
| --- | ---: | ---: | ---: | --- | ---: |
| EE singlet | 860 | 38 | 19 / 2 | 0.308197287738, 0.391939286778 | 3.00e-11 |
| IP doublet | 205 | 18 | 16 / 2 | 0.427890833823, 0.502268611847 | 6.86e-11 |
| EA doublet | 328 | 30 | 18 / 2 | 0.190505920204, 0.283452288568 | 7.89e-12 |

`spectrum_complete=False` for all three solves. Maximum selected right/left
residuals respectively: EE 2.29e-11 / 1.84e-10 Ha; IP 6.26e-11 / 2.27e-10 Ha;
EA 1.03e-11 / 2.77e-11 Ha. Every guard residual is below 7.5e-10 Ha. Maximum
biorthogonality error is 7.02e-16. HF and CCSD total energy differences are
1.28e-13 and 7.26e-12 Ha. The native script took 31.66 s; the independent full
reference comparison took 32.37 s, both including compilation.

### Why the complete reference matters

For this exact PySCF 2.9.0 calculation, its default **two-root** singlet EE
iteration returns 0.308197287663 and 0.401631429116 Ha. The full reference matrix
has an intervening root at 0.391939286748 Ha. Requesting three roots finds it.
This is an observed iterative root-selection difference, not a disagreement
between the CCSD equations. The default two-root result is retained in the
comparison report rather than silently treated as the complete lowest spectrum.
Our full-support guesses find both lower roots here, but that does not prove
that any finite iterative search will always discover all lower states.

## Derivatives and memory

H2/STO-3G checks all three complete tiny spaces. Water/STO-3G checks all three
**incomplete** search spaces with `max_space=16`, at least one restart/expansion,
and residual tolerance 1e-10 Ha. In the water tests,
`h(t)=h+t*diag(linspace(-0.15,0.2,nmo))` and `g(t)=g*(1+0.015*t)` in a fixed
MO frame. CC is solved inside AD and at each finite-difference endpoint.
Iterative and dense energy derivatives agree at absolute tolerance 2e-7 Ha;
central differences at step 1e-4 use 3e-6 Ha tolerance. These are assertion
tolerances, not measured physical errors.

The final focused non-Hermitian suite passed 16 tests in 17.28 s after the
initial-guess and guard fixes. The complete regression
includes six larger/frozen-space EOM comparisons against PySCF for both dense
and Davidson solving.

Two failures were reproduced before their fixes: an exact requested root with
an unconverged guard incorrectly reported convergence, and exact canonical
guesses could terminate on an invariant sector above a hidden lower nonnormal
2x2 block. Guards now gate both convergence and response. Default diagonal
guesses receive independent full-support perturbations; if preconditioned
corrections all lose independence, raw residual expansion prevents stagnation.
Explicit user guesses remain unchanged, so local spectral diagnostics must
still be interpreted with care.

A 300-dimensional operator test forbids all action blocks wider than 24 while
setting `max_dense=4`. A separate 160-dimensional nonsymmetric operator test
recursively inspects forward and reverse JAXPRs for physical `(n,n)` tensors.
Neither dense probing nor a dense backward path is used. This does not bound
CC's integral/intermediate memory, which is unchanged by the eigensolver work.

## Final regression

`tests/solvers tests/cc/test_eom.py`: **119 passed in 407.79 s**, no skips or
failures, after the guess/guard fixes. This includes all existing shared solver
checks and the EE/IP/EA dense/iterative, frozen-space and derivative tests.
An additional constructor compatibility check passed (1 passed, 9 deselected,
0.80 s) after preserving the original positional configuration argument order.
The default mode remains dense and existing calls need no changes.

Independent read-only review covered the iterative core, shared energy response,
EOM forwarding and the guard/initial-guess fixes. Formatting, Python compilation,
documentation links and `git diff --check` also passed.

## Reproduce

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/cc/eom_iterative.py
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/cc/compare_eom_iterative_pyscf.py
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python -m pytest -q tests/solvers tests/cc/test_eom.py
```

The native example imports no PySCF. The comparison deliberately uses dense
reference matrices only in the separate validation script.

## Remaining boundaries

Dense remains the default. Davidson validates observed requested/guard residuals,
Ritz gaps and eigenpair conditioning. With an incomplete search space, its
`response_valid` is conditional on actual full-spectrum isolation. Root ordering
and exclusion of unseen degeneracy are not certified. Vector/cluster response,
higher derivatives, open-shell sectors, transition properties and GPU behavior
are not covered. No complete repository test-suite claim is made.
