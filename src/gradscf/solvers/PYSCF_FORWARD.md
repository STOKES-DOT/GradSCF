# Unified eigensolver forward comparison with PySCF

Executed 2026-09-23 on the independent `feat/unified-eigen-response` worktree.
Production commit: `80845578c30af341e6ba051b3a56a24f056eba46`. PySCF 2.9.0, JAX 0.8.1, NumPy 2.3.4, Python 3.12.2; macOS-26.6.2-arm64-arm-64bit, CPU float64.

## Reproduction and numerical contract

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 /opt/anaconda3/bin/python tests/comparisons/compare_unified_eigen_pyscf.py
```

The fixed comparison suite completed with all assertions passing: **31 comparison records**, total wall time **26.95 seconds**. These are comparison records, not a count of pytest tests.

Raw values, molecular coordinates, basis choices, root lists, residuals and default-PySCF probe results are in [the JSON artifact](../../../reproducibility/solver_validation/unified_eigen_pyscf_cpu.json). The script is [compare_unified_eigen_pyscf.py](../../../tests/comparisons/compare_unified_eigen_pyscf.py).

- Cartesian orbitals; coordinates in Angstrom; energies and residuals in Hartree. Oscillator strengths are dimensionless, length gauge.
- SCF energy tolerance 1e-12, PySCF orbital-gradient tolerance 1e-9. Native GradSCF additionally uses density tolerance 1e-11 and gradient tolerance 1e-10.
- Main PySCF and GradSCF excitation residual tolerance 1e-11; maximum 200 cycles. GradSCF Davidson capacity min(dimension,40).
- The separate default-PySCF initial-guess probe uses residual tolerance 1e-9. PySCF lindep=1e-12.
- PBE/B3LYP use the installed PySCF functional definitions and level-1 integration grids. B3LYP uses the installed default VWN-RPA variant.
- OMP_NUM_THREADS=1 is an environment setting, not a claim that every JAX/XLA operation runs on one thread.

## Same response operator: isolate forward solver error

For each case, PySCF `gen_vind` builds the reference action. Its bounded matrix is passed to GradSCF dense and Davidson solves; the TDA method adapters are checked separately on the same action. The PySCF reference uses `x0=I` and `nstates=dimension` before selecting the requested low roots. This is deliberate: PySCF 2.9.0 truncates its initial block to an iteration-space increment depending on nstates, so passing I alone does not necessarily provide the full space. All reference roots are also checked against an independent NumPy diagonalization.

The table reports the maximum across both GradSCF solver methods. Cluster gaps <=1e-7 define comparison groups; roots are extended when needed to avoid cutting a group. Projector errors use the Frobenius norm and do not depend on signs or rotations inside a degenerate group.

| Case | Basis | Dimension / roots | Max energy error (Ha) | Max cluster strength error | Max cluster projector error |
| --- | --- | ---: | ---: | ---: | ---: |
| water HF singlet | 6-31g | 40 / 4 | 1.221e-15 | 1.130e-14 | 2.648e-13 |
| water HF triplet | 6-31g | 40 / 4 | 2.220e-15 | 0.000e+00 | 1.214e-13 |
| water PBE singlet | 6-31g | 40 / 4 | 1.110e-15 | 3.983e-15 | 8.542e-13 |
| water B3LYP singlet | 6-31g | 40 / 4 | 7.216e-16 | 4.552e-15 | 2.107e-13 |
| Be HF singlet | sto-3g | 6 / 3 | 9.936e-15 | 4.663e-15 | 1.354e-15 |
| Be HF triplet | sto-3g | 6 / 3 | 5.856e-15 | 0.000e+00 | 1.211e-15 |
| OH UHF | 6-31g | 58 / 4 | 1.388e-15 | 2.474e-15 | 5.957e-13 |

Restricted PySCF TDA amplitudes have squared norm 1/2: multiplying X by sqrt(2) gives the Euclidean eigenvector convention, while the corresponding dipole probes also carry sqrt(2). UHF uses the combined alpha/beta unit norm. Triplet electric-dipole strengths are zero. Be provides complete threefold-degenerate groups; both their projection actions and energy sums pass through the unified subspace API, with response_valid=True. No derivative benchmark is claimed here.

## Default PySCF roots require a completeness check

For this OH/6-31G UHF case, PySCF's default guesses returned a fourth root of **0.459879262169 Ha** with converged flags. The complete-basis reference gives **0.424447424902 Ha**, which GradSCF also finds. The discrepancy is **0.03543184 Ha**; the default fourth return corresponds to a higher member of the qualifying spectrum. This is a missing-root issue for the observed initial space, not a discrepancy between the two solvers on the same requested eigenpair.

Selection is above the configured 1e-3 Hartree threshold, not simply above zero. The raw artifact retains the default result as well as the complete reference. This observation is specific to the recorded setup; it is not a claim that all PySCF default calculations miss roots.

## CI with matched MO integrals

GradSCF constructs its own determinant Hamiltonian from common PySCF MO integrals, then solves with both dense and Davidson. PySCF CISD/UCISD uses conv_tol=1e-13 and max_cycle=200. Restricted coefficient arrays use different representations and are not directly compared; energy and one-particle densities are compared instead.

| Case | Max total-energy error (Ha) | Max correlation-energy error (Ha) | Max 1-RDM element error |
| --- | ---: | ---: | ---: |
| H4 CISD | 7.550e-15 | 7.508e-15 | 3.669e-09 |
| LiH frozen-core CISD | 9.770e-15 | 8.396e-15 | 1.100e-14 |
| H3 UCISD | 7.550e-15 | 8.472e-15 | 4.479e-08 |

## Independent native HF to TDA chains

Here both programs run their own HF and construct their own response, on matched STO-3G geometries. PySCF uses the complete-basis TDA reference; GradSCF uses its native HF/TDA facade.

| Case | SCF energy error (Ha) | Max excitation error (Ha) | Max strength error |
| --- | ---: | ---: | ---: |
| native water RHF/TDA | 0.000e+00 | 3.484e-11 | 2.413e-11 |
| native H3 UHF/TDA | 2.220e-16 | 3.265e-10 | 3.176e-10 |

## Discovered facade defect and regression checks

The native UHF facade initially failed before diagonalization: its HF string was sent to a spin semilocal functional class that only accepted LDA/GGA. Commit `8084557` adds the exact zero semilocal HF HVP while retaining exact_exchange_fraction and the existing nonlocal exchange action. It does not replace exchange with zero.

The new native UHF TDA/TDHF and zero-HVP tests failed before the fix. After matching PySCF SCF gradient convergence, the narrow regression command passed **13 tests in 8.20 seconds**:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 /opt/anaconda3/bin/python -m pytest -q tests/test_native_uhf_response.py tests/test_pyscf_style_excited_state_api.py --tb=short
```

Two additional pre-existing B3LYP spin-HVP tests were attempted and failed because the optional jax_xc package is absent in this environment. They are not counted as passed or silently treated as skipped. No dependency was installed for this comparison.

## Scope limits

- PBE/B3LYP rows validate the unified solver and method adapters on a common PySCF response matrix. They do **not** validate independently assembled native GradSCF DFT kernels or quadrature.
- The CI rows use matched MO integrals, whereas the two native HF/TDA rows include independently converged SCF and response construction.
- These are finite molecular CPU/float64 forward checks. They do not establish GPU agreement, large-system scaling, gradients, general eigenvector uniqueness at degeneracy, or comprehensive GW/full-BSE correspondence to PySCF.
- The benchmark uses bounded complete reference spaces to establish the intended roots; default small-subspace iterative results are not assumed complete merely because residuals are small.
