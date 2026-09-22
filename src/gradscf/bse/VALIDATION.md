# Static molecular TDA-BSE validation

Executed on 2026-09-22, macOS 26.6.2 arm64, Python 3.12, JAX 0.8.1 CPU,
float64, PySCF 2.9.0. These checks validate the P0/P1 scope in
[README.md](README.md); they do not establish full-BSE or GPU support.

## Automated checks

From the repository root, using `/opt/anaconda3/bin/python`:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q \
  tests/bse tests/solvers \
  tests/gw/test_gw_qp.py tests/gw/test_gw_differentiability.py \
  tests/gw/test_gw_evgw.py tests/gw/test_gw_cd_rgw.py \
  tests/gw/test_gw_cd_ugw.py tests/test_pyscf_style_excited_state_api.py
```

Result: **106 passed**, no skips, 8 warnings, 182.58 seconds. This broader
regression preceded the final cache and auxiliary-block boundary refinements.
After those refinements, the final affected-area run was:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q \
  tests/bse tests/solvers/test_block_linear.py --tb=short
```

Result: **29 passed**, no skips, 1 warning, 38.63 seconds. Counts overlap and
must not be added. Warnings are JAX `ComplexWarning` messages from the existing
GW complex-to-real reverse pass; the corresponding finite-difference check
passes. The complete repository suite was not run.

Coverage includes:

- Independent NumPy dense-loop kernels versus matrix-free actions, diagonals,
  singlet/triplet roots and strengths; bounded dense versus Davidson results.
- Separate QP/screening spectra and optical/screening windows; no-screening
  HF/CIS limit versus PySCF, and the independent-particle limit.
- First derivatives through QP energies, screening, factors, isolated
  eigenvectors and dipoles, including a fixed-MO G0W0+BSE factor perturbation.
- Invalid QP coverage/convergence, gaps, negative modes, root nonconvergence
  and exact degeneracies; energy-only mode rejects incomplete property AD.
- Eager source freshness, failed-rerun invalidation, lazy dipole completion,
  capacity rejection before MO transformation, and oversized block requests.
- Shared direct multiple-RHS solves, nonsymmetric transpose response and
  failed-solve diagnostics.

## Executed QuAcK kernel oracle

The comparison script compiles unmodified QuAcK `phRLR_A.f90` and
`RGW_phBSE_static_kernel_A.f90` with GNU Fortran 15.1.0. Revision:
`2236bfcda24ff0971358f5107636b506dd201bb1`. Source SHA256 hashes, compiler flags,
backend and errors are recorded in
[the fixture metadata](../../../tests/bse/data/quack_static_seed83.json).

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  /opt/anaconda3/bin/python tests/comparisons/compare_quack_bse.py
```

The successful rerun took 2.70 seconds using cached, SHA256-verified upstream
files. Download attempts during this rerun failed; the cached sources came
from the earlier successful retrieval. Ordinary pytest uses the committed
fixture and does not require a network connection or Fortran compiler.

The synthetic reference has seed 83, 5 auxiliary functions, 5 spatial orbitals
and 2 occupied orbitals, with distinct QP and screening energies. Screening
uses an independent NumPy **full direct-RPA pole expansion**, eta=0 and
lambda=1. It does not use TDA screening. QuAcK supplies the bare response
matrix and static screening correction; NumPy supplies poles and transition
densities. This tests independent kernel algebra, not an external SCF/GW chain.

| Maximum absolute error (Hartree) | Singlet | Triplet |
| --- | ---: | ---: |
| TDA matrix | 2.22e-16 | 2.22e-16 |
| Excitation energies | 6.66e-16 | 3.55e-15 |
| Screening correction | 1.54e-16 | 1.60e-16 |

The fixture tests use 2e-13 Hartree for matrices and 2e-12 Hartree for roots.
Attribution and primary source
links are in [REFERENCES.md](REFERENCES.md). MOLGW and VOTCA were consulted
for planning but were not executed as numerical references.

## Native water example

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python examples/bse/water_tda.py
```

Geometry in Angstrom: O (0,0,0), H (0,-0.757,0.587), H (0,0.757,0.587).
STO-3G, native GradSCF RHF (`xc="hf"`), SCF tolerance 1e-12, G0W0 frequency
grid `nw=100`, default GW eta=1e-3 Hartree. All seven QP levels and the full
screening space are retained; BSE uses three roots and tolerance 1e-10 Hartree.

| Root | Singlet energy (eV) | Singlet oscillator strength | Triplet energy (eV) |
| --- | ---: | ---: | ---: |
| 1 | 12.8063271021 | 0.00343541344 | 10.7065034901 |
| 2 | 15.0149160453 | 1.01e-28 | 13.4388398840 |
| 3 | 16.7514947481 | 0.0795011079 | 13.6898877491 |

All QP levels converged. Maximum reported TDA residual was 4.25e-15 Hartree;
triplet oscillator strengths were zero. Example wall time was not recorded.

At `nw=40`, one valence QP root retained a Dyson residual of 3.04e-4 Hartree
and the BSE adapter correctly rejected it. At `nw=100`, the maximum QP
residual was 2.39e-10 Hartree. A separate `nw=160` check converged all QP
levels and changed the first three singlet/triplet excitations by a few
1e-8 Hartree (comparison from printed eight-decimal Hartree values).
This is a small-basis integration example, not an accuracy benchmark against
experiment or a complete external molecular GW+BSE package.

## Original P0/P1 validation boundaries

At the original P0/P1 checkpoint, no full-BSE X/Y property response,
open-shell/periodic systems, GPU performance,
degenerate-cluster observables, higher derivatives, nuclear/basis gradients,
or complete evGW/qsGW fixed-point response is claimed. The G0W0+BSE AD test
perturbs factors in a fixed orbital frame; it does not differentiate through
an SCF orbital optimization. Resource caps bound selected array dimensions,
not measured peak memory.

## Full-BSE and GW-provenance increment (2026-09-22)

This increment adds the bounded stable full-BSE reference and first-order X/Y
response described in [FULL_BSE_PLAN.md](FULL_BSE_PLAN.md). The backend remains
macOS arm64, CPU, JAX 0.8.1, float64. The full regression command in the first
section now ran **129 tests: all passed**, no skips, 8 existing GW complex-cast
warnings, 208.84 seconds. This preceded a final reduced-residual diagnostic
refinement and two additional solver regressions; see the final affected-area
verification below. Test counts from separate runs overlap.

Executed new checks cover the coupling block, full-BSE energies and oscillator
strengths against independent doubled matrices, positive-definiteness and
metric normalization, B=0 and independent-particle limits, native H2 G0W0/evGW
screening provenance, the water HF+W=v limit against PySCF TDHF for both spins,
and joint first-order QP/screening/factor/dipole response. TDHF comparison uses
water STO-3G, the example geometry, HF tolerance 1e-13, TDHF tolerance 1e-11,
and absolute comparison tolerance 2e-9 Hartree (energy) / 2e-9 (strength).

Review and diagnostic regressions address two cases:

- A request mixing isolated and degenerate roots now invalidates the whole
  requested derivative set, consistently for JVP and VJP. A smaller isolated
  prefix recovers finite response. Forward stable roots remain available.
- The reduced squared-frequency problem and reconstructed physical problem
  must both pass their solver residual checks. Large energy scales can give
  a small physical residual but a failed absolute reduced residual; this no
  longer reports valid response while the inner solver rejects its derivative.

### Additional executed QuAcK B-block oracle

The same opt-in comparison script now also compiles and executes unmodified
`phRLR_B.f90` and `RGW_phBSE_static_kernel_B.f90` at the same pinned revision.
All six reference sources are checked by SHA256. The new files downloaded
successfully; `phRLR_B.f90` required a retry after a timeout. The existing
A-block source cache was reused. The four actual Fortran A/B kernels run with
the same independent NumPy full direct-RPA screening poles as before.

Metadata and arrays: `tests/bse/data/quack_full_static_seed83.json` and `.npz`.
Command: the `compare_quack_bse.py` invocation above. Elapsed time: 4.60 seconds.
Full-BSE oracle eigenvalues are obtained by NumPy diagonalization of the doubled
matrix made from the Fortran blocks; no QuAcK eigensolver or complete external
molecular GW+BSE driver was run.

| Maximum absolute error (Hartree) | Singlet | Triplet |
| --- | ---: | ---: |
| Coupling B matrix | 6.94e-18 | 6.94e-18 |
| Full-BSE excitation energies | 2.66e-15 | 5.33e-15 |

Normal pytest now checks both the original TDA fixture and the new full-BSE
fixture offline, at 2e-13 Hartree for B and 2e-12 Hartree for energies.

### Native water full-BSE example

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python examples/bse/water_full.py
```

Same RHF/STO-3G geometry and nw=100 G0W0 setup as the original TDA example.
All QP levels converged. `tda=False, solver="dense"`, three roots, physical
residual tolerance 1e-10 Hartree. Example wall time was not recorded.

| Root | Singlet energy (eV) | Singlet strength | Triplet energy (eV) |
| --- | ---: | ---: | ---: |
| 1 | 12.75085069 | 0.00311599129 | 10.66783729 |
| 2 | 14.99788371 | 3.82e-29 | 13.25957339 |
| 3 | 16.56916724 | 0.0681077152 | 13.67628826 |

Triplet strengths are zero. Maximum full-BSE physical residual: 1.72e-14
Hartree. Minima of (A-B,A+B), Hartree: singlet (0.42686559,0.51438209), triplet
(0.42686559,0.36004790). These certify the specified static BSE matrices only.
The small-basis values are integration checks, not experimental benchmarks.

Current boundaries: full BSE is dense and bounded, requires stable real
closed-shell input, and supports first-order isolated-state response only.
Matrix-free full-BSE response, degenerate cluster properties, open-shell and
periodic kernels, higher/nuclear derivatives, and outer evGW/qsGW AD remain
unverified or unimplemented. There is no new scGW spectral continuation.

### Final affected-area verification

After the reduced-residual validity fix and additional solver regressions:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python -m pytest -q tests/bse tests/solvers --tb=short
```

**95 passed**, no skips, one existing GW complex-cast warning, 133.84 seconds.
This overlaps the 129-test run above; counts are not additive. The full repository
suite and GPU backends were not run. Independent code review found no remaining
important issue within this bounded real stable full-BSE scope after the
mixed-degeneracy validity fix.
