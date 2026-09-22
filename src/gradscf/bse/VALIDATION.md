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

## Remaining validation boundaries

No full-BSE X/Y property response, open-shell/periodic systems, GPU performance,
degenerate-cluster observables, higher derivatives, nuclear/basis gradients,
or complete evGW/qsGW fixed-point response is claimed. The G0W0+BSE AD test
perturbs factors in a fixed orbital frame; it does not differentiate through
an SCF orbital optimization. Resource caps bound selected array dimensions,
not measured peak memory.
