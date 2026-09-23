# Initial EOM-CCSD validation

Run date: 2026-09-23. macOS arm64 CPU (`TFRT_CPU_0`), Python 3.12.2,
JAX 0.8.1, PySCF 2.9.0, float64, `OMP_NUM_THREADS=1`. Timings include JAX
compilation and are validation timings, not performance benchmarks.

## Reproduction

From the repository root, using the environment with the built native CPU
integrals library:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python -m pytest -q tests/cc/test_eom.py tests/solvers/test_nonhermitian.py
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python -m pytest -q tests/cc tests/solvers
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/cc/eom_ccsd.py
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/cc/eom_response.py
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/cc/compare_eom_pyscf.py
```

Tests enable float64 through `conftest.py`; examples enable it before creating
arrays. PySCF is optional in production but required for the oracle tests.

## Independent water spectrum

Each package ran its own RHF -> CCSD -> EOM chain. Water/STO-3G, Angstrom:
`O 0 0 0; H 0 -.757 .587; H 0 .757 .587`. GradSCF uses native integrals and
HF via `dft.RKS(xc="hf")`; HF energy tolerance 1e-12 Ha, CC energy tolerance
1e-12 Ha, amplitude residual tolerance 1e-11 Ha. PySCF HF/CC energy tolerance
1e-13 Ha and CC amplitude tolerance 1e-12; EOM tolerance 1e-11. GradSCF dense
EOM checks residuals at 1e-9 Ha. Three roots per sector, no frozen orbitals.
Total comparison time was **25.56 s**, including compilation.

| Sector | GradSCF roots / Ha | Maximum PySCF energy difference / Ha | Maximum right / left residual / Ha |
| --- | --- | --- | --- |
| EE singlet | 0.456622882848, 0.541104173311, 0.598609834708 | 4.12e-10 | 1.16e-14 / 1.30e-14 |
| IP doublet | 0.309287455184, 0.401117136947, 0.610740905107 | 8.75e-10 | 1.41e-14 / 1.17e-14 |
| EA doublet | 0.603042022771, 0.727608619376, 1.050180235380 | 2.78e-10 | 5.63e-15 / 8.68e-15 |

HF energies agreed at printed precision; the CCSD total energy difference was
7.11e-13 Ha. Maximum `||L.T R-I||` was 3.15e-16. All requested isolated-energy
response flags were true. EA values use E(N+1)-E(N), not conventional affinities.
These small-basis checks do not establish physical accuracy or continuum states.

## First-order response

Native H2/STO-3G at 0.74 Angstrom, fixed MO frame. The example perturbs
`h(t)=h+t*[[0.2,0.04],[0.04,-0.1]]`, with dimensionless t and the matrix in
Hartree. ERIs are fixed; CC amplitudes are recomputed within the differentiated
function. CC tolerances are 1e-12 Ha energy and 1e-11 Ha residual. Central
finite differences use step 1e-4 and independently reconverge CC at both points.

| Sector | d omega / dt, AD / Ha | Absolute AD–FD difference / Ha |
| --- | --- | --- |
| EE | -0.292400324113787 | 8.47e-12 |
| IP | -0.192400324113787 | 8.84e-12 |
| EA | -0.092400324113787 | 8.94e-12 |

Tests additionally perturb ERIs together with h, compare JIT reverse mode and
forward JVP, and verify invalid ground states cannot return valid derivatives.

## Regression coverage

The complete `tests/cc tests/solvers` suite passed **194 tests in 578.56 s**
(9 min 38 s), with no skips or failures. This covers the existing restricted,
unrestricted and QCI ground-state methods, properties/triples, and shared
linear, nonlinear, Hermitian, RPA and non-Hermitian solvers. The whole repository
suite was not run.

The focused suite passed **25 tests in 98.79 s** after the full-doubles guard
fix. An additional condition-limit regression was subsequently added and is
included in the complete suite recorded below. The standalone non-Hermitian
suite passed all 6 tests in 4.59 s. Extra transpose-action and empty/capacity
assertions were checked separately: 4 passed, 16 deselected in 12.82 s.

Action tests use H2, asymmetric H4 (z = 0, 0.8, 1.9, 3.1 Angstrom), and LiH
(1.6 Angstrom), all STO-3G. LiH freezes occupied 0 and virtual 5. Every action
column is compared with PySCF after converting packing conventions; absolute
tolerances are 2e-10 Ha for H2 and 5e-9 Ha for larger cases. Energies are also
compared with independent NumPy eigensolutions of the PySCF action matrix.
H2 IP/EA energies match exact FCI particle-number differences at 2e-10 Ha.

Other checks cover biorthogonal left/right residuals, complex/degenerate/Jordan
roots, capacity limits, nonnormal conditioning, selected versus excluded
degeneracy, negative EA roots, stale eager state, wrong ground models and
complete doubles finiteness/symmetry. The latter was first observed failing in
all three sectors, then fixed; an upper-triangle modification can no longer
escape packed-coordinate validation.

GPU, large basis sets, open-shell references, vector/cluster response,
second derivatives, transition properties and nuclear gradients were not
validated and are outside this increment.

## Public API environment boundary

The public example/ground/excited-state API checks produced **20 passed and
2 failed in 2.80 s**. Both failures are the real RKS H2/water DFT smoke tests:
the current environment lacks optional `jax_xc`. Running those exact two tests
in the unchanged base checkout reproduced both failures (2.53 s). They are not
EOM or HF failures; no substitute XC implementation or dependency installation
was introduced.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_examples_public_api_usage.py \
  tests/test_pyscf_style_ground_state_api.py \
  tests/test_pyscf_style_excited_state_api.py
```
