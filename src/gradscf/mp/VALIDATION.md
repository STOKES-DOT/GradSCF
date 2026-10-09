# Initial canonical molecular MP validation

Date: 2026-10-09. Local macOS arm64, CPU backend, JAX float64,
Python 3.12.2, JAX 0.8.1, PySCF 2.9.0. Native forward integrals reuse the
same built library as the parent checkout; no vendor or native-kernel change.

## Focused tests and shared-code regressions

```bash
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q \
  tests/mp tests/cc/test_ground.py tests/cc/test_ucc.py tests/ci/test_ci.py \
  tests/test_pyscf_style_namespace.py tests/test_integral_namespace.py
```

Result: **120 passed**, 229.20 s. This includes 37 MP tests and existing CI/CC
regressions for the generalized selected-block MO transformation. No dependency
tests were skipped. One PySCF warning reported that OpenMP is unavailable in
the installed build; it did not affect the assertions.

- RHF H2O/Cartesian 6-31G*: all-electron, frozen core and frozen core/virtual
  MP2 energies, E2 spin components and amplitudes match PySCF. Energy tolerances
  are 1e-11 Ha (correlation) and 1e-10 Ha (total); amplitudes 2e-9.
- UHF H3/STO-3G, spin=1: MP2 with no freezing and different alpha/beta frozen
  spaces matches PySCF under the same tolerances. Reference SCF uses
  `conv_tol=1e-14`, `conv_tol_grad=1e-11`.
- GradSCF RHF water/STO-3G and UHF H3/STO-3G: selected packed/DF facade results
  match full-MO array reference results at 1e-12 Ha; monkeypatched full-MO
  transforms must not be called by MP2. RHF energy-only and stored-amplitude
  modes are both exercised.
- MP3 H2/3-21G and LiH/STO-3G: individual E2 and E3 match independent determinant
  perturbation coefficients at 1e-11 Ha, including frozen core/virtual cases.
- JIT energy derivatives use an interaction-scaling identity with a fixed Fock
  operator, preserving the canonical-input condition. MP2/UMP2 HVPs and MP3
  gradients are checked. The MP3 HVP assertion added to that same test was then
  rerun with the MP3 file, separately from the 120-test run: **11 passed**,
  25.19 s.
- Invalid denominators and noncanonical states are rejected; Pauli-forbidden
  same-spin entries are excluded from minimum-denominator diagnostics. Empty
  doubles spaces return valid zero correlation energy. Stale facade sources
  raise and clear previous successful result/status.

## Public examples

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/mp/molecular.py
```

H2O geometry in Angstrom: O (0,0,0), H (0,0.75,0.58), H (0,-0.75,0.58).
Cartesian 6-31G*, frozen=1; native RHF, `conv_tol=1e-12`,
`conv_tol_grad=1e-10`. DF uses the default `def2-universal-jkfit` auxiliary basis.

| Quantity | Hartree |
| --- | ---: |
| RHF total | -76.010721112637 |
| E2 correction | -0.185314572684 |
| E3 correction | -0.005996691699 |
| MP3 total | -76.202032377021 |
| DF-MP2 total | -76.196039459056 |

UHF H3/STO-3G at z=(0,0.85,1.9) Angstrom, spin=1, all-electron:
UMP2 total -1.562947444299 Ha.

`examples/mp/implicit_gradient.py` is exercised by the test suite. H2/STO-3G
at 0.9 Angstrom, dimensionless scaling of one primitive contraction coefficient
on the first atom; shared implicit orbital HF response; centered reconverged
finite difference step 1e-4, `atol=2e-7`, `rtol=1e-5`.

| Method | Total / Ha | AD derivative / Ha | FD derivative / Ha |
| --- | ---: | ---: | ---: |
| MP2 | -1.109271845527 | 0.025838166 | 0.025838166 |
| MP3 | -1.116349572980 | 0.025318695 | 0.025318695 |

The example computes native primitive integrals once and differentiates JAX
contraction, implicit HF orbitals, MO integrals and MP energy. It does not prove
nuclear-coordinate gradient coverage. Output comments in both examples are
these measured values, not external reference tables.

## Boundaries

No entire-project test run, GPU execution, GPU peak-memory measurement or
large-system scaling benchmark was performed. MP3 remains in-core; UMP3,
MP4/higher orders, noncanonical and ROHF variants, and density matrices are not
implemented in this version. See README.md for result/amplitude semantics.
