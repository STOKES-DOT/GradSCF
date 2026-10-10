# Canonical molecular MP validation

## Residual/Taylor extension, 2026-10-10

Local macOS arm64 CPU, Python 3.12.2, JAX 0.8.1, PySCF 2.9.0, float64.
The generic engine is described in [SERIES.md](SERIES.md). The working branch
uses the same native CPU library as the parent checkout; no integral-kernel or
shared-solver source changes are included.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /opt/anaconda3/bin/python -m pytest -q \
  tests/mp tests/ci/test_ci.py tests/ci/test_uci.py tests/test_integral_namespace.py -x
```

Result: **127 passed**, 201.39 s. No skips. The one warning is PySCF's
unavailable OpenMP build. This includes restricted individual E2--E6 and complete
state coefficients on H2, LiH and asymmetric H6/STO-3G; unrestricted E2--E5 on
Li/6-31G; frozen spaces; JIT outputs; integral gradients/HVPs; capacity preflight;
empty spaces; invalid response NaNs; and full implicit HF-to-MP4 contraction
parameter response against reconverged finite differences. The reference uses
PySCF full-space Hamiltonian actions and an independent RS recurrence, not
Taylor AD or GradSCF CI connections. It is a numerical oracle, not a production
FCI calculation used by the MP engine.

The subsequent water public-example regression passed separately:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /opt/anaconda3/bin/python -m pytest -q \
  tests/mp/test_series.py::test_water_public_example_matches_independent_pyscf
```

Result: **1 passed**, 8.98 s. The example also ran directly with assertions:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /opt/anaconda3/bin/python examples/mp/compare_water_pyscf.py
```

Water: O (0,0,0), H (0,0.75,0.58), H (0,-0.75,0.58) Angstrom, Cartesian
STO-3G, all electrons, real RHF with `conv_tol=1e-13`, `conv_tol_grad=1e-11`.
Both programs converge their own HF/integrals. GradSCF and PySCF RHF totals
are -74.961376619994 Ha. Both retained and reference spaces contain 441
determinants; this small molecule's finite excitation space is exhausted.

| Order | GradSCF correction / Ha | PySCF-action RS correction / Ha | Absolute difference / Ha |
| --- | ---: | ---: | ---: |
| E2 | -0.034747934183 | -0.034747934183 | 7.339e-14 |
| E3 | -0.009340345100 | -0.009340345100 | 4.046e-14 |
| E4 | -0.002813547195 | -0.002813547195 | 1.718e-14 |
| E5 | -0.000916760464 | -0.000916760464 | 7.240e-15 |
| E6 | -0.000318643445 | -0.000318643445 | 3.748e-15 |
| E7 | -0.000116080027 | -0.000116080027 | 1.902e-15 |
| E8 | -0.000043500906 | -0.000043500906 | 9.195e-16 |

MP8 partial total is -75.009673431315 Ha; PySCF FCI total is
-75.009700279727 Ha. The remaining 2.6848412e-5 Ha is a finite-order truncation
error in this basis, not disagreement between the MP coefficient engines.
PySCF 2.9.0 and the inspected [official MP interface](https://pyscf.org/user/mp.html)
provide native MP2; the higher-order reference is explicitly our independent
recurrence using PySCF's action, not a native PySCF MP8 API.

The same water geometry with Cartesian 6-31G*, frozen=1, gives native MP2
correlation energies -0.185314572684 Ha in both programs, differing by
4.241e-14 Ha. This larger-basis check is MP2 only. High-order full-space water
calculations with 6-31G* were not performed.

The first cold end-to-end STO-3G runs took 5.600 s (GradSCF RHF + Taylor) and
0.043 s (PySCF RHF + RS). These regions have different compilation/AD work and
are not a matched speed benchmark. Cold startup is included. No warmed timing,
GPU peak memory or scaling claim follows from them.

The native GradSCF H6 [example](../../../examples/mp/taylor_series.py) also ran:
HF -3.172676142364 Ha, E2 -0.051659165678, E3 -0.016557683628,
E4 -0.006267145387, E5 -0.002416510625, E6 -0.000954427910 Ha;
MP6 partial total -3.250531075592 Ha. It retains all 400 allowed determinants
through actual rank six and complete states through order three. Maximum
linear-response residual is 4.813e-17 Ha. Values are included as output comments.

No entire-project suite, GPU execution or high-order large-system benchmark
was performed. The generic path is in-core and grows combinatorially. Real
canonical RHF/UHF are supported; ROHF/noncanonical/complex variants and MP
density matrices remain outside this implementation. The parameter-response
test does not establish nuclear-coordinate derivative coverage or convergence
of an arbitrary MP series at lambda=1.

## Initial low-order implementation, 2026-10-09

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
implemented in that initial version. The 2026-10-10 extension above adds UMP3
and generic higher orders. See README.md for result/amplitude semantics.
