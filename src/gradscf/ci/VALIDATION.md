# CI validation

## CID and total-spin diagnostics — 2026-09-22

Branch: `feat/ci-cc-forward-parity`, based on `b8f8535` for this increment.
Environment: local arm64 CPU, JAX 0.8.1, PySCF 2.9.0, float64. PySCF supplies
independent FCI/spin-operator oracles, not production runtime calculations.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python -m pytest -q tests/ci tests/cc tests/solvers/test_nonlinear.py
```

**165 passed, 2 warnings, 406.04 seconds**, no skipped tests. The warnings are
PySCF OpenMP-availability notices. The 18 new CID/spin cases also passed as a
focused invocation (**18 passed, 15.25 seconds**), before this regression.

Subsequent inspection found that changing `excitation_ranks` on an already
constructed CIS/UCIS object bypassed its constructor rejection. A failing
regression was added to the existing scope test, and the same guard was added
at `kernel()` entry. The modified scope case plus restricted singlet/triplet
CIS and UHF CIS/TDA comparisons passed (**4 passed, 1 warning, 4.21 seconds**).
Those four cases are repeats of cases in the 165-test selection, not additional
distinct tests. This is targeted validation, not the full repository suite.

Coverage includes:

- Direct CID rank selection and retained-space budget accounting; invalid
  rank sets and unattainable ranks; frozen, empty and polarized spaces.
- Restricted H4 and unrestricted H3 CID energies against independently
  projected PySCF FCI Hamiltonians; Davidson, multiple roots and energy AD.
- Arbitrary real CI vectors (seed 77) in common and different orbital frames,
  including frozen electrons, against `pyscf.fci.spin_op`.
- Known S=0, 1 and 2 eigenstates, giving S^2=0, 2 and 6, respectively.
- JIT/AD in CI coefficients and cross-spin overlaps; converged-state spin
  response to fixed-MO integral perturbations with implicit eigenvectors.
- Explicit-overlap requirements, stale rank changes, invalid shapes/complex
  inputs/nonfinite states, and native UHF/ROHF facade integration.

CID energy comparisons use 2e-10 Hartree (2e-9 for default-tolerance dispatch),
spin diagnostics 2e-12 to 2e-11 in hbar^2 units, and derivative comparisons
2e-8 absolute tolerance with steps 1e-4 (integrals) or 1e-5 (vector/overlap).

Independent review performed two additional probes, not counted as pytest cases:
720 small-space rank/frozen/budget combinations for nmo=1..5 agreed with direct
determinant enumeration; two different three-dimensional MO subspaces in a
six-dimensional AO space gave spin diagnostics within 4.44e-16 of PySCF even
though the cross-frame overlap was not orthogonal. No blocking issue was found.

## Native example

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/ci/cid_spin.py
```

This passed using only GradSCF for RHF/UHF/ROHF and CI. STO-3G, SCF tolerance
1e-12, CI residual tolerance 1e-11, float64 CPU; H4 z coordinates are
0, 0.8, 1.9, 3.1 Angstrom, and H3 z coordinates are 0, 0.85, 1.9 Angstrom.

| System/reference | Method/root | Energy / Hartree | S^2 / hbar^2 | Effective multiplicity |
| --- | --- | ---: | ---: | ---: |
| H4/RHF | CID/0 | -2.160079573952509 | 0.0 | 1.0 |
| H4/RHF | CID/1 | -1.4261379503015075 | 2.22e-16 | 1.0000000000000004 |
| H4/RHF | CISD/0 | -2.160092360349407 | -4.44e-16 | 0.9999999999999991 |
| H4/RHF | CISD/1 | -1.9365436926554969 | 1.9999999999999998 | 3.0 |
| H3/UHF | CID/0 | -1.5732621174265453 | 0.7529224790377552 | 2.0029203469312056 |
| H3/ROHF | CID/0 | -1.5690262487385913 | 0.7633245057281185 | 2.013280413383211 |

Near-zero signed S^2 values are roundoff and are not clipped by the code.
The H3 results illustrate mixed-spin truncated CI states; a ROHF determinant
does not guarantee that this rank-selected, fixed-M_s CI space is spin adapted.
Example wall time was not separately measured; no performance benchmark is
claimed. Large CI spaces, GPU, automatic total-spin selection, complex/GHF
states and full nuclear/orbital response are outside this validation.

For the earlier CI/CC forward-property validation, see
[`cc/VALIDATION.md`](../cc/VALIDATION.md). Definitions and current limits are in
[CID_SPIN.md](CID_SPIN.md).
