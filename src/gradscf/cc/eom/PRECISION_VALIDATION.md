# Shared branch search, stability and precision validation

2026-09-23. CPU (`TFRT_CPU_0`), macOS arm64, Python 3.12.2, JAX 0.8.1,
PySCF 2.9.0, float64, OMP_NUM_THREADS=1. The implementation extends existing
modules instead of adding another eigensolver, energy formula or retry engine.

## Reuse and scope

- `multistart` owns the single baseline and all seed/amplitude trials. The old
  single-seed call remains valid. A multi-seed call adds no wrapper baseline loop.
- Occupation topology, Cayley rotations and density construction are extracted
  from the existing orbital objective factory and reused unchanged.
- One curvature core constructs JAX gradient/HVP actions, calls the public
  `solve_hermitian`, and checks true residuals. The private Davidson call formerly
  in unrestricted stability is removed. RKS and UKS retain their own original
  energy evaluators; compressed native AO ERIs are not expanded to four indices.
- SCF diagnostics use the existing `||2 F_vo||` RKS residual. The EOM layer report
  references existing CC/EOM results and freshness checks. It does not duplicate
  CC residual equations, rerun a solver or change a tolerance.

Restricted stability is real and internal only. Positive partial Ritz curvature
is an estimate; dense curvature is certified only after its residual checks
pass. No spin/complex/global-minimum or non-Hermitian degenerate-AD claim is made.

## Stretched F2 branch search

F 0 0 0; F 0 0 2.2 Angstrom, STO-3G, RHF. SCF energy/density/gradient targets
1e-12/1e-10/1e-9, max_cycle=250. One multistart call uses seeds (0,1), amplitudes
(0.5,1.0), and `require_stable=True`; its source remains unsolved/unmodified.

| Attempt | Seed | Amplitude | HF energy / Ha | SCF converged | Internal stability |
| ---: | --- | ---: | ---: | --- | --- |
| 0 | baseline | 0.0 | -195.524616699278 | True | False |
| 1 | 0 | 0.5 | -195.596566567657 | True | False |
| 2 | 0 | 1.0 | -195.596566567657 | True | False |
| 3 | 1 | 0.5 | -195.671626240735 | True | True |
| 4 | 1 | 1.0 | -195.596566567657 | True | False |

Only five candidates were run (one baseline plus four rotations). Candidate 3
was selected, with lowest computed curvature 0.151716184571 Ha per squared
normalized angle. Its agreement with the previously validated reference branch
is fixture evidence, not a general proof of global minimization.

## He2 upstream precision

He 0 0 0; He 0 0 3 Angstrom, 6-31G. The report requests SCF gradient <=1e-11,
CC residual <=1e-11 and EOM residual <=1e-9 Ha. CC and IP use three roots, with
the same definitions and Hamiltonian as the preceding molecular study.

| Native SCF settings | Reported half-angle gradient norm | CC residual | Largest EOM R/L residual | Layered check |
| --- | ---: | ---: | ---: | --- |
| energy 1e-12; default density/gradient targets | 6.8508e-8 | 2.9943e-12 | <2.0e-15 | False: SCF target |
| energy/density/gradient 1e-14/1e-12/1e-11 | 3.3190e-16 | 2.9943e-12 | <3.0e-15 | True |

The third IP energy changes from 2.387780775600549 to 2.387780779034967 Ha.
The new report identifies the upstream stationarity target instead of suggesting
more EOM iterations. Neither a passed report nor a small gradient is a formal
bound on the spectral error; derivative validity and stability are separate.
For an explicit MO-only CC reference, the missing SCF report and combined check
are None, rather than an invented successful SCF measurement.

## Numerical and architecture verification

The initial missing-API tests failed before implementation. Focused tests then
passed (10 passed, 3 optional-XC skips, 46.15 s). Independent restricted Hessian
and nonstationarity checks passed (2 passed, 11.47 s): water/STO-3G native
curvatures are compared with twice PySCF's analytic RHF `gen_g_hop_rhf` action,
which uses the half-gradient convention. Absolute comparison tolerance is
2e-9 Ha; dense and iterative native checks use 2e-8 Ha. These are assertion
tolerances, not claimed measured errors. Existing UHF/UKS curvature/escape tests
retain their own independent oracles.

The broader selected regression completed **77 passed, 3 skipped in 445.24 s**,
covering SCF branch/precision, R/U stability, orbital optimization and gradients,
higher-order SCF, EOM and unified eigen tests. The three skipped tests require
optional `jax_xc`, which is unavailable; no DFT XC accuracy is claimed here.
Independent review found one failed-dense certificate flag issue. A too-strict
Hessian residual target reproduced it; the certificate now also requires a
successful residual check. The final five new precision tests were rerun after
that fix: **5 passed in 36.91 s**. No whole-repository/GPU validation is claimed.

The native example below also completed and generated the measured values above.
Independent read-only reviews confirmed the reuse paths, compressed-ERI handling,
curvature scaling, source immutability, freshness and missing-SCF semantics.

## Reproduce

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
python examples/cc/precision_and_branch_checks.py

PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_scf_precision.py tests/test_restricted_multistart.py \
  tests/test_uhf_stability.py tests/test_uks_stability.py \
  tests/test_scf_orbital_optimization.py tests/test_scf_orbital_gradients.py \
  tests/test_scf_higher_order.py tests/cc/test_eom.py \
  tests/solvers/test_unified_eigen.py
```

Settings and results are in Hartree and Angstrom as specified above. Times
include JAX compilation and overlapping local validation work; they are not
performance benchmarks. Ordinary single-run SCF/CC/EOM defaults are unchanged.
