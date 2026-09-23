# Degenerate Ritz and restricted SCF robustness validation

2026-09-23. macOS arm64 CPU (`TFRT_CPU_0`), Python 3.12.2, JAX 0.8.1,
PySCF 2.9.0, float64. Numerical baseline before these fixes: `74353c9`.
The original failed results remain in [the initial molecular study](MOLECULAR_VALIDATION.md)
and its CSV. This followup does not overwrite or relabel those observations.

## Root cause and numerical repair

CO/STO-3G (C at the origin, O at z=1.128 Angstrom) had full-rank complex left
Ritz vectors with singular values approximately (1.3810, 1.0000, 0.3047).
The first two columns were conjugates, with imaginary norms 0.2372. Their
real parts had singular values (1.3739, 1.0000, 7.04e-17). Thus taking `.real`
discarded an independent direction; the right eigensolve happened to use a
real representation of the same repeated eigenvalue. Small eigen-equation
residuals did not establish a complete dual basis.

The shared numerical extraction now handles real clusters unresolved at
`128*eps*max(1, ||A||_infinity)`. For a cluster center lambda, form

```
A - lambda I = U Sigma V.T
R = V0
G = U0.T V0
L = U0 G^(-T)
```

U0 and V0 span the full approximate left/right null spaces for the cluster.
Construct the complete dual before selecting requested columns, so cutting a
cluster does not pair unrelated partial left/right bases. Isolated roots and
genuine complex roots keep their existing path. For Davidson, this work occurs
only in the bounded projected matrix. There is no physical dense fallback,
pseudoinverse, or full eigenvector-matrix inverse involving excluded states.
Final physical residuals, overlap rank and conditioning still control acceptance;
defective Jordan clusters are not declared successful.

This is a forward basis repair. Repeated ordered roots and individual vectors
still have no AD contract; the existing isolated-energy gap policy is unchanged.
For the original failing CO setup (nroots=3, space=48, seed=0, tolerance=1e-9),
all forward checks now pass. The smallest left singular value is about 1.01191,
and ||L.T R-I|| decreases from 3.055 to 6.41e-16. Maximum energy error against
the complete PySCF spectrum remains 6.34e-11 Ha. Seed 1, space 64, tolerance
1e-10 and the dense reference also pass. The previously unsuccessful tighter
threshold case no longer loses the left-space rank.

## Explicit SCF branch selection

The new opt-in [RKS multistart API](../../scf/MULTISTART.md) reuses native orbital
rotations and records every attempt. It clones its source, preserves controls,
and selects only finite converged candidates. A single ordinary `run()` is
unchanged. No reference-software density/orbitals are used to initialize SCF.
The baseline N2 hcore branch near -106.769673857813 Ha remains recorded. Rotation
amplitudes 0.15 and 0.4 (seed 20260923) reach -107.496500511798 Ha. Selection is
explicit and discrete; it is not a stability/global-minimum proof or an AD rule.

## Eight-system forward rerun

Geometries and basis sets are the same fixed fixtures as the initial study.
Three roots per EE/IP/EA sector, max_space=48, max_cycle=180, guard_roots=1,
seed=0, true-residual tolerance 1e-9 Ha. CC energy/residual tolerances are
1e-12/1e-11 Ha; reference HF/CC energy tolerance 1e-13 Ha. The full PySCF action
spectra are independent forward oracles. All eight systems pass all three
sectors: **24 sectors, 72 roots**. N2 alone explicitly uses multistart with SCF
energy/density/gradient tolerances 1e-12/1e-10/1e-9 and max_cycle=150.

| Molecule | Basis | SCF start | Largest energy error / Ha | Maximum ||L.T R-I|| | Forward sectors |
| --- | --- | --- | ---: | ---: | ---: |
| LiH | 6-31g | Default hcore | 8.705e-12 | 1.305e-15 | 3/3 |
| HF | 6-31g | Default hcore | 7.462e-11 | 1.143e-15 | 3/3 |
| N2 | sto-3g | Explicit multistart | 5.109e-13 | 6.613e-16 | 3/3 |
| CO | sto-3g | Default hcore | 1.285e-10 | 6.410e-16 | 3/3 |
| NH3 | sto-3g | Default hcore | 1.272e-10 | 5.052e-16 | 3/3 |
| CH4 | sto-3g | Default hcore | 4.024e-12 | 9.686e-16 | 3/3 |
| CH2O | sto-3g | Default hcore | 1.755e-10 | 9.155e-16 | 3/3 |
| C2H4 | sto-3g | Default hcore | 6.526e-12 | 9.679e-16 | 3/3 |

Per-case comparison timers sum to 275.49 s, including compilation
and reference diagonalization but excluding the preceding N2 multistart step.
Other local regressions ran concurrently; these are reproduction timings, not
a performance comparison.

[ROBUSTNESS_VALIDATION.csv](ROBUSTNESS_VALIDATION.csv) records all 72 roots,
reference energies, residuals, response flags, condition numbers and iterations.
Degenerate root groups retain `response_valid=False`. CH2O and C2H4 have valid
isolated-root flags in each tested sector. The molecular sweep validates forward
outputs; derivative accuracy is checked separately by the AD regressions.

## Final verification

The complete targeted regression passed **141 tests**, with 2 explicitly
deselected DFT smoke tests, in **518.73 s**. No selected test failed or skipped.
This covers all shared solvers, EOM, the restricted multistart tests and the
available ground-state facade checks. The native multistart/CCSD/EOM example
also completed successfully. No complete-repository or GPU validation is claimed.

## Reproduction and tests

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tests/comparisons/compare_eom_robustness.py > robustness.jsonl

PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
python examples/cc/nitrogen_multistart.py

PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q \
  tests/solvers tests/cc/test_eom.py tests/test_restricted_multistart.py \
  tests/test_pyscf_style_ground_state_api.py -k 'not real_rks_kernel_smoke'
```

The two real-RKS DFT smoke tests are deselected because optional `jax_xc` is
absent in this environment; that pre-existing dependency boundary is unrelated
to these HF changes. GPU, general/open-shell multistart, complete nuclear
responses and differentiable non-Hermitian degenerate projectors are unverified
or outside this increment.

Initial red/green evidence: three deterministic repeated-root cases failed
before the cluster fix; the 26 focused non-Hermitian tests passed afterward
(27.01 s). The three multistart tests failed before its API existed and passed
afterward (3.45 s). Independent review additionally checked random semisimple
matrices, truncated/multiple clusters and Jordan blocks, without finding a
blocking issue. Timings include compilation and are not performance claims.
