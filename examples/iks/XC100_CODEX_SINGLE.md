# Single-molecule Codex/iKS acceptance, 2026-11-16

The first live design loop completed locally with the official Codex Python SDK,
GPT-5.6 Luna / high, and the new GradSCF controller. The same SDK thread proposed
a formula, received its fitted reward, and proposed a revision. Both candidates
were evaluated without candidate SCF or a prior potential inversion.

## Fixed physical setup

- XC100 sample 010, neutral closed-shell BH3, eight electrons. The target is the
  published CI-derived KS determinant density, not an interacting CI 1-RDM.
- Qualified explicit cc-pVTZ contractions and original coordinates; 72 spherical
  AOs, embedded into 80 Cartesian AOs only for quadrature.
- Original level-5 grid, 113,778 points; no basis, grid or density alteration.
- CPU float64. A fixed PBE XC reference is bound through `dft.custom_xc`.
  The external PBE/PW92 formula is separately checked against a frozen libxc
  table. PySCF is used solely as an independent baseline oracle, without SCF.
- Declared vacuum model: n=sqrt(rho^2+eta^2), eta=1e-12 bohr^-3. Spatial
  derivatives in sigma=|grad rho|^2 and lap=laplacian rho remain those of rho.
- Coefficients enter affinely and stay in [-0.2,0.2]. AO directions are built
  once using AD; fitting then uses the shared bounded optimizer and small
  spherical-matrix eigensolves. Grid density diagnostics run after fitting.

## Baseline acceptance

Independent PySCF/libxc weak-form evaluation uses exactly the same explicit
basis, geometry, quadrature and declared vacuum model. No second implementation
participates in the production candidate calculation.

| Check | Measured result |
| --- | ---: |
| PBE XC energy | -5.074318472640085 Ha |
| Absolute XC-energy difference from oracle | 4.44e-15 Ha |
| Maximum spherical AO potential difference | 6.88e-15 Ha |
| Maximum overlap difference | 9.99e-16 |
| Energy directional AD/FD difference, step 1e-4 | 1.96e-10 Ha |
| Integrated grid electron-count error | 1.06e-9 electron |
| Baseline occupied/virtual gap | 0.1981558413 Ha |

The real molecule exposed a whitening-roundoff issue: a symmetric AO operator
acquired tiny antisymmetric error in X F X and failed the shared structure
guard despite ~2e-14 eigen residuals. A failing 24-AO regression now covers it.
The caller restores the exact mathematical symmetry after whitening; solver,
gap and response tolerances remain unchanged.

## Agent formulas and measured effects

Define

    s2 = sigma / [4 (3 pi^2)^(2/3) n^(8/3)]
    u  = s2 / (1+s2)
    q  = lap / [4 (3 pi^2)^(2/3) n^(5/3)]
    z  = q / sqrt(1+q^2).

The model proposed energy-density corrections to PBE, in Ha/bohr^3:

1. `Delta e1 = n^(4/3) * (a*u + b*u*z)`;
   fitted `a=-0.08320490451158145`, `b=0.2`.
2. After the first score was returned to the same thread,
   `Delta e2 = n^(4/3) * (a*u + b*u*tanh(2*z))`;
   fitted `a=-0.09544560013634085`, `b=0.2`.

Both corrections vanish in the uniform-density limit through the factor u;
three sampled uniform densities also give exactly zero. This does not establish
all exact functional constraints. The scalar-energy AD supplies the AO XC
potential; no energy is invented by integrating n*vxc.

| Candidate | Delta W (mHa) | Reward | G (mHa) | Relative grid density L2 |
| --- | ---: | ---: | ---: | ---: |
| PBE | 0 | 0 | 3.211349 | 0.004604419 |
| First proposal | 0.294444 | 0.03680545 | 2.916906 | 0.003963016 |
| Revised proposal | 0.752399 | 0.09404984 | 2.458951 | 0.003171575 |

Here W=Es-Tr[(F-T)Dtarget], G=Tr(F Dtarget)-Es, and
reward=DeltaW/(8*0.001 Ha). The revised proposal reduces the measured density
L2 error by 31.12% and G by 23.43% relative to the PBE starting point on this
one target. These are fixed-target one-solve quantities, not converged SCF
density or thermochemistry errors.

Candidate parameter AD agrees with the density-moment identity within 9.1e-13;
central finite differences at step 1e-4 differ by at most 7.5e-10 Ha. Both fits
terminate by objective-change convergence and reach the upper bound b=0.2.
The nonzero raw gradient in that bounded direction is retained in the record;
the result is not claimed as an unconstrained optimum or a globally best formula.

## Runtime and reproducibility

The baseline plus independent checks took 7.05 s in its separate acceptance run.
The two candidate evaluations, including fitting and derivative checks, took
9.67 s and 8.52 s. The full live SDK workflow took 158.23 s, including two model
turns and preparation. These wall-clock scopes must not be conflated.
The computation parent process peaked at 923.6 decimal MB RSS; this excludes
the SDK runtime child process. Execution was macOS arm64, Python 3.12.2,
JAX/JAXlib 0.8.1, SciPy 1.15.2, CPU float64, OMP4/BLAS1 (not CPU affinity).

`examples/iks/xc100_codex.py` is the thin live-run example. It validates the
baseline before opening a design thread. Reusing its run directory resumes
completed rounds; changed numerical source/configuration requires a fresh run.
It performs real model calls when run, using existing SDK authentication.

Large data remain on TF. The ignored `artifacts/xc100-codex-single/` directory
contains baseline checks, candidate gradient checks, sampled physical checks,
raw formulas, parameters, per-sample scores, deterministic aggregate feedback,
thread/parent lineage, source manifest, run history and the final acceptance
receipt. The two accepted rows share one thread ID; round two names round one
as its parent. Feedback exposes no sample IDs, raw matrices or dataset paths.

Regression command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gradscf-mplcache python -m pytest -q tests/iks tests/model tests/test_eager_minimize.py tests/test_spherical_representation.py tests/test_xc_design_pbe.py tests/test_xc_design_grid.py --ignore=tests/iks/test_reference.py
```

Result: **208 passed in 61.61 s**, exit 0. Independent code review found no
Important/Critical issue blocking this single-molecule acceptance.

No sealed test data, GPU, remote numerical job, candidate SCF or broad
transferability validation was used. The benchmark establishes the real
proposal/evaluate/feedback path on one training sample; it is not a validated
new general-purpose XC functional.
