# Methane MACE + basis energy minimization — 2026-09-16

This is a single-geometry variational RHF experiment. The geometry, Gaussian
exponents, core and polarization shells remain fixed; all 5,638 MACE/basis-head
trainable parameters enter the optimizer. It does not establish transferability.

## Setup

- Ideal tetrahedral CH4, C-H = 1.09 Angstrom, charge 0, spin 0.
- All-electron `szp3-initial`, Cartesian AOs: 27 contracted AOs, 45 primitive AOs.
- Native primitive integrals cached outside AD (ERI array: 32.805 MB decimal).
- MACE: elements H/C, 8 channels, 2 interactions, max_ell=1, correlation=2,
  zero-initialized output head, seed 0. The initial basis is recovered exactly.
- Stationary RHF Lagrangian derivative includes normalization and the Pulay
  overlap term. This driver supports first-order energy training only.
- Each candidate SCF starts from the usual hcore initialization, max_cycle=150,
  energy tolerance 1e-12 Ha, density tolerance 1e-10, orbital-gradient tolerance 1e-9.
- L-BFGS-B: maxiter=80, gtol=5e-7, ftol=1e-14, maxls=30.
- c20, Linux x86_64, CPU float64, affinity 0-3, OMP/OpenBLAS threads=1;
  JAX/JAXlib 0.10.0. No GPU used. Package versions are in metadata.json.

## Result

| Quantity | Value |
|---|---:|
| Initial energy / Ha | -39.992122564778569 |
| Final energy / Ha | -40.003818703349914 |
| Energy decrease / mHa | 11.696138571345 |
| Accepted updates / optimizer evaluations | 18 / 20 |
| Maximum final parameter gradient | 2.43824355e-6 |
| Final SCF orbital residual | 1.36182392e-10 |
| Final minimum overlap eigenvalue | 0.09165259 |
| Initial network directional AD-FD error | 7.58636209e-11 |
| Final network directional AD-FD error | 4.17761320e-11 |
| Initial independent PySCF energy error / Ha | 1.84741111e-13 |
| Final independent PySCF energy error / Ha | 9.23705556e-14 |
| Encoder parameter change norm | 0.00150643 |
| Equivalent H output discrepancy | 0 |

The optimizer reported success by relative energy reduction. The final parameter
gradient remains above the requested 5e-7 gradient threshold; this is an
energy-converged demonstration, not a certified global or fully gradient-converged
minimum. Accepted energies decrease within floating-point precision. Coefficient
updates did not approach tanh saturation (maximum absolute tanh output 0.3251).
The backbone changed, so optimization was not restricted to a constant output bias.
Elapsed time including construction, compilation, optimization and endpoint
network-gradient checks was 27.39845 seconds; the separate PySCF check is excluded.

## Contraction coefficients

These are raw input coefficients before physical Gaussian normalization.
Reference-vector norm is fixed to remove common-scale redundancy. Exponents are
unchanged. All four hydrogens have identical coefficients by molecular symmetry.

| Shell | Initial coefficients | Final coefficients |
|---|---|---|
| C valence s | -0.39589516, 1.21583436, 1.00000000 | -0.40693730, 1.46886707, 0.55840642 |
| C valence p | 0.23645995, 0.86061881, 1.00000000 | 0.27446413, 0.98822728, 0.86293356 |
| H valence s (all four H) | 0.15628498, 0.90469089, 1.00000000 | 0.14898270, 0.84448385, 1.05239811 |

## Verification and files

The methane energy-gradient regression checks a combined direction spanning
C s, C p and symmetry-shared H s against four-point differences of reconverged
SCF energies. This test plus tool/API-boundary checks: 9 passed locally.
Endpoint network directions also use four-point differences with step 1e-4.
`tests/comparisons/validate_methane_nnao.py` independently reconstructs each
atom-specific basis and reconverges PySCF RHF at both endpoints (tolerance 1e-8 Ha).

- `summary.json`, `optimization.csv`, `history.json`: setup, energies, gradients,
  diagnostics, explicit initial/final bases and timing.
- `optimization.png/pdf`: energy and gradient trajectories.
- `checkpoint.npz`: initial/final flattened NNX parameters and model outputs.
- `parameters.msgpack`: final NNX parameter state; reconstruct architecture with
  summary.model_config and seed 0, retaining the pinned MACE/template version.
- `optimization.log`, `reference.log`, `pyscf-validation.json`: actual run evidence.
- `metadata.json`: environment and command; summary.source_sha256 matches the
  current experiment script.

With a correctly installed optional MACE stack and native library:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
  python tools/optimize_methane_nnao.py --maxiter 80 \
  --output-dir artifacts/methane-nnao
python tests/comparisons/validate_methane_nnao.py artifacts/methane-nnao/summary.json
```

The recorded run used the verified snapshot at
`/tmp/gradscf-nnao-verified-20260916`, with the isolated dependency arrangement
recorded in the NNAO foundation validation. This experiment optimizes the defined
RHF objective for one methane geometry. No geometry optimization, DFT training,
excited-state validation, or chemical-accuracy/transferability claim is made.
