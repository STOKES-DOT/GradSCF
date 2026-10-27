# Inverse-reference functional assessment

Run from the repository root; scripts have no CLI. All inverse/electronic-
structure algorithms are in `src/gradscf`, with the existing neural architecture
under `model/neural_xc`. The example declares the system, external Dirac energy
channel, optimizer/loss and reporting.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python examples/iks/carbon_fci.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python examples/iks/compare_carbon_fci_pyscf.py
```

The first script uses GradSCF ROHF/FCI and UHF input preparation. The independent
comparison uses PySCF only and does not participate in training. Outputs go to
the ignored `artifacts/iks-carbon-sto3g-20261026/` folder: reference NPZ, summary
JSON, all101 training rows, last-step parameters, and PNG/PDF curves.

## Physical and numerical setup

- Neutral carbon, spin2 (=2Ms), Cartesian STO-3G,5 spatial AOs and all6 electrons;
  no frozen space. FCI dimension50 for (Nalpha,Nbeta)=(4,2).
- Equally average the three degenerate lowest triplet spatial roots at Ms=1;
  each has S(S+1)=2. This removes arbitrary orientation of a selected3P root.
- Grid level1,5180 points. The reference energy is a zero-temperature FCI total
  energy; density is the declared degenerate-ground-space ensemble.
- Explicit `thermal_matrix`, auxiliary T=0.01Ha, AO-product local potential
  basis:30 coefficients, retained rank28. Column-normalized regularization and
  relative rank cutoff are1e-12; maximum metric-density/operator errors1e-6.
  Target occupations are not clipped.
- Existing ResidualMixingMLP, widths(16,16),817 parameters, one supplied spin-
  Dirac energy-density channel. HFX/PT2 channels are off. Seed0; initial Flax
  parameters are lifted from float32 to float64 before training.
- 100 Adam updates, lr0.002; loss=delta_E**2+abs(delta_E)+0.01*operator_mse,
  with numerical energies normalized to1Ha. The FCI density defines the inverse
  operator target. No candidate SCF/density loss is used.

This auxiliary finite-basis reference is not zero-temperature Wu–Yang, an exact
thermal many-body reference or a unique continuum XC potential. It requires a
strictly interior1-RDM; other targets need other inverse methods. Inverse-map
AD/JIT is explicitly unsupported. Model derivatives against the fixed reference
remain live. See the [library interface](../../src/gradscf/dft/iks/README.md).

The full-1-RDM constraint also fixes its one-body kinetic expectation. It must
not be used to manufacture an exact zero-temperature KS XC-energy decomposition
or identify the auxiliary operator with the continuum XC potential. Comparisons
below are fixed-trial-state energies and auxiliary-reference operator scores.

## Local results, 2026-10-26

Apple M4 Pro, CPU, macOS arm64, Python3.12.2, JAX/JAXlib0.8.1, PySCF2.9.0,
Flax0.12.1, Optax0.2.6, float64. OMP4/BLAS1 are environment controls, not a
claim that JAX's process affinity is restricted to4 CPUs.

FCI total energy: **-37.21873355063647Ha**. Independent PySCF checks give maximum
root-energy error7.11e-15Ha and spin-density AO error1.45e-14. The inverse meets
the original tolerances: metric-density max error2.285e-7, aligned operator max
error8.209e-8Ha, grid-density relative L2 error5.807e-9. Normalized retained
projection condition is about3.212e5.

| Model/state | Total energy (Ha) | Energy error (Ha) | Operator MSE (Ha^2) |
| --- | ---: | ---: | ---: |
| Analytic spin-Dirac exchange | -36.5704390755 | +0.6482944751 | 2.7041513 |
| Neural, initial | -34.9207215713 | +2.2980119793 | 2.3463628 |
| Neural, last step100 | -37.1938683602 | +0.0248651904 | 2.8154606 |

Loss changes7.6023347 ->0.0536381. The minimum observed loss occurs at step56,
with energy error-0.00108676Ha; the saved model and table use the last step.
Final MAE is24.865mHa, failing a1mHa target. Operator MSE increases, even though
energy error and commutator stationarity residual decrease. This demonstrates
the interface and distinct training signals, not an accurate general functional.

A deterministic parameter-direction check includes the operator-loss mixed
derivative: AD gives0.00085331057435; centered FD at1e-3/3e-4 differs by
2.62e-11/6.12e-11. The complete script exits0 in about20.22s including reference
construction, inverse, compilation, training, derivative checks and plotting,
with peak child RSS about1.80GB. This is not a warm-step benchmark or an
exclusive-machine speed comparison.

The first unnormalized projection failed its unchanged density tolerance and
correctly stopped before training; its evidence is preserved. A BSD time wrapper
also failed a sandboxed sysctl after the child finished. A separate standard-
library subprocess run records the successful final exit and source equality.
