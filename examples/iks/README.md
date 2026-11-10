# Inverse-reference functional assessment

`fixed_reference.py` demonstrates the new orbital-reference/assessment layer
with a manufactured two-AO system, an external analytic custom_xc and an
external MLP. It reuses the existing fixed-density Trainer and exposes the same
raw objective to an agent. No inverse solver, candidate SCF or molecular
accuracy claim is involved. Run `PYTHONPATH=src JAX_PLATFORMS=cpu python
examples/iks/fixed_reference.py`.

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


## PBE analytic-functional design loop

`wu_yang_pbe.py` verifies a known H2/PBE density -> inverse reference -> PBE
round trip. `pbe_agent_functionals.py` contains the external c001/c002 analytic
GGA proposals. `pbe_agent_loop.py` replays the measured parameter-AD diagnostic
and plots the recorded search/test results, with no CLI or model invocation.

The molecular pilot uses30distinct CAFES-QM9 species, one frozen frame each,
with a20/5/5 train/adaptive-validation/sealed-test split. Labels are unrelaxed
CCSD Lambda1RDMs and CCSD total energies including nuclear repulsion, from the
published def2-SVP dataset. They are not FCI data. The documented spherical
AO map is embedded in normalized Cartesian def2-SVP and checked against the
source overlap. Grid level1 is fixed. All30local references meet relative
density L2<=1e-3, gradient max<=1e-8 and occupied/virtual gap>1e-7Ha. These
are approximate finite-basis density references, not unique exact potentials.

The parent prepares/validates references and owns all code, scheduler work,
training and scoring. GPT-5.6 Luna/high proposes formulas from public literature
and receives only explicitly authorized anonymous train/validation feedback.
Its raw proposals, formulas, bounds and parameter initial values are preserved.
The pilot ran two scored formulas and one measured feedback iteration, within
the authorized maximum of5rounds. It does not train LLM policy weights.

For each molecule the shared `dft.Functional` scalar energy supplies both its
AO potential and mixed parameter/density derivatives. The affine tables reuse
PBE E/F and require no repeated SCF or ERI during correction fitting. The
training objective is mean[(DeltaE/1Ha)^2+abs(DeltaE/1Ha)+.01*||kappa||^2],
using the shared occupied-virtual response with a0.05Ha denominator floor.
L-BFGS reports objective-change convergence; the stored point gradient of MAE
at a zero-error kink is not a global-optimality certificate.

The frozen reward uses validation WRMSD J in Hartree:
R=(0.85*J_PBE)/max(J,1e-12Ha)*0.9^N_failed. Sampled checks cover finite
first derivatives/HVP, spin interchange, a zero uniform-density correction,
and finite actual fixed-reference response. A sampled pass is not a proof of
all exact functional constraints. No test metric participates in selection.

| Formula | Train WRMSD (mHa) | Validation WRMSD (mHa) | Validation rotation norm squared | Reward |
| --- | ---: | ---: | ---: | ---: |
| c000: PBE | 332.0686 | 370.4044 | 0.00780356 | 0.8500 |
| c001: q,q^2,q*zeta^2 | 26.0310 | 21.5553 | 0.00794445 | 14.6063 |
| c002: q^2,q^2*(1-q)^2 | 25.1142 | 20.0988 | 0.00795844 | 15.6648 |

c001's polarization direction is exactly zero on restricted inputs and is
not an identified spin response. c002 removes it and suppresses the leading
small-gradient correction. Its validation energy improves, while its proposed
stationarity improvement is not observed. c002 was frozen before one final
5-species test. These are fixed-reference absolute total-energy results;
self-consistent accuracy, atomization/reaction energies and broad transfer are
not established. The fixed vacuum scale breaks strict coordinate scaling in
the tail; exact exchange spin scaling and a global Lieb-Oxford bound are not
enforced by either proposal.

To replay a downloaded result bundle (BUNDLE.json, immutable references/caches,
records and the final test), set its root explicitly:

```bash
GRADSCF_IKS_BUNDLE=/path/to/pbe_loop_results PYTHONPATH=src JAX_PLATFORMS=cpu python examples/iks/pbe_agent_loop.py
```

The bundle retains original file hashes. Explicit loader path overrides allow
relocation without rewriting teacher JSON or cache metadata. The example
checks a training-parameter directional AD derivative against finite differences
at a regular point, then writes `figures/pbe_agent_loop.png` and `.pdf`.
All molecular calculations were Slurm CPU jobs on jy_gpu_2 (CPU4,16GiB for
baseline/candidates/finaltest, noGPU); JAX0.10.0 and float64. No GPU result is
claimed. The data and numeric tables are external artifacts, not source files.

Primary references: [PBE](https://doi.org/10.1103/PhysRevLett.77.3865),
[agentic functional search](https://arxiv.org/abs/2605.05460),
[stationarity constraints](https://arxiv.org/abs/2610.30022).

Final held-out results (5species, opened once after freezing c002): PBE WRMSD
0.386136810Ha and MAE0.382856646Ha;
c002 WRMSD0.023639536Ha and MAE0.022795644Ha.
The mean squared rotation response changes from0.010160658
to0.010426151; density improvement is not demonstrated.
The remote replay passed with empty stderr and directionalAD0.0758799152882
versus finite difference0.0758799124055. All formula and per-molecule score
records are retained separately; non-winner test results are not evaluated.

## Standard finite-lambda ZMP

`zmp.py` uses only GradSCF to generate small PBE targets and perform restricted
HeH+ and unrestricted Li ZMP inversions. The Coulomb callback reuses the packed
J-only integral action. It may be replaced by a direct/DF action without
changing the inversion implementation. All model evaluation/training ports
continue to accept the resulting valid `OrbitalReference`.

The independent `compare_kspies_zmp.py` uses only KSPies/PySCF/NumPy on saved
targets, checking densities and physical Fock matrices at lambda1,5,10,20,50,100.
Neither example defines a CLI. Set `GRADSCF_IKS_VALIDATION` to the shared result
directory. KSPies is an optional external reference, never a runtime dependency
of `gradscf.dft.iks`. See [the theory and method boundary](../../src/gradscf/dft/iks/KSPies.md).


CPU verification2026-11-10:159tests pass and12KSPies comparisons pass. Maximum
AO-density difference across both systems/lambdas is3.71e-10, maximum physical
Fock difference4.86e-10Ha. Atlambda100, relative density reconstruction errors
are4.07464e-4 (HeH+) and6.04270e-4 (Li, spin-resolved); both meet1e-3.
These are finite-lambda/known-PBE validation results, not an exact-density claim.
The independent comparison uses a process-local legacy SciPy angular-function
alias and documented small-history/level-shift controls, with upstream ZMP
source unchanged. No performance-equivalence claim is attached.


## SCF-free single-sample reward

`single_sample_reward.py` assesses three external scaled-Dirac exchange formulas
on a qualified, cached CAFES-QM9 CH4 CCSD/Wu-Yang reference: Cartesian def2-SVP,
35 AOs, grid level1 (15068 points). No HF, candidate SCF or new inversion is
performed. Each score builds a frozen Fock once, fills its lowest orbitals once
and reports the one-shot/target density error alongside the inverse baseline.
This demonstrates the interface, not a fitted or general-purpose XC functional.
See the [equations and interpretation](../../src/gradscf/dft/iks/README.md#single-sample-reward-without-candidate-scf).

Point `GRADSCF_IKS_TEACHER` to a qualified teacher JSON and optionally
`GRADSCF_IKS_REFERENCE` to its relocated `.reference.npz`. The existing loader
validates the exact NPZ hash. These labels and metadata stay outside Git.
The local copy for this run is under
`/Volumes/TF/GradSCF-data/CAFES-QM9/iks-single-reward-20261110/`.

```bash
GRADSCF_IKS_TEACHER=/Volumes/TF/GradSCF-data/CAFES-QM9/iks-single-reward-20261110/teacher.json \
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/iks/single_sample_reward.py
```

The JSON output records every candidate formula, coefficient, raw diagnostic,
reward scale and validity flag. It separates input preparation, compilation and
warm evaluation time and checks the reward parameter derivative against a
centered finite difference. No CLI or external agent is launched.

In the CPU4 validation, the inverse baseline was 0.07242% relative L2. The
one-shot density errors were 3.22182%, 1.59232% and 1.58280% for scales 0.8,
1.0 and1.2; corresponding rewards were 0.0565686,0.141990 and0.135336.
The default reward combines density error with the variational defect, so its
ordering need not equal density-only ordering. Warm evaluations took about
56--58ms; this excludes preparation and compilation and is not a full-SCF
speed comparison. Full SCF density accuracy has not been measured in this test.
