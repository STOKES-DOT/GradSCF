# Methane RHF basis comparison — 2026-09-16

All rows were independently reconverged with PySCF 2.13.0 at the exact geometry
used for NNAO training: tetrahedral methane, C-H=1.09 Angstrom, charge 0, spin 0.
Cartesian functions (6D/10F) are used consistently. The learned NNAO basis is
specific to this molecule and geometry; conventional bases were not refitted.

## Results

| Basis | Contracted AO | Primitive AO | RHF energy / Ha | Above def2-TZVP / mHa |
|---|---:|---:|---:|---:|
| sto-3g | 9 | 27 | -39.726700052055 | 486.410354 |
| 3-21g | 17 | 27 | -39.976739994771 | 236.370412 |
| NNAO-initial | 27 | 45 | -39.992122564779 | 220.987842 |
| NNAO-final | 27 | 45 | -40.003818703350 | 209.291703 |
| 3-21g+same-P | 35 | 45 | -40.016367430028 | 196.742977 |
| 6-31g | 17 | 38 | -40.180384752149 | 32.725654 |
| 6-31g(d) | 23 | 44 | -40.195061167383 | 18.049239 |
| 6-31g(d,p) | 35 | 56 | -40.201591561486 | 11.518845 |
| def2-svp | 35 | 53 | -40.171506967059 | 41.603439 |
| cc-pvdz | 35 | 55 | -40.198778426864 | 14.331980 |
| def2-tzvp | 60 | 83 | -40.213110406539 | 0.000000 |

The larger def2-TZVP result is a finite-basis reference, not a CBS or exact
energy. Lower energy here means a better variational RHF energy for this
geometry; it does not establish better forces, excitations or thermochemistry.

## Interpretation

- NNAO optimization reduces its own initial energy by 11.696139 mHa.
- Optimized NNAO lies 27.078709 mHa below unpolarized 3-21G.
- The controlled 3-21G+same-P baseline retains the original split valence and
  appends exactly the NNAO C d and H p polarization shells. It shares the
  core basis and 45 primitive-AO pool, but has 35 rather than 27 contracted AOs.
  Optimized NNAO remains 12.548727 mHa above this baseline.
- Optimized NNAO is 167.688264 mHa above def2-SVP and 191.242464 mHa above
  6-31G(d). It has not reached these bases' total-energy quality in this test.
- Cross-family total-energy differences contain core as well as valence and
  polarization basis effects. The fixed 3-21G core cannot be adjusted by the
  current network. No quantitative decomposition of the energy gap was performed.
- AO counts do not alone measure runtime: NNAO and the matched split-valence
  baseline use the same primitive pool. No speedup claim follows from this run.

## Verification and runtime

All 11 SCFs converged; energy tolerance=1e-12 Ha, orbital-gradient tolerance=1e-9,
max_cycle=200, common hcore (`1e`) initial guess. Recorded final orbital-gradient
norms are all below 4e-10. NNAO endpoints also agree with the recorded GradSCF
energies within 2e-13 Ha. The script checks each endpoint at tolerance 1e-8 Ha.

Hardware: Intel Xeon Platinum 8457C, CPU affinity 0–3, one OMP/MKL/OpenBLAS thread.
Total comparison runtime: 1.297 s. Per-case timings include molecule construction
and SCF, but exclude MACE inference, neural optimization and JAX compilation;
they must not be compared directly with the full training runtime.

Source and geometry-summary hashes were checked against the current local files.
`comparison.json` contains versions, hardware, settings, energies and diagnostics;
`comparison.csv` is the numeric table; `run.log` is the actual output;
`basis_comparison.png/pdf` show the energy gaps and controlled comparison.

Reproduce in an environment with PySCF:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tests/comparisons/compare_methane_basis_sets.py \
  artifacts/methane-nnao-20260916/summary.json \
  --output artifacts/methane-basis-comparison-20260916
```

The original remote run used
`/tmp/gradscf-nnao-verified-20260916/artifacts/methane-nnao/summary.json`.
No model retraining, geometry relaxation or DFT comparison was performed here.
