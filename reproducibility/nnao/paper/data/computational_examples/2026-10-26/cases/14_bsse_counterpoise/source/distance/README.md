# NNAO noncovalent curves and BH9 barriers

## Current execution

The representative run finished on 2026-10-20: all **852** cases attempted,
**831 passed, 21 failed**. The final numerical report was written at
22:44:41 +08:00 and the automatic plot summary at 22:45:01 +08:00.
The numerical case loop took 7243.31 seconds, excluding startup/checkpoint loading.

| Job | Purpose | Request | Final evidence |
|---|---|---|---|
| 102209 | 39-case short validation | V100, 32 CPU, 128 GiB, 10 min | 39 passed, zero failed |
| 102212 | 852-case representative panel | A100 80G, 32 CPU, 512 GiB, 3 h | Final report and FINISHED log; 831 passed, 21 failed |
| 102213 | Automatic figures/CSV | No GPU, 1 CPU, 8 GiB, 5 min | All 28 PNG/PDF outputs generated |

The second job checked the short numerical report: complete, 39 passed,
zero failed. A scheduler exit of zero is not a substitute for numerical checks.
The user approved the representative request before submission.
Other user's jobs and training checkpoints were not changed.

Remote checkout: `/home/user_yuanjiao/private/homefile/GradSCF_mace500_20261014`.
Job logs and case outputs are under the same relative artifact directory.
Definitions and reproducible commands are in [RUN_SPEC.md](RUN_SPEC.md).

## What is tested

- Six S66x8 complexes, all eight original distances: 48 points.
- Six HB300SPXx10 complexes covering P, S, F, Cl and Br, all ten original
  distances: 60 points. This is a representative subset, not all 3000 points.
- Nine BH9 reactions, one small supported reaction per original class,
  34 distinct species. Source charge, multiplicity and every fragment are retained.
- STO-6G, 6-31G(d,p), def2-SVP, cc-pVDZ, cc-pVTZ and frozen NNAO/MACE
  update 3351, using native all-electron DF-HF, spherical AOs and float64.

Source data are pinned to the author-maintained refdata commit
`2b09de29c7e0c23e621b2de988bc41c8090d4481`. The 2208 downloaded source files
were checked against their Git blob hashes. Out of 300 HB equilibrium
complexes, 52 contain unsupported model elements and are recorded explicitly.
All 449 BH9 reaction definitions were audited for charge/element balance and
reference energy closure. See `dataset.json` for exact selections and exclusions.

## Completed gates

Eleven new protocol tests pass: signed stoichiometry, explicit charge/spin,
forward/reverse/reaction identity, submerged barriers, rigid CP fragment
geometry, exact BSE general/SP contraction columns, and rejection of ECPs.
The sixteen existing native/core checks also passed during this preparation.

One missing conventional asset, Br/6-31G(d,p), comes directly from BSE
version 1 without approximation. All 50 required element/basis combinations
load successfully. An independent HBr CPU float64 check against same-shell
PySCF agrees to **4.55e-12 Eh**; see [basis-asset-check.json](basis-asset-check.json).
No finite differences or gradients were performed.

## Measured results

| Part | Passed | Failed |
|---|---:|---:|
| S66x8, six curves | 287 | 1 |
| HB300SPXx10, six curves | 340 | 20 |
| BH9 species | 204 | 0 |

All 54 BH9 reaction/basis profiles were assembled from complete species
energies. NNAO/MACE passed 139/142 cases; its three failures are benzene-water
at scale 2.00 and HBr-ammonia at scales 0.80 and 0.90. Eight cc-pVDZ
HBr-ammonia cases and ten STO-6G H2S-POCl3 cases also failed. These are
HF convergence/residual failures, preserved without substitution.

Large BH9 energy outliers appear in both neural and conventional bases.
The current results establish stationary SCF branches, not their global
orbital stability. Distinguish basis errors from branch selection before
drawing basis-quality conclusions. No automatic rerun was performed.

Local figures and data, with downloaded result SHA256 checked against remote:

- [BH9 overview](representative/BH9-reaction-profiles.png),
  [annotated example](representative/BH9_01_1-profile.png),
  [reaction values](representative/reaction-cases.csv).
- [S66x8 model curves](representative/S66x8-model.png),
  [frozen-basis CP control](representative/S66x8-cp-control.png).
- [HB300SPXx10 model curves](representative/HB300SPXx10-model.png),
  [frozen-basis CP control](representative/HB300SPXx10-cp-control.png).
- [Full report](representative/results.json) and
  [reaction error metrics](representative/reaction-metrics.csv).

## Figure conventions

The figure layout was rendered and visually checked using only the published
BH9 reference energies: [nine-class preview](reference-preview/BH9-reaction-profiles.png)
and [annotated example](reference-preview/BH9_01_1-profile.png).
Those previews contain no NNAO calculations. The completed measured
comparisons are in `representative/`, linked above.

The comparison source is in `tests/comparisons/nnao_reactivity/` and reuses
`tests/comparisons/nnao_core/native.py`; production code is unchanged.

- NCI model interaction curves, based on separate dimer/isolated-monomer
  neural predictions, and separate frozen-dimer-basis CP diagnostic curves.
- BH9 R/TS/P energy diagrams with forward/reverse barrier and reaction-energy
  arrows, per-reaction numeric tables and a nine-panel overview.
- CSV data for every conventional basis; failures and missing data stay visible.
- Errors against same-method HF/cc-pVTZ and published correlated references
  reported separately. HF does not recover correlated dispersion.

Connectors between BH9 stationary levels are schematic, not IRC trajectories.
All energies are electronic, not Gibbs free energies. Charged and radical
MACE cases extend beyond its neutral closed-shell training domain. The checks
establish converged stationary HF branches, not guaranteed global orbital minima.
