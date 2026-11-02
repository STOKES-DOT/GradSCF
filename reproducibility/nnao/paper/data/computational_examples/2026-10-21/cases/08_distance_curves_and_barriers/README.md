# Repaired representative S66x8, HB300SPXx10 and BH9 comparisons

## Methods

Method: DF-HF (RHF/UHF according to spin), def2-universal-jkfit. Cohort: `MACE-update-3351/native-UKS-repair`. Exact settings, commands, hardware and original status fields remain in the imported source records. Numerical source files are copied without editing.

## Results

Canonical merged report: 852/852 passed (S66x8 288, HB300SPXx10 360, BH9 204 species), yielding 54 reaction/basis profiles. Independent energies 31/31, maximum error 2.09184e-11 Eh; 13 selected internal-stability checks passed.

Archive status: `complete`. Exported geometry/state records: **166**.

## Interpretation boundary

Only the 242-case repair subset received the specified native real-internal-stability checks. Global HF minima and external spin/complex stability are not certified. BH9 connectors are schematic R/TS/P levels, not IRC paths; no ZPE/thermal terms. HF lacks correlated dispersion.

## Files

- `structures/`: generated standard XYZ with Angstrom coordinates, charge and 2S in the comment. State fields omitted in one record are resolved only by a unique exact-coordinate/state match in the same case; any remaining unknown is explicit `None`.
- `artifacts/`: source data, coefficients, plots, scripts and protocol records with their original relative paths. Original-language notes are retained as source records.
- `../../structure-index.json`: coordinate/state provenance and aliases; `../../FILE_MANIFEST.json`: original paths and SHA256 checksums.

### Reports

- [artifacts/nnao-curves-barriers-repair-20261021/combined-retry/results.json](artifacts/nnao-curves-barriers-repair-20261021/combined-retry/results.json)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/results.json](artifacts/nnao-curves-barriers-repair-20261021/merged/results.json)
- [artifacts/nnao-curves-barriers-repair-20261021/retry/results.json](artifacts/nnao-curves-barriers-repair-20261021/retry/results.json)

### Figures

- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9-reaction-profiles.pdf](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9-reaction-profiles.pdf)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9-reaction-profiles.png](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9-reaction-profiles.png)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_01_1-profile.pdf](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_01_1-profile.pdf)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_01_1-profile.png](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_01_1-profile.png)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_02_134-profile.pdf](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_02_134-profile.pdf)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_02_134-profile.png](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_02_134-profile.png)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_03_1-profile.pdf](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_03_1-profile.pdf)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_03_1-profile.png](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_03_1-profile.png)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_04_1-profile.pdf](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_04_1-profile.pdf)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_04_1-profile.png](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_04_1-profile.png)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_05_1-profile.pdf](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_05_1-profile.pdf)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_05_1-profile.png](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_05_1-profile.png)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_06_1-profile.pdf](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_06_1-profile.pdf)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_06_1-profile.png](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_06_1-profile.png)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_07_1-profile.pdf](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_07_1-profile.pdf)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_07_1-profile.png](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_07_1-profile.png)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_08_1-profile.pdf](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_08_1-profile.pdf)
- [artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_08_1-profile.png](artifacts/nnao-curves-barriers-repair-20261021/merged/BH9_08_1-profile.png)

## Provenance

Original GradSCF artifact directories: `nnao-curves-barriers-repair-20261021`.
Copied drivers are evidence snapshots and may retain original checkout paths. Running a numerical driver requires the corresponding GradSCF environment; this import did not rerun SCF, optimization or training.
