# Earlier aniline GPU4PySCF conventional-basis comparison

## Methods

Method: DF-RHF; final GPU4PySCF comparison uses cc-pVQZ-JKFIT. Cohort: `legacy-2026-09-aniline-model`. Exact settings, commands, hardware and original status fields remain in the imported source records. Numerical source files are copied without editing.

## Results

Original single-geometry basis-size and total-energy tables; the final comparison includes a 98-AO trained NNAO basis and conventional minimal-to-triple-zeta controls.

Archive status: `historical_examples`. Exported geometry/state records: **1**.

## Interpretation boundary

This older model and cc-pVQZ-JKFIT auxiliary basis differ from the update-3351/def2-universal-jkfit cohort. The single aniline example does not establish generalization.

## Files

- `structures/`: generated standard XYZ with Angstrom coordinates, charge and 2S in the comment. State fields omitted in one record are resolved only by a unique exact-coordinate/state match in the same case; any remaining unknown is explicit `None`.
- `artifacts/`: source data, coefficients, plots, scripts and protocol records with their original relative paths. Original-language notes are retained as source records.
- `../../structure-index.json`: coordinate/state provenance and aliases; `../../FILE_MANIFEST.json`: original paths and SHA256 checksums.

### Reports

- [artifacts/aniline-basis-comparison-20260920/local-smoke/comparison.json](artifacts/aniline-basis-comparison-20260920/local-smoke/comparison.json)
- [artifacts/aniline-basis-comparison-20260920/remote-polarized/comparison.json](artifacts/aniline-basis-comparison-20260920/remote-polarized/comparison.json)
- [artifacts/aniline-basis-comparison-20260920/remote-smoke/comparison.json](artifacts/aniline-basis-comparison-20260920/remote-smoke/comparison.json)
- [artifacts/aniline-gpu4pyscf-df-basis-final-20260923/comparison.json](artifacts/aniline-gpu4pyscf-df-basis-final-20260923/comparison.json)

### Figures

- [artifacts/aniline-gpu4pyscf-df-basis-final-20260923/basis_comparison.pdf](artifacts/aniline-gpu4pyscf-df-basis-final-20260923/basis_comparison.pdf)
- [artifacts/aniline-gpu4pyscf-df-basis-final-20260923/basis_comparison.png](artifacts/aniline-gpu4pyscf-df-basis-final-20260923/basis_comparison.png)

## Provenance

Original GradSCF artifact directories: `aniline-gpu4pyscf-df-basis-final-20260923`, `aniline-basis-comparison-20260920`.
Copied drivers are evidence snapshots and may retain original checkout paths. Running a numerical driver requires the corresponding GradSCF environment; this import did not rerun SCF, optimization or training.
