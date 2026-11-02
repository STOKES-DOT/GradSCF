# Frozen 500-system model inference on 18 historical relative energies

## Methods

Method: All-electron DF-RHF, def2-universal-jkfit. Cohort: `MACE-update-3351`. Exact settings, commands, hardware and original status fields remain in the imported source records. Numerical source files are copied without editing.

## Results

39 structures, two frozen coefficient predictions per structure: native and independent energy checks each passed 78/78. Maximum native/reference difference: 7.6738615462e-12 Eh. Eighteen relative energies span reaction, barrier and noncovalent categories.

Archive status: `complete`. Exported geometry/state records: **39**.

## Interpretation boundary

This is not full GMTKN55 or WTMAD2. Binding here is positive (monomer sum minus dimer), without CP; curve examples use negative interaction energies. Earlier 200-system models differ in data, pools, core assets and architecture.

## Files

- `structures/`: generated standard XYZ with Angstrom coordinates, charge and 2S in the comment. State fields omitted in one record are resolved only by a unique exact-coordinate/state match in the same case; any remaining unknown is explicit `None`.
- `artifacts/`: source data, coefficients, plots, scripts and protocol records with their original relative paths. Original-language notes are retained as source records.
- `../../structure-index.json`: coordinate/state provenance and aliases; `../../FILE_MANIFEST.json`: original paths and SHA256 checksums.

### Reports

- [artifacts/nnao-relative500-20261018/summary.json](artifacts/nnao-relative500-20261018/summary.json)

### Figures

- [artifacts/nnao-relative500-20261018/relative-energy-mae.pdf](artifacts/nnao-relative500-20261018/relative-energy-mae.pdf)
- [artifacts/nnao-relative500-20261018/relative-energy-mae.png](artifacts/nnao-relative500-20261018/relative-energy-mae.png)

## Provenance

Original GradSCF artifact directories: `nnao-relative500-20261018`.
Copied drivers are evidence snapshots and may retain original checkout paths. Running a numerical driver requires the corresponding GradSCF environment; this import did not rerun SCF, optimization or training.
