# Fixed STO-6G polarization controls on the ethane torsion scan

## Methods

Method: All-electron DF-RHF, def2-universal-jkfit. Cohort: `update-3351-and-fixed-basis-controls`. Exact settings, commands, hardware and original status fields remain in the imported source records. Numerical source files are copied without editing.

## Results

75 new SCFs for STO-6G, STO-6G+C(d), and STO-6G+C(d)+H(p), all passed. Combined tables contain 11 methods/bases over 25 geometries.

Archive status: `complete`. Exported geometry/state records: **25**.

## Interpretation boundary

The augmented STO-6G bases are explicitly defined custom controls, not a unique standard STO-6G* or STO-6G**. Absolute-energy and relative-curve quality are separate.

## Files

- `structures/`: generated standard XYZ with Angstrom coordinates, charge and 2S in the comment. Missing state information is explicitly `None`, not assigned a guessed state.
- `artifacts/`: source data, coefficients, plots, scripts and protocol records with their original relative paths. Original-language notes are retained as source records.
- `../../structure-index.json`: coordinate/state provenance and aliases; `../../FILE_MANIFEST.json`: original paths and SHA256 checksums.

### Reports

- [artifacts/nnao-ethane-sto6g-polarized-20261020/results.json](artifacts/nnao-ethane-sto6g-polarized-20261020/results.json)

### Figures

- [artifacts/nnao-ethane-sto6g-polarized-20261020/ethane-absolute-comparison.pdf](artifacts/nnao-ethane-sto6g-polarized-20261020/ethane-absolute-comparison.pdf)
- [artifacts/nnao-ethane-sto6g-polarized-20261020/ethane-absolute-comparison.png](artifacts/nnao-ethane-sto6g-polarized-20261020/ethane-absolute-comparison.png)
- [artifacts/nnao-ethane-sto6g-polarized-20261020/ethane-relative-comparison.pdf](artifacts/nnao-ethane-sto6g-polarized-20261020/ethane-relative-comparison.pdf)
- [artifacts/nnao-ethane-sto6g-polarized-20261020/ethane-relative-comparison.png](artifacts/nnao-ethane-sto6g-polarized-20261020/ethane-relative-comparison.png)

## Provenance

Original GradSCF artifact directories: `nnao-ethane-sto6g-polarized-20261020`.
Copied drivers are evidence snapshots and may retain original checkout paths. Running a numerical driver requires the corresponding GradSCF environment; this import did not rerun SCF, optimization or training.
