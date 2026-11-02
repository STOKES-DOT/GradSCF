# HCN training geometry and HCN/HNC transition geometry

## Methods

Method: All-electron DF-RHF, def2-universal-jkfit. Cohort: `MACE-update-3351`. Exact settings, commands, hardware and original status fields remain in the imported source records. Numerical source files are copied without editing.

## Results

Two finite-budget two-start free-contraction comparisons. MACE-to-free gaps are 0.8033984204 mEh for the training HCN geometry and 30.0321905941 mEh for the transition geometry.

Archive status: `complete`. Exported geometry/state records: **2**.

## Interpretation boundary

The free endpoints satisfy the recorded coefficient-gradient criterion; they are not certified global minima.

## Files

- `structures/`: generated standard XYZ with Angstrom coordinates, charge and 2S in the comment. State fields omitted in one record are resolved only by a unique exact-coordinate/state match in the same case; any remaining unknown is explicit `None`.
- `artifacts/`: source data, coefficients, plots, scripts and protocol records with their original relative paths. Original-language notes are retained as source records.
- `../../structure-index.json`: coordinate/state provenance and aliases; `../../FILE_MANIFEST.json`: original paths and SHA256 checksums.

### Reports

- [artifacts/nnao-free500-hcn-train-20261018/summary.json](artifacts/nnao-free500-hcn-train-20261018/summary.json)
- [artifacts/nnao-free500-hcnts-20261018/summary.json](artifacts/nnao-free500-hcnts-20261018/summary.json)

### Figures


## Provenance

Original GradSCF artifact directories: `nnao-free500-hcn-train-20261018`, `nnao-free500-hcnts-20261018`.
Copied drivers are evidence snapshots and may retain original checkout paths. Running a numerical driver requires the corresponding GradSCF environment; this import did not rerun SCF, optimization or training.
