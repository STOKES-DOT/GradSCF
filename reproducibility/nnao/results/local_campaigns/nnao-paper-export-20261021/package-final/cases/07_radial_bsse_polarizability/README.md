# Atomic radial density, S22 counterpoise and static polarizability

## Methods

Method: DF-HF (RHF/UHF according to spin), def2-universal-jkfit. Cohort: `MACE-update-3351`. Exact settings, commands, hardware and original status fields remain in the imported source records. Numerical source files are copied without editing.

## Results

284 attempted cases: 272 passed, 12 failed. All 35 NNAO cases passed. Failures are ma-def2-QZVPP GPU-memory errors: ten S22 dimers and two polarizabilities. Atomic cases 36/36; CP 166/176; polarizability 70/72.

Archive status: `complete_with_failures`. Exported geometry/state records: **35**.

## Interpretation boundary

Polarizabilities hold the predicted basis fixed under the field. Atomic charge/spin states extrapolate from neutral closed-shell molecular training. Shared-subset S22 statistics must not be described as complete QZ coverage.

## Files

- `structures/`: generated standard XYZ with Angstrom coordinates, charge and 2S in the comment. Missing state information is explicitly `None`, not assigned a guessed state.
- `artifacts/`: source data, coefficients, plots, scripts and protocol records with their original relative paths. Original-language notes are retained as source records.
- `../../structure-index.json`: coordinate/state provenance and aliases; `../../FILE_MANIFEST.json`: original paths and SHA256 checksums.

### Reports

- [artifacts/nnao-core-benchmarks-20261020/full/results.json](artifacts/nnao-core-benchmarks-20261020/full/results.json)

### Figures

- [artifacts/nnao-core-benchmarks-20261020/full/atomic-radial-density.pdf](artifacts/nnao-core-benchmarks-20261020/full/atomic-radial-density.pdf)
- [artifacts/nnao-core-benchmarks-20261020/full/atomic-radial-density.png](artifacts/nnao-core-benchmarks-20261020/full/atomic-radial-density.png)
- [artifacts/nnao-core-benchmarks-20261020/full/s22-counterpoise.pdf](artifacts/nnao-core-benchmarks-20261020/full/s22-counterpoise.pdf)
- [artifacts/nnao-core-benchmarks-20261020/full/s22-counterpoise.png](artifacts/nnao-core-benchmarks-20261020/full/s22-counterpoise.png)
- [artifacts/nnao-core-benchmarks-20261020/full/static-polarizability.pdf](artifacts/nnao-core-benchmarks-20261020/full/static-polarizability.pdf)
- [artifacts/nnao-core-benchmarks-20261020/full/static-polarizability.png](artifacts/nnao-core-benchmarks-20261020/full/static-polarizability.png)

## Provenance

Original GradSCF artifact directories: `nnao-core-benchmarks-20261020`.
Copied drivers are evidence snapshots and may retain original checkout paths. Running a numerical driver requires the corresponding GradSCF environment; this import did not rerun SCF, optimization or training.
