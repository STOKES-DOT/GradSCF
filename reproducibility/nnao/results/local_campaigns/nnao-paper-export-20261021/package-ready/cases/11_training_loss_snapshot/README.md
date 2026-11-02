# Historical 500-system training loss and logarithmic energy-change trends

## Methods

Method: All-electron DF-RHF, def2-universal-jkfit. Cohort: `500-system-training/through-epoch-596`. Exact settings, commands, hardware and original status fields remain in the imported source records. Numerical source files are copied without editing.

## Results

History reaches epoch 596; the saved update 5969 is a partial epoch 597. Epochs 5-596 supply 592 complete online means. Epoch 4 is missing, and epochs 0-3 use the earlier full-evaluation policy.

Archive status: `historical_time_limit`. Exported geometry/state records: **500**.

## Interpretation boundary

Online means combine pre-update energies at ten parameter versions. They are not final-vector full-set scores, best-model criteria or CBS errors. The evaluation checkpoint used in recent examples is update 3351, not update 5969.

## Files

- `structures/`: generated standard XYZ with Angstrom coordinates, charge and 2S in the comment. State fields omitted in one record are resolved only by a unique exact-coordinate/state match in the same case; any remaining unknown is explicit `None`.
- `artifacts/`: source data, coefficients, plots, scripts and protocol records with their original relative paths. Original-language notes are retained as source records.
- `../../structure-index.json`: coordinate/state provenance and aliases; `../../FILE_MANIFEST.json`: original paths and SHA256 checksums.

### Reports

- [artifacts/nnao-mace500-loss-20261019/summary.json](artifacts/nnao-mace500-loss-20261019/summary.json)

### Figures

- [artifacts/nnao-mace500-loss-20261019/energy-log-trends.pdf](artifacts/nnao-mace500-loss-20261019/energy-log-trends.pdf)
- [artifacts/nnao-mace500-loss-20261019/energy-log-trends.png](artifacts/nnao-mace500-loss-20261019/energy-log-trends.png)
- [artifacts/nnao-mace500-loss-20261019/training-loss.pdf](artifacts/nnao-mace500-loss-20261019/training-loss.pdf)
- [artifacts/nnao-mace500-loss-20261019/training-loss.png](artifacts/nnao-mace500-loss-20261019/training-loss.png)

## Provenance

Original GradSCF artifact directories: `nnao-mace500-loss-20261019`.
Copied drivers are evidence snapshots and may retain original checkout paths. Running a numerical driver requires the corresponding GradSCF environment; this import did not rerun SCF, optimization or training.
