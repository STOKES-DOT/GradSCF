# Nine S/M/L x SZP/DZP/TZP finite-budget free-contraction diagnostics

## Methods

Method: All-electron DF-RHF, def2-universal-jkfit. Cohort: `historical-legacy-core6/free-contraction`. Exact settings, commands, hardware and original status fields remain in the imported source records. Numerical source files are copied without editing.

## Results

Seven structures x nine families = 63 independently checked cases. Four starts per case have different optimizer outcomes retained in the status table. Maximum independent energy difference: 3.93106e-9 Eh.

Archive status: `complete_finite_budget`. Exported geometry/state records: **20**.

## Interpretation boundary

Historical numerical core6 and candidate pools differ from the later BSE-reference-core cohort. Many starts reached budgets or were rejected. S/M/L pools are not strictly nested; this is not a global basis-capacity limit. TZP adds valence radial columns, not the angular structure of cc-pVTZ.

## Files

- `structures/`: generated standard XYZ with Angstrom coordinates, charge and 2S in the comment. Missing state information is explicitly `None`, not assigned a guessed state.
- `artifacts/`: source data, coefficients, plots, scripts and protocol records with their original relative paths. Original-language notes are retained as source records.
- `../../structure-index.json`: coordinate/state provenance and aliases; `../../FILE_MANIFEST.json`: original paths and SHA256 checksums.

### Reports

- [artifacts/free-family9-20261014/comparison.json](artifacts/free-family9-20261014/comparison.json)
- [artifacts/free-family9-20261014/m-calibration/summary.json](artifacts/free-family9-20261014/m-calibration/summary.json)
- [artifacts/free-family9-20261014/results/l-1/gpu4-verification/comparison.json](artifacts/free-family9-20261014/results/l-1/gpu4-verification/comparison.json)
- [artifacts/free-family9-20261014/results/l-1/summary.json](artifacts/free-family9-20261014/results/l-1/summary.json)
- [artifacts/free-family9-20261014/results/l-2/gpu4-verification/comparison.json](artifacts/free-family9-20261014/results/l-2/gpu4-verification/comparison.json)
- [artifacts/free-family9-20261014/results/l-2/summary.json](artifacts/free-family9-20261014/results/l-2/summary.json)
- [artifacts/free-family9-20261014/results/l-3/gpu4-verification/comparison.json](artifacts/free-family9-20261014/results/l-3/gpu4-verification/comparison.json)
- [artifacts/free-family9-20261014/results/l-3/summary.json](artifacts/free-family9-20261014/results/l-3/summary.json)
- [artifacts/free-family9-20261014/results/m-1/gpu4-verification/comparison.json](artifacts/free-family9-20261014/results/m-1/gpu4-verification/comparison.json)
- [artifacts/free-family9-20261014/results/m-1/summary.json](artifacts/free-family9-20261014/results/m-1/summary.json)
- [artifacts/free-family9-20261014/results/m-2/gpu4-verification/comparison.json](artifacts/free-family9-20261014/results/m-2/gpu4-verification/comparison.json)
- [artifacts/free-family9-20261014/results/m-2/summary.json](artifacts/free-family9-20261014/results/m-2/summary.json)
- [artifacts/free-family9-20261014/results/m-3/gpu4-verification/comparison.json](artifacts/free-family9-20261014/results/m-3/gpu4-verification/comparison.json)
- [artifacts/free-family9-20261014/results/m-3/summary.json](artifacts/free-family9-20261014/results/m-3/summary.json)
- [artifacts/free-family9-20261014/results/s-1/gpu4-verification/comparison.json](artifacts/free-family9-20261014/results/s-1/gpu4-verification/comparison.json)
- [artifacts/free-family9-20261014/results/s-1/summary.json](artifacts/free-family9-20261014/results/s-1/summary.json)
- [artifacts/free-family9-20261014/results/s-2/gpu4-verification/comparison.json](artifacts/free-family9-20261014/results/s-2/gpu4-verification/comparison.json)
- [artifacts/free-family9-20261014/results/s-2/summary.json](artifacts/free-family9-20261014/results/s-2/summary.json)

### Figures


## Provenance

Original GradSCF artifact directories: `free-family9-20261014`.
Copied drivers are evidence snapshots and may retain original checkout paths. Running a numerical driver requires the corresponding GradSCF environment; this import did not rerun SCF, optimization or training.
