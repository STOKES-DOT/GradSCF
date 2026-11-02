# Restricted-HF ethane C-C stretching stress test

## Methods

Method: All-electron DF-RHF, def2-universal-jkfit. Cohort: `MACE-update-3351`. Exact settings, commands, hardware and original status fields remain in the imported source records. Numerical source files are copied without editing.

## Results

All 56 distances were processed; 51 free endpoints met the coefficient-gradient criterion. Five endpoints at 2.70, 2.75, 2.80, 2.85 and 2.90 Angstrom did not. Both valid and unconverged records/geometries are retained.

Archive status: `complete_with_unconverged_endpoints`. Exported geometry/state records: **56**.

## Interpretation boundary

This follows a restricted RHF branch with rigid methyl groups, not physical homolytic dissociation. The nonconverged interval and nearby branch changes must remain explicit.

## Files

- `structures/`: generated standard XYZ with Angstrom coordinates, charge and 2S in the comment. State fields omitted in one record are resolved only by a unique exact-coordinate/state match in the same case; any remaining unknown is explicit `None`.
- `artifacts/`: source data, coefficients, plots, scripts and protocol records with their original relative paths. Original-language notes are retained as source records.
- `../../structure-index.json`: coordinate/state provenance and aliases; `../../FILE_MANIFEST.json`: original paths and SHA256 checksums.

### Reports

- [artifacts/nnao-ethane-dissociation-20261019/distance-000/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-000/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-001/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-001/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-002/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-002/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-003/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-003/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-004/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-004/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-005/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-005/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-006/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-006/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-007/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-007/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-008/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-008/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-009/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-009/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-010/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-010/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-011/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-011/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-012/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-012/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-013/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-013/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-014/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-014/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-015/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-015/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-016/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-016/summary.json)
- [artifacts/nnao-ethane-dissociation-20261019/distance-017/summary.json](artifacts/nnao-ethane-dissociation-20261019/distance-017/summary.json)

### Figures

- [artifacts/nnao-ethane-dissociation-20261019/carbon-radial-functions.pdf](artifacts/nnao-ethane-dissociation-20261019/carbon-radial-functions.pdf)
- [artifacts/nnao-ethane-dissociation-20261019/carbon-radial-functions.png](artifacts/nnao-ethane-dissociation-20261019/carbon-radial-functions.png)
- [artifacts/nnao-ethane-dissociation-20261019/carbon-radial-response.pdf](artifacts/nnao-ethane-dissociation-20261019/carbon-radial-response.pdf)
- [artifacts/nnao-ethane-dissociation-20261019/carbon-radial-response.png](artifacts/nnao-ethane-dissociation-20261019/carbon-radial-response.png)

## Provenance

Original GradSCF artifact directories: `nnao-ethane-dissociation-20261019`.
Copied drivers are evidence snapshots and may retain original checkout paths. Running a numerical driver requires the corresponding GradSCF environment; this import did not rerun SCF, optimization or training.
