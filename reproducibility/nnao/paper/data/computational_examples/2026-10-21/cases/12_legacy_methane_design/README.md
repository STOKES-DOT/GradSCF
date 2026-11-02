# Early methane contraction optimization and conventional-basis comparisons

## Methods

Method: Legacy HF/NNAO prototype settings; retain each source run separately. Cohort: `legacy-2026-09-prototypes`. Exact settings, commands, hardware and original status fields remain in the imported source records. Numerical source files are copied without editing.

## Results

Original optimization records and fixed-geometry AO/Hamiltonian-size/energy comparisons are preserved as separate historical source runs.

Archive status: `historical_examples`. Exported geometry/state records: **2**.

## Interpretation boundary

Prototype basis/core/pool and network definitions differ from the later 500-system model. These rows must not be relabelled as update-3351 results.

## Files

- `structures/`: generated standard XYZ with Angstrom coordinates, charge and 2S in the comment. State fields omitted in one record are resolved only by a unique exact-coordinate/state match in the same case; any remaining unknown is explicit `None`.
- `artifacts/`: source data, coefficients, plots, scripts and protocol records with their original relative paths. Original-language notes are retained as source records.
- `../../structure-index.json`: coordinate/state provenance and aliases; `../../FILE_MANIFEST.json`: original paths and SHA256 checksums.

### Reports

- [artifacts/methane-basis-comparison-20260916/comparison.json](artifacts/methane-basis-comparison-20260916/comparison.json)
- [artifacts/methane-basis-comparison-expanded-20260916/artifacts/methane-szp663/comparison/comparison.json](artifacts/methane-basis-comparison-expanded-20260916/artifacts/methane-szp663/comparison/comparison.json)
- [artifacts/methane-basis-comparison-expanded-20260916/artifacts/methane-szp663/summary.json](artifacts/methane-basis-comparison-expanded-20260916/artifacts/methane-szp663/summary.json)
- [artifacts/methane-basis-comparison-expanded-20260916/artifacts/methane-szp663-160steps/comparison/comparison.json](artifacts/methane-basis-comparison-expanded-20260916/artifacts/methane-szp663-160steps/comparison/comparison.json)
- [artifacts/methane-basis-comparison-expanded-20260916/artifacts/methane-szp663-160steps/summary.json](artifacts/methane-basis-comparison-expanded-20260916/artifacts/methane-szp663-160steps/summary.json)
- [artifacts/methane-basis-comparison-expanded-20260916/comparison.json](artifacts/methane-basis-comparison-expanded-20260916/comparison.json)
- [artifacts/methane-core6-basis-comparison-20260916/comparison.json](artifacts/methane-core6-basis-comparison-20260916/comparison.json)
- [artifacts/methane-nnao-20260916/summary.json](artifacts/methane-nnao-20260916/summary.json)

### Figures

- [artifacts/methane-basis-comparison-20260916/basis_comparison.pdf](artifacts/methane-basis-comparison-20260916/basis_comparison.pdf)
- [artifacts/methane-basis-comparison-20260916/basis_comparison.png](artifacts/methane-basis-comparison-20260916/basis_comparison.png)
- [artifacts/methane-basis-comparison-expanded-20260916/comparison.pdf](artifacts/methane-basis-comparison-expanded-20260916/comparison.pdf)
- [artifacts/methane-basis-comparison-expanded-20260916/comparison.png](artifacts/methane-basis-comparison-expanded-20260916/comparison.png)
- [artifacts/methane-core6-basis-comparison-20260916/comparison.pdf](artifacts/methane-core6-basis-comparison-20260916/comparison.pdf)
- [artifacts/methane-core6-basis-comparison-20260916/comparison.png](artifacts/methane-core6-basis-comparison-20260916/comparison.png)
- [artifacts/methane-nnao-20260916/optimization.pdf](artifacts/methane-nnao-20260916/optimization.pdf)
- [artifacts/methane-nnao-20260916/optimization.png](artifacts/methane-nnao-20260916/optimization.png)

## Provenance

Original GradSCF artifact directories: `methane-nnao-20260916`, `methane-basis-comparison-20260916`, `methane-basis-comparison-expanded-20260916`, `methane-core6-basis-comparison-20260916`.
Copied drivers are evidence snapshots and may retain original checkout paths. Running a numerical driver requires the corresponding GradSCF environment; this import did not rerun SCF, optimization or training.
