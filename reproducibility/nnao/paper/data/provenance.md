# Data provenance record

This directory contains data imported from GradSCF and demonstration results generated with the local GradSCF checkout.
The training files below were copied on 2026-10-08; local calculations are recorded separately.
**Do not manually edit numerical data files**; corrections should be recomputed with the corresponding script or reimported.

| File in this repository | Source |
| --- | --- |
| `pilot/metrics.csv` | `/Users/jiaoyuan/Documents/GitHub/GradSCF/artifacts/nnao-gmtkn55-pilot-20261007/metrics.csv` |
| `pilot/PILOT_README.md` | `.../nnao-gmtkn55-pilot-20261007/README.md` |
| `pilot/DIAGNOSTICS.md` | `.../nnao-gmtkn55-pilot-20261007/DIAGNOSTICS.md` |
| `stage1-training/RUN_SPEC.md` | `.../nnao-gmtkn55-first-training-20261007/RUN_SPEC.md` |
| `stage2/STAGE2_RUN_SPEC.md` | `.../nnao-stage2-20261007/RUN_SPEC.md` |
| `stage2/REPAIR_REPORT.md` | `.../nnao-stage2-20261008-repair/REPAIR_REPORT.md` |

(In this table, `.../` = `/Users/jiaoyuan/Documents/GitHub/GradSCF/artifacts`.)

## Not yet imported (waiting for remote jobs to finish)

- 93537 gate: complete 200/50 numerical admission results → `data/stage2/gate/`
- 93538 formal training: 1000-epoch training curves and energies of the 50 held-out structures,
  plus the `best_validation.npz` summary → `data/stage2/training/`
- First-round training (93108): `training_report` summary → `data/stage1-training/`

Record the corresponding entries in checksums.sha256 together with each import.

## Local classical-basis BSSE examples (2026-10-08)

- Geometry: bsse/s22_hbond_geometries.json, extracted from the local ASE 3.26.0
  S22 equilibrium positions. Water, ammonia, and formic-acid dimers; the dimer geometries are fixed.
- Data: bsse/classical_hf/energies.json and summary.csv, generated with local
  GradSCF native CPU / float64 / spherical RHF. 3 systems ×
  3 bases (STO-3G, 3-21G, def2-SVP) × 5 counterpoise components.
  The original run script is frozen at bsse/classical_hf/run_script.py; its SHA-256
  matches script_sha256 in energies.json.
- Independent validation: all 45 components were compared against independent integrals/SCF from PySCF 2.9.0.
  energies.json contains the original components, integral errors, electron-count and convergence checks,
  source and native-library SHA-256 values, original commands, and timings.
- Initial readable example:
  [GradSCF/examples/basis/bsse_counterpoise.py](/Users/jiaoyuan/Documents/GitHub/GradSCF/examples/basis/bsse_counterpoise.py).
  The first rewrite adopted a PySCF-examples-style top-level script, removed CLI
  and archiving logic, and retained spherical bases and five explicit energies; the current API version is described below.
- Rewrite validation: bsse/simple_example_run.log and
  bsse/simple_example_validation.json; all 45 components were rerun, and the nine
  raw/CP/BSSE groups agreed with the validated original batch (log values printed to 9 decimal places).
- Hardware: bsse/hardware.json, Apple M4 Pro / 24 GB, JAX CPU;
  BLAS/OpenMP environment thread counts were set to 1. Timings include initial JAX compilation
  and PySCF comparisons and are not used to claim a backend performance advantage.

These are counterpoise demonstrations for traditional fixed bases, independent of the NNAO training results.
The NNAO learned-basis ghost/CP path remains unvalidated.

### Public RHF API version (2026-10-08)

The user requested a direct call to scf.RHF(mol).kernel(). The high-level interface currently supports only
Cartesian AOs, so the example explicitly selects cart=True and recomputes the results.
Each def2-SVP d shell has six Cartesian components; these values must not be mixed with the earlier spherical results.
STO-3G / 3-21G contain only s/p shells, and their results in this comparison agree between the two representations.

To support ghosts in high-level RHF, the HF-ghost failure caused by constructing an unnecessary
XC grid before SCF was repaired: KeyError: 0. The repair uses empty quadrature only for pure HF with zero-nuclear-charge centers.
The basis is constructed independently from the current spec to avoid reusing incorrect nuclear charges.
Ordinary HF/DFT paths retain their existing grid logic. The validation scope here is HF energies.

- Original five-component Cartesian energies: bsse/cartesian_reference/energies.json and
  summary.csv; all 45 components were again compared against independent PySCF results.
- High-level API example output: bsse/rhf_api_example_run.log (staged validation) and
  bsse/rhf_api_installed_run.log (actual checkout).
- Comparison against the independent Cartesian baseline: bsse/rhf_api_validation.json.
  Result comments at the end of the code use that Cartesian baseline and retain the fixed geometry and kcal/mol units.
- Existing bsse/classical_hf files remain the spherical-version archive; original data are retained and not overwritten.

### Differentiable BSSE contraction-optimization example (2026-10-08)

The user requested BSSE optimization using differentiability while keeping the code concise and using high-level
APIs where possible, and explicitly prohibited src changes. This example adds only examples/basis content and regression tests;
the separately staged source API was not installed.

- Example:
  [optimize_bsse_contractions.py](/Users/jiaoyuan/Documents/GitHub/GradSCF/examples/basis/optimize_bsse_contractions.py).
  It uses the existing public implicit shared-orbital SCF solver, with equal-spin
  occupations giving closed-shell HF. Native primitive data are fixed,
  JAX supplies contraction and normalization derivatives, and jax.value_and_grad supplies optimization gradients.
- System: S22 water dimer / STO-3G; fixed geometry, exponents, O 1s core contractions,
  and AO dimension. Six scale-reduced variables for O 2s/O 2p/H 1s are shared between the two fragments.
- Protocol: SLSQP minimizes counterpoise BSSE; the six free coefficients remain within
  ±0.15 of their reference values, with an own-basis fragment-energy increase budget of 1e-4 Ha for each fragment.
- Results: bsse/contraction_optimization_result.json stores the five original component energies,
  coefficients, optimization status, and diagnostics; logs and the final example snapshot are also under bsse/.
- Verification: bsse/contraction_installed_test.log (2 passed) checks the constrained optimization
  and reconverged directional FD. bsse/contraction_optimized_pyscf_check.json
  checks the five optimized component energies against independent PySCF values.
- Complete run provenance: bsse/contraction_optimization_provenance.json, including
  CPU/float64, software versions, source/example/geometry hashes, and elapsed time for the actual validation workflow.

This objective controls basis borrowing only. The CP interaction energy also changes substantially;
a reduction in BSSE is not used to infer improved interaction-energy accuracy or shared-model generalization.

## Integrity hashes

Stage 2 scientific-source SHA-256 (see REPAIR_REPORT.md):
`7f066083658982a552fda47e0144e762796f69788c6a1f1c7e70c77fbfec403a`

## English-language synchronization note

The imported Markdown source records have been translated into English; this language synchronization is not a new calculation or result. Original Markdown and pilot CSV records are retained in the ignored local backups under `.local/english-sync-originals/`.
The pilot CSV conversion changes display labels only; its label mapping and provenance are to be recorded in `data/pilot/english-labels-provenance.json`. Numerical fields, source identifiers, and recorded validation boundaries are preserved. This Markdown translation does not itself modify CSV or JSON files.
