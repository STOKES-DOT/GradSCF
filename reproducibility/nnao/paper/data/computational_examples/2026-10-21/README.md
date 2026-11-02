# NNAO computational examples — 2026-10-21

Data-only handoff for the manuscript Codex. No manuscript text was edited or numerical calculation rerun.

## Case index

| Case | Cohort | Evidence status | XYZ records |
| --- | --- | --- | ---: |
| [HCN training geometry and HCN/HNC transition geometry](cases/01_hcn_mace_vs_free/README.md) | MACE-update-3351 | complete | 2 |
| [Rigid ethane torsion: MACE, free contraction and conventional bases](cases/02_ethane_torsion/README.md) | MACE-update-3351 | complete | 25 |
| [Ethane with a second carbon valence-p radial channel](cases/03_ethane_second_p/README.md) | free-contraction-only/update-3351-geometry | complete | 25 |
| [Carbon valence s/p radial functions during ethane torsion](cases/04_ethane_radial_response/README.md) | MACE-update-3351 | complete | 25 |
| [Restricted-HF ethane C-C stretching stress test](cases/05_ethane_cc_stretch/README.md) | MACE-update-3351 | complete_with_unconverged_endpoints | 56 |
| [Fixed STO-6G polarization controls on the ethane torsion scan](cases/06_ethane_fixed_polarization/README.md) | update-3351-and-fixed-basis-controls | complete | 25 |
| [Atomic radial density, S22 counterpoise and static polarizability](cases/07_radial_bsse_polarizability/README.md) | MACE-update-3351 | complete_with_failures | 35 |
| [Repaired representative S66x8, HB300SPXx10 and BH9 comparisons](cases/08_distance_curves_and_barriers/README.md) | MACE-update-3351/native-UKS-repair | complete | 166 |
| [Frozen 500-system model inference on 18 historical relative energies](cases/09_historical_relative_energies/README.md) | MACE-update-3351 | complete | 39 |
| [Nine S/M/L x SZP/DZP/TZP finite-budget free-contraction diagnostics](cases/10_nine_family_free_contractions/README.md) | historical-legacy-core6/free-contraction | complete_finite_budget | 13 |
| [Historical 500-system training loss and logarithmic energy-change trends](cases/11_training_loss_snapshot/README.md) | 500-system-training/through-epoch-596 | historical_time_limit | 500 |
| [Early methane contraction optimization and conventional-basis comparisons](cases/12_legacy_methane_design/README.md) | legacy-2026-09-prototypes | historical_examples | 2 |
| [Earlier aniline GPU4PySCF conventional-basis comparison](cases/13_legacy_aniline_basis_comparison/README.md) | legacy-2026-09-aniline-model | historical_examples | 1 |

## Reading order and evidence boundaries

For recent checkpoint-3351 examples, start with cases 01-09. Case 10 is a historical nine-family free-contraction diagnostic, case 11 is the training-history snapshot, and cases 12-13 are earlier methane/aniline prototypes.
Case 08 contains the corrected 852-case view and 31 independently checked density/orbital states. Case 07 retains all 12 QZ memory failures. Case 05 retains all five nonconverged free endpoints. Do not infer global variational minima, complete QZ coverage or physical RHF homolytic dissociation.

## Structures, units and model

There are 1420 case-local geometry/state XYZ exports, with 821 distinct exact coordinate/charge/spin signatures across cases. These are not counts of chemically distinct molecules. All are exported from explicitly Angstrom-labelled source fields with exact float roundtrips; original XYZ/SDF files are also preserved where present.
Absolute energies are normally Hartree and relative energies kcal/mol; raw columns retain their source conventions. Radial grids/moments use Bohr and polarizability uses Bohr^3. Positive historical binding energies and negative interaction energies must not be mixed without an explicit sign conversion.
The shared `models/update-3351/` contains an inference-only `parameters.npz` whose parameter bytes exactly match the frozen checkpoint, plus its original parameter/source identity, network configuration, fixed exponent candidate and 500 training / 26 calibration reference structures. `snapshot.json` is the original record; use the derived `inference-snapshot.json` with the smaller export. The original 390-MB training state stays at its recorded source; optimizer moments and saved training gradients are not needed for inference. This is evaluated update 3351, not a claimed final or best epoch-596 model.

## Reproducibility

`catalog.json` maps cases and evidence boundaries. `structure-index.csv`/`.json` maps every generated XYZ to original records, including molecular charge and 2S where available. `FILE_MANIFEST.json` maps imported files to original local/remote paths and hashes. `SHA256SUMS` covers imported and generated payloads.
Frozen per-atom basis exports allow independent same-basis recomputation. Full MACE inference or GradSCF driver replay additionally requires the recorded source/environment; original absolute paths are intentionally retained in raw records. The import itself performs file/schema/checksum checks only.
Large primitive-integral caches, JIT caches, opaque duplicate archives and oversized logs are omitted and listed in `EXCLUDED_FILES.json`. Coefficient checkpoints, optimization histories, radial arrays, explicit failure records and independent-check reports are retained. No full four-index ERI cache is added.

## Ongoing work

`contexts/ongoing-training/` is a dated snapshot of the remaining eight-family campaign and its source/configuration hashes. Its queued preparation/training jobs are not completed calculations. `contexts/basis-design/` preserves current structural definitions separately from the historical cases.

Original-language Markdown/log notes under `artifacts/` are source evidence, not manuscript prose. English case READMEs provide the handoff summaries.
