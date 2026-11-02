# NNAO training set: 500 structures

This dataset is the exact structure pool used by the frozen
`nnao-family500-20261021` campaign, exported for the paper workspace on
2026-10-26. It also matches the existing update-3351 model's training-reference
file **byte for byte**. Remote L-DZP and M-TZP training metadata were checked
against the same complete ordered structure list. No data selection, geometry
optimization or electronic-structure calculation was performed during export.

## Composition

| Property | Value |
|---|---:|
| Structures | 500 |
| GMTKN55-v2 | 400 |
| QH9Stable | 100 |
| Atoms per structure | 1–30 |
| Total atoms | 5402 |
| Observed elements | 25 |
| Molecular/complex structures | 499 |
| Isolated-atom reference structures | 1 |
| Charge / spin field | 0 / 0 for every structure |

The observed element vocabulary is H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg,
Al, Si, P, S, Cl, Ar, K, Ca, Ge, As, Se, Br and Kr. Exact structure/atom counts
per element are in [element-coverage.csv](element-coverage.csv); dataset counts
and atom-count distributions are also exported separately.

## Files

- **[structure-index.csv](structure-index.csv)**: 500 rows in original training
  order, with structure IDs, formula, atom/electron counts, charge, spin,
  multiplicity, source dataset/subset/path, weights, XYZ paths and checksums.
- **[all-structures.xyz](all-structures.xyz)**: concatenated 500-frame XYZ file.
- **`structures/`**: 500 individual XYZ files. Numeric filename prefixes are
  one-based; `training_index_zero_based` records the actual list index.
- **[atoms.csv](atoms.csv)**: 5402 atomic coordinate rows, including the parent
  structure ID, zero-based atom index, atomic number, symbol and x/y/z.
- **[references.json](references.json)**: untouched source definition and all
  original per-structure metadata, including the original readiness flags.
- `summary.json`, `source-counts.csv`, `element-coverage.csv`,
  `atom-count-distribution.csv`: descriptive counts only.
- `remote-lineage.json`, `provenance.json`, `FILE_MANIFEST.json`,
  `VERIFY_REPORT.json`: input identity, remote metadata checks and file checks.

## Units and training meaning

Coordinates are in **Angstrom**. XYZ atom order and coordinates are preserved;
17 significant decimal digits give an exactly checked float round-trip.
XYZ comment lines retain structure ID, training index, charge, spin and
multiplicity. The spin field follows the electronic-structure convention
`N_alpha - N_beta = 2 M_s`; the campaign uses the restricted closed-shell path.

All 500 structures have equal unit training weights. The objective minimizes
converged SCF energies variationally; the package supplies geometries and
states, not fitted reference-energy/force labels. These are training
structures, not an independent benchmark or held-out validation split.

The source JSON's `basis_family` is the L-SZP template value. The family
campaign overrides that field for each SZP/DZP/TZP basis while retaining the
identical structure pool. Structures are therefore separated from basis
parameters in this handoff. Seed-0 epoch shuffling is a training operation;
the export retains the original ordered list, not a shuffled epoch.

Original `scf_validated`, `training_ready`, `required_support` and selection
flags are preserved as upstream selection/prevalidation metadata. They must
not be treated as current per-job SCF status. Export verification checks
geometry/state/identity only and does not certify convergence of all 500
structures for every evolving checkpoint or basis.

## Verification and lineage

The original `references.json` SHA256 is
`5e1b21c837f318da1bedc0b55f3f2bbfefc45218390928ce1b217db0607dc299`.
It matches the current remote campaign input and the historical paper-model
training references. The remote checks also confirm complete ordered
structure equality in actual training metadata after the basis-family
template override. See `remote-lineage.json` for the checked timestamp.

The export verifies 500 unique structure IDs, 500 unique original geometry
signatures, atom counts, electron counts, charge/spin labels, original order,
all 5402 coordinate rows, multi-frame/individual XYZ agreement and payload
checksums. It does not infer bonds, rotate structures or substitute missing
source metadata.

```bash
/opt/anaconda3/bin/python verify_training_data.py .
```

Existing model weights, basis parameters, benchmark archives, manuscript text,
figures and live training state are preserved.
