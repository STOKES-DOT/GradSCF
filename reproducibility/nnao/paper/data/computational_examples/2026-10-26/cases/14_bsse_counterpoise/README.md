# Case 14 — BSSE and frozen-basis counterpoise

This is a focused data/figure handoff of existing calculations, assembled on
2026-10-26. Open **[index.html](index.html)** for the figure gallery. No SCF,
optimization or training was rerun. Numerical source files and frozen Gaussian
basis definitions are copied without editing; additional plots and CSV tables
are derived from the archived records.

## Contents and coverage

| Benchmark | Scope | Passed | Failed |
|---|---|---:|---:|
| S22 counterpoise | 22 dimers × 8 bases | 166 | 10 |
| S66×8 representative curves | 6 dimers × 8 distances × 6 bases | 288 | 0 |
| HB300SPX×10 representative curves | 6 dimers × 10 distances × 6 bases | 360 | 0 |

The ten S22 failures are ma-def2-QZVPP GPU-memory failures. They remain explicit
in the full table with empty energy cells and original error messages; no
missing energy is replaced by zero. The new S22 summary shows both seven
complete bases over **all 22 dimers** and all eight bases over the **same 12
successful dimers**. These are separate cohorts, never mixed in one average.

All calculations use spherical AOs, float64, native DF-HF and
`def2-universal-jkfit`. The original runs used an A100 80GB on n001 and 32 CPUs.
Per-case elapsed times remain in the tables. Parent-report total times include
other tasks/retries and must not be quoted as BSSE-only runtime.

The MACE model is the **historical update-3351 checkpoint**, original checkpoint
SHA256 `a7144d211a7b3d9fc4cb6899bf930072992e073bbc79d8cae282611682bc7c1f`.
It is not the current SZP/DZP/TZP family-training checkpoint. The existing
[shared inference model](../../../2026-10-21/models/update-3351/) is referenced
instead of duplicated; its parameter-export identity is checked during assembly.

## Definition, signs and units

At each fixed dimer geometry, let B be the dimer orbital basis and B_A/B_B its
fragment subsets. The **same dimer-predicted coefficients** are held fixed in
all five calculations. Ghost calculations remove the partner's nuclei and
electrons while retaining its orbital functions. Every calculation uses one
fixed dimer auxiliary basis/metric, with own-fragment DF factors obtained by
AO slicing.

\[
\Delta E_{\rm frozen}=E_{AB}^{B}-E_A^{B_A}-E_B^{B_B},\qquad
\Delta E_{\rm CP}=E_{AB}^{B}-E_A^{B}-E_B^{B},
\]
\[
\mathrm{BSSE}=E_A^{B_A}+E_B^{B_B}-E_A^{B}-E_B^{B}
=\Delta E_{\rm CP}-\Delta E_{\rm frozen}.
\]

Positive BSSE therefore raises the frozen-basis interaction energy after CP.
Raw energies are in Hartree; plots use kcal/mol with
`1 Eh = 627.5094740631 kcal/mol`. `scale` is a dimensionless source distance
factor, not a distance in Angstrom. XYZ coordinates use Angstrom.

For environment-dependent MACE bases, independently predicted isolated monomers
can have different contractions and geometries from the frozen dimer fragments.
The distance table therefore retains `model_interaction_kcal_mol` separately.
**Do not obtain the frozen CP curve by adding BSSE to this independent-model
interaction curve.** Both frozen uncorrected and CP interactions are exported
directly from their own five-energy records.

## Figures and exact plotted values

- [New S22 summary](figures/s22-bsse-summary.pdf): full-22 versus common-12
  comparisons, using logarithmic x axes for positive mean BSSE.
- [BSSE–distance overview](figures/distance-bsse-overview.png) and
  [12-system PDF](figures/distance-bsse-all-systems.pdf). Individual near-square
  figures are in `figures/distance-bsse/`, in PNG, PDF and SVG.
- Distance figures use **logarithmic y axes** to retain the long-distance tails;
  every plotted value is positive. Markers are the original eight/ten computed
  points; connecting lines introduce no new calculated or interpolated samples.
- Original S22 and S66/HB300 frozen-CP figures are retained alongside the new
  figures. Their archived reference interactions contain electronic-method
  differences as well as basis differences; BSSE is not a binding-energy error.
- [plot-values.json](plot-values.json) records all 648 distance BSSE values and
  15 cohort/basis means actually supplied to the new plots.

## Data and structures

- [S22: all 176 attempted cases](tables/s22-counterpoise-all-cases.csv), including
  all five energies for each successful calculation, AO counts, statuses,
  elapsed times, source links and frozen basis paths.
- [Distance: all 648 cases](tables/distance-counterpoise-all-cases.csv), including
  independent-model interactions/monomer energies and upstream references.
- [Cohort summaries](tables/s22-summary.csv) and
  [ten failed cases](tables/failed-cases.csv).
- [structure-index.json](structure-index.json): 130 dimer geometries, fragment
  partitions and XYZ paths. Fragment XYZ files retain **dimer geometries**.
  Independent reference monomer geometries are stored separately where used.
- `frozen-bases/`: 824 exact per-case Gaussian basis files, including the ten
  failed attempts. Original geometry, exponents and contractions are retained.
- `source/`: original reports, configurations, datasets and protocol/repair
  records. Broad parent reports also contain atomic/polarizability or BH9
  results; the focused tables select only `bsse` and `curve` tasks.

## Verification and limitations

`VERIFY_REPORT.json` records source/table equality, file hashes, all 814
successful five-energy CP identities, nested own/ghost variational checks,
all plotted values and XYZ atom counts. Selected repaired basis hashes are
checked against the recorded repair configuration. `SOURCE_MANIFEST.json`
retains original paths and checksums; `FILE_MANIFEST.json` records payload
checksums. No finite differences are used.

The distance dataset contains twelve representative dimers, not all of S66 or
HB300SPX. Numerical convergence and selected real internal-stability checks
do not certify global HF minima or external spin/complex stability. The
copied independent-check report covers a selected broader repair subset, not
an independent validation of every five-energy CP calculation.

To redraw figures from this directory:

```bash
/opt/anaconda3/bin/python plot_bsse.py
```

To recheck the unchanged exported package:

```bash
/opt/anaconda3/bin/python verify_bundle.py .
```

Old dated archives, existing `output/` figures, manuscript text and training
state are preserved. This supplement provides evidence to the manuscript
thread and does not insert any claims into the paper.
