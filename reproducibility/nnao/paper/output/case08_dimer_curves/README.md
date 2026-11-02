# Case 08 dimer curves with molecular structures

Twelve independent figures cover the six S66x8 and six HB300SPXx10
representative dimers. Each figure is available as PNG, PDF, and SVG.

- [Main curves: 12-page PDF](dimers-model-with-structures.pdf)
- [Main-curve overview](overview-model.png)
- [BSSE-corrected interactions with fixed dimer basis: 12-page PDF](dimers-frozen-cp-with-structures.pdf)
- [CP overview](overview-frozen-cp.png)
- [Individual-figure index](index.html)
- [Exact plotted values](plotted-points.csv)
- [Sources, layout, and representative structures](plot-report.json)
- [Verification record](verification.json)

The square 6.3-inch figures reuse the approved BH9 figure style: DejaVu Sans,
complete black frames, inward ticks, frameless legends inside the axes, black
reference curves, and red dashed NNAO/MACE curves. Large xyzrender molecular
insets show the archived distance-scale-1.00 geometries. No titles, formulas,
or explanatory paragraphs are placed inside the figures.
The cc-pVTZ curve is emphasized with a saturated blue solid line (2.8 pt)
and larger circular data markers (4.2 pt).

Horizontal coordinates are the dimensionless source distance scale factors,
not absolute distances in Angstrom. The lines connect the original eight-point
S66x8 or ten-point HB300SPXx10 data; no new interpolated data are introduced.
The representative benzene `TS` label denotes the T-shaped dimer.

`model/` uses the recorded `model_interaction_kcal_mol`: the dimer and isolated
monomers are evaluated independently. `frozen-cp/` uses the recorded
`cp_interaction_kcal_mol`: the dimer contractions remain fixed, and the own/ghost
monomer diagnostics share the dimer auxiliary metric. These CP energies are
read directly, not computed by adding BSSE to the main interaction curves.
Both versions retain the original reference values; comparison with those
references includes electronic-method effects in addition to basis effects.

All 1080 plotted values match the archived table exactly. The molecular images
are rasterized from the existing xyzrender SVGs; no source XYZ, geometry,
electronic energy, SCF solution, or model parameters are changed.

Regenerate from the repository root with the existing local environment:

```bash
/opt/anaconda3/bin/python scripts/plot_case08_dimers.py
```
