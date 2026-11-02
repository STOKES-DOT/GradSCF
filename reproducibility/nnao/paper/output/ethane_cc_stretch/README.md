# Ethane C-C stretching - corrected 2026-10-22 data

- [PDF](ethane-cc-stretch.pdf)
- [PNG preview](ethane-cc-stretch.png)
- [SVG](ethane-cc-stretch.svg)
- [Molecular schematic](ethane-stretch-structure.png)
- [Plotted values and status flags](plotted-points.csv)
- [Source and display manifest](plot-report.json)
- [Verification](verification.json)

The figure now uses the canonical `curves.csv` in the corrected
`data/computational_examples/2026-10-22/` supplement. It contains 56 distances
and nine curves: 504 accepted geometry/basis energy points. The complete
MACE and free-contraction data have been rechecked, and all 56 free endpoints
meet the recorded coefficient-gradient criterion. The previous five missing
line segments and their unconverged markers are superseded by corrected
values, without interpolation or smoothing.

The archive attributes the historical MACE energy jump near 2.75 Angstrom
to selection of a higher, internally stable hcore-initialized RHF solution.
The corrected energies replace those values; the MACE update-3351 model,
predicted coefficients, exponent pool, fixed core and geometries are unchanged.

Seven conventional bases are now included: STO-3G, STO-6G, the custom fixed
STO-6G+C(d)+H(p) control, 3-21G, 6-31G(d,p), cc-pVDZ and cc-pVTZ. The two
NNAO curves are MACE prediction and independent free-contraction optimization.
The cc-pVTZ curve uses the approved saturated blue, 2.8-pt solid line.

This remains a restricted closed-shell DF-RHF rigid stretching stress test,
not a physical homolytic dissociation curve or a certified global RHF minimum.
The two methyl fragments are translated symmetrically while their internal
geometries are held fixed. The C-C bond is horizontal in the display, and two
outward arrows indicate increasing separation. The source geometry is unchanged.

Each method uses its own energy at R0 = 1.5259763767561005 Angstrom as zero.
The Hartree-to-kcal/mol conversion factor is 627.5094740631. Relative energies
are read directly from the corrected source table and checked against the
absolute energies. The figure retains the approved square canvas, full black
frame, inward ticks and internal frameless legend.

No SCF, contraction optimization, or training is rerun. Regenerate from the
repository root using:

```bash
/opt/anaconda3/bin/python scripts/plot_ethane_cc_stretch.py
```
