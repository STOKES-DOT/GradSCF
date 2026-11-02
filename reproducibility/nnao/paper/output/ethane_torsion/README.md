# Ethane methyl rotation

- [Relative-energy curve with rotation arrow (PDF)](ethane-torsion-rotation.pdf)
- [PNG preview](ethane-torsion-rotation.png)
- [SVG figure](ethane-torsion-rotation.svg)
- [0, 60, and 120 degree structures](rotation-structures.png)
- [Plotted data](plotted-points.csv)
- [Source and camera manifest](plot-report.json)
- [Verification](verification.json)

This is the archived rigid ethane C2H6 scan from 0 to 120 degrees in 5-degree
steps. Each method uses its own 0-degree energy as zero. The seven curves are
read from the existing combined comparison table without changing their values.
The expanded 50-AO carbon-p curve is free-contraction-only. `STO-6G + pol.` is the
custom fixed STO-6G+C(d)+H(p) control, not a separate standard published basis.

The C-C bond is horizontal in the molecular view. The left methyl group stays
fixed, and the arrow identifies the positive rotation of the right methyl group.
Only source atom indices 5, 6, and 7 (zero-based hydrogen indices) rotate. The
arrow follows the source right-hand Rodrigues rotation about C0->C1 and is
projected through the same camera as the molecule. Its sign was checked against
the archived 0-to-5-degree coordinate change. The displayed source geometries
are unchanged; camera transforms are rigid rotations only.

The plot uses the same square canvas, full black frame, inward ticks, internal
frameless legend, and highlighted blue cc-pVTZ curve as the case-08 figures.
The PNG/SVG renders with rotation arrows are saved under `structures/`;
the original XYZ paths and hashes are retained in the source manifest.
No electronic calculations are run.

Regenerate from the repository root:

```bash
/opt/anaconda3/bin/python scripts/plot_ethane_torsion.py
```
