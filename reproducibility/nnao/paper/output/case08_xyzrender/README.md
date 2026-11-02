# Case 08 molecular structures

Rendered from the archived Angstrom XYZ files with xyzrender 0.3.8.

| Dataset | Test groups | Individual structures |
| --- | ---: | ---: |
| S66x8 representative subset | 6 | 60 |
| HB300SPXx10 representative subset | 6 | 72 |
| BH9 representative subset | 9 | 34 |
| Total | 21 | 166 |

- [HTML gallery](index.html): all structures, original XYZ links, PNG and SVG files.
- [All-group overview](overview.png): scale-1.00 dimers and the nine BH9 transition states.
- [S66x8 overview](S66x8-overview.png).
- [HB300SPXx10 overview](HB300SPXx10-overview.png).
- [BH9 overview](BH9-overview.png).
- [24-page PDF atlas](../pdf/case08-structures.pdf): three dataset overviews followed by 21 group sheets.
- [Render manifest](manifest.json): source paths and hashes, electronic states, display settings, and output hashes.
- [Gallery report](gallery-report.json): coverage and generated files.

All source XYZ bytes are preserved. The render applies rigid display transforms
only. All hydrogens are shown. Each noncovalent group shares one camera rotation
and physical display scale. BH9 species are individually oriented and share a
physical scale within each reaction. Reactant and product fragments are rendered
from their separate XYZ files; no combined geometry is constructed. Individual
PNG files use a 900-pixel canvas; the SVG files retain vector graphics.
The gallery thumbnails and group sheets remove whitespace with one shared
foreground crop per group, preserving the relative display scale. Overview
panels frame each representative separately; sizes across those panels are
not a quantitative size comparison.

The ball-and-stick lines are distance-perceived visual guides with single-line
bonds. Cross-fragment covalent lines are suppressed for the noncovalent dimers.
No transition-state bond assignment or electronic calculation is performed.
The S66 name `Benzene-Benzene_TS` means a T-shaped dimer; only the BH9 `TS` role
denotes a transition state. Distance scale factors are dimensionless. Electronic
state labels report charge and `2S`, as recorded in the input index.

To regenerate from the repository root using the existing local environment:

```bash
/opt/anaconda3/bin/python scripts/render_case08_structures.py
/opt/anaconda3/bin/python scripts/build_case08_gallery.py
```

Rendering and gallery construction do not rerun SCF, optimization, or training.
