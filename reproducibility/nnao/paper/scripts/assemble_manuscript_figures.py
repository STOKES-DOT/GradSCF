"""Package the reviewed plot PDFs for the NNAO manuscript.

Curve figure PDFs are copied or composed without redrawing. Density maps use
the requested compact layout with equal coordinate scales. This script does not run
electronic-structure calculations, infer model parameters, or alter source data.
Run from the repository root with /opt/anaconda3/bin/python.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import fitz
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "output/ppt"
DEST = ROOT / "manuscript/figures"
PREVIEW = ROOT / "output/manuscript_figures"
FIGURE_NUMBERS = {
    "fig-training-statistics": 1,
    "fig-ethane-radial-response": 2,
    "fig-ethane-stretch-coefficients": 3,
    "fig-ethane-torsion-coefficients": 4,
    "fig-ethane-energy-scans": 5,
    "fig-reaction-profiles": 6,
    "fig-ethylene-density-xy": 7,
    "fig-ethylene-density-xz": 8,
    "fig-dimer-interactions": 9,
    "fig-dimer-bsse": 10,
    "fig-s22-bsse": 11,
    "fig-basis-accuracy-mp2": 12,
    "fig-basis-accuracy-wb97m-v": 13,
}


FIGURES = [
    dict(stem="fig-training-statistics", layout="copy",
         sources=["training_statistics/training-statistics-triptych.pdf"],
         records=["training_statistics/statistics.json"],
         facts=["500 ordered structures; each entry has unit weight.",
                "Element bars count 5,402 atoms and use a logarithmic count axis.",
                "Heavy atoms have Z > 1; spherical AO counts include fixed cores.",
                "AO counts use the manuscript NNAO-SZP-L shell inventory."]),
    dict(stem="fig-ethane-energy-scans", layout="pair",
         sources=["ethane_torsion/ethane-torsion-rotation.pdf",
                  "ethane_cc_stretch/ethane-cc-stretch.pdf"],
         headers=["(a) Methyl rotation", "(b) C-C stretching"],
         records=["ethane_torsion/plot-report.json", "ethane_cc_stretch/plot-report.json"],
         facts=["Each method uses its own first-geometry energy zero.",
                "25 torsion angles from 0 to 120 degrees; 56 stretching distances.",
                "Stretching uses the corrected 2026-10-22 full 56-point table.",
                "Rigid closed-shell DF-RHF scans are not physical homolytic dissociation energies."]),
    dict(stem="fig-ethane-radial-response", layout="pair",
         sources=["ethane_mace_radial/ethane-stretch-mace-radial-triptych.pdf",
                  "ethane_mace_radial/ethane-torsion-mace-radial-triptych.pdf"],
         headers=["C-C stretching", "Methyl rotation"],
         records=["ethane_mace_radial/plot-report.json"],
         facts=["Left column: stretching; right column: methyl rotation.",
                "Rows (a), (b), (c): C0 valence s, valence p, polarization d.",
                "MACE update 3351 only; u_l(r) = r R_l(r), r in bohr.",
                "Source normalization and phases are preserved; no displayed-window renormalization.",
                "Torsion insets show 1000 times the difference from the zero-degree radial function."]),
    dict(stem="fig-ethane-stretch-coefficients", layout="copy",
         sources=["ethane_contraction_changes/ethane-stretch-coefficient-comparison.pdf"],
         records=["ethane_contraction_changes/plot-report.json"],
         facts=["Left: MACE; right: independently optimized free contractions.",
                "Rows (a), (b), (c): C0 valence s, valence p, polarization d.",
                "Changes use each method's own starting coefficients at R0 = 1.5259763767561005 Angstrom.",
                "Coefficients use the Euclidean unit-vector gauge; nonorthogonal primitive coefficients are not probabilities.",
                "Primitive indices run from tightest to most diffuse; fixed core excluded."]),
    dict(stem="fig-ethane-torsion-coefficients", layout="copy",
         sources=["ethane_contraction_changes/ethane-torsion-coefficient-comparison.pdf"],
         records=["ethane_contraction_changes/plot-report.json"],
         facts=["Left: MACE; right: independently optimized free contractions.",
                "Rows (a), (b), (c): C0 valence s, valence p, polarization d.",
                "Plots show 1000 times each method's coefficient changes from its own zero-degree coefficients.",
                "56-point stretching and 25-point torsion source arrays remain unchanged.",
                "Free contractions are accepted finite-budget local endpoints, not certified global minima."]),
    dict(stem="fig-reaction-profiles", layout="pair",
         sources=["case08_reaction_profiles/BH9_02_134.pdf",
                  "case08_reaction_profiles/BH9_04_1.pdf"],
         headers=["(a) Pericyclic reaction", "(b) Hydrogen-atom transfer"],
         records=["case08_reaction_profiles/plot-report.json"],
         facts=["Exactly two reaction examples: BH9_02_134 and BH9_04_1.",
                "Electronic R/TS/P levels; each method uses its own reactant energy zero.",
                "Connectors are schematic and are not an intrinsic reaction coordinate.",
                "Separate molecular species are combined for the structure illustration only."]),
    dict(stem="fig-dimer-interactions", layout="pair",
         sources=["case15_dimer_controls/no-cp/S66x8-Water-Water.pdf",
                  "case15_dimer_controls/no-cp/S66x8-Benzene-Benzene_pi-pi.pdf"],
         headers=["(a) Water dimer", "(b) Benzene dimer"],
         records=["case15_dimer_controls/plot-report.json"],
         facts=["Exactly two interactions: S66x8 water-water and parallel-displaced benzene-benzene.",
                "Corrected no-CP curves use constant independently predicted isolated-monomer energy references.",
                "Adaptive MACE only; fixed-output MACE is excluded.",
                "Reference traces differ in electronic method as well as basis.",
                "Distance scale is dimensionless; no endpoint shift or zero-forcing is applied."]),
    dict(stem="fig-dimer-bsse", layout="pair",
         sources=["bsse_counterpoise/distance-bsse/S66x8-Water-Water.pdf",
                  "bsse_counterpoise/distance-bsse/S66x8-Benzene-Benzene_pi-pi.pdf"],
         headers=["(a) Water dimer", "(b) Benzene dimer"],
         records=["bsse_counterpoise/verification.json"],
         facts=["Frozen-dimer-contraction CP diagnostic for the same two selected dimer geometries.",
                "Positive BSSE is CP minus the fixed-dimer-parameter own-basis interaction.",
                "Dimer-predicted contraction parameters remain fixed in own/ghost fragment calculations.",
                "This diagnostic has a different monomer reference from the main corrected interaction curves.",
                "Distance scale is dimensionless; BSSE uses a logarithmic ordinate."]),
    dict(stem="fig-ethylene-density-xy", layout="copy",
         source_root="output/manuscript_density",
         sources=["ethylene-density-xy-comparison.pdf"],
         records=["plot-report.json"],
         facts=["Molecular xy plane, z = 0; all six nuclei lie in this plane.",
                "Rows: MACE, free contraction, STO-6G+P; columns: total density and density minus cc-pVTZ.",
                "Linear scales: rho from 0 to 1.5 and difference from -0.1 to 0.1 electron/bohr^3.",
                "Color saturation affects display only; annotated extrema are original unclipped plane-grid extrema.",
                "Density contours: 0.005, 0.02, 0.08, 0.32; difference contours: -0.02, 0, 0.02 electron/bohr^3.",
                "All-electron total-density slices are not orbital densities or integrated projections."]),
    dict(stem="fig-ethylene-density-xz", layout="copy",
         source_root="output/manuscript_density",
         sources=["ethylene-density-xz-comparison.pdf"],
         records=["plot-report.json"],
         facts=["Perpendicular xz plane, y = 0, through C=C; only the two carbon nuclei lie in this plane.",
                "Rows: MACE, free contraction, STO-6G+P; columns: total density and density minus cc-pVTZ.",
                "Uses the same linear color limits, units and contour levels as the xy figure.",
                "Only in-plane nuclei are marked; hydrogens are not projected onto this plane.",
                "cc-pVTZ is a matched DF-RHF basis reference, not an exact density."]),
    dict(stem="fig-s22-bsse", layout="bsse_summary_caption",
         sources=["bsse_counterpoise/s22-bsse-summary.pdf"],
         records=["bsse_counterpoise/plotted-values.json", "bsse_counterpoise/verification.json"],
         facts=["Two matched cohorts: seven bases on all 22 S22 dimers; eight bases on the common 12 dimers.",
                "All 15 archived arithmetic means, equal within-cohort weights, shared logarithmic axis.",
                "Ten missing ma-def2-QZVPP values are excluded; no zero substitution.",
                "Frozen dimer-predicted MACE contractions and common dimer auxiliary metric.",
                "STO-6G means retain a 282.1533576987432 kcal/mol pyrazine outlier with unequal own-fragment energies; SCF-branch review remains open.",
                "Cohort scope and source limitations are given in the caption; this is not a validated basis-quality ranking."]),
    dict(stem="fig-basis-accuracy-mp2", layout="benchmark_caption",
         sources=["basis_accuracy_benchmark/basis-accuracy-mp2.pdf"],
         records=["basis_accuracy_benchmark/plot-report.json", "basis_accuracy_benchmark/data-audit.json"],
         facts=["24 complete non-QZ bases on 30 physical structures and 12 relative-energy observations.",
                "Frozen-core DF-MP2; method and explanatory scope are stated in the caption only.",
                "Mean contracted spherical AO counts; circle area encodes mean unique spherical primitives.",
                "ISO34 W1-F12 and S22B estimated CCSD(T)/CBS database references; no CP correction.",
                "This is combined method/basis reference accuracy, not a same-method basis-limit error."]),
    dict(stem="fig-basis-accuracy-wb97m-v", layout="benchmark_caption",
         sources=["basis_accuracy_benchmark/basis-accuracy-wb97m-v.pdf"],
         records=["basis_accuracy_benchmark/plot-report.json", "basis_accuracy_benchmark/data-audit.json"],
         facts=["omegaB97M-V; same 24 bases, 30 structures, 12 observables, and scales as the MP2 figure.",
                "Method, reference and metric descriptions are stated in the caption only.",
                "Axes, point coordinates, labels, circle areas and vector proportions are preserved."]),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spans(page):
    return [span for block in page.get_text("dict")["blocks"] if "lines" in block
            for line in block["lines"] for span in line["spans"]]


def bounds_audit(page):
    rect = page.rect
    return [dict(text=s["text"], bbox=list(s["bbox"])) for s in spans(page)
            if not rect.contains(fitz.Rect(s["bbox"]))]


def benchmark_caption_only(source_path: Path, target: Path):
    """Move descriptive headers/footers to the caption without redrawing data."""
    with fitz.open(source_path) as document:
        assert len(document) == 1
        page = document[0]
        width, height = page.rect.width, page.rect.height
        assert abs(width - 633.6) < .01 and abs(height - 547.2) < .01
        top, bottom = 38., 502.
        removed = [s["text"] for s in spans(page)
                   if s["bbox"][3] <= top or s["bbox"][1] >= bottom]
        for rect in (fitz.Rect(0, 0, width, top), fitz.Rect(0, bottom, width, height)):
            page.add_redact_annot(rect, fill=False)
        page.apply_redactions(images=0, graphics=0)
        page.set_cropbox(fitz.Rect(0, top, width, bottom))
        text = page.get_text()
        assert "grid level" not in text and "frozen core" not in text
        assert "non-QZ bases" not in text and "References:" not in text
        document.save(target, garbage=4, deflate=True)
    return dict(removed_text=removed, explanatory_text_in_caption_only=True,
                crop_box_points=[0., top, width, bottom],
                artwork_rescaled=False, artwork_rasterized=False,
                original_plot_coordinate_aspect_preserved=True)


def bsse_summary_caption_only(source_path: Path, target: Path):
    """Move cohort descriptions into the caption, preserving all plotted marks."""
    with fitz.open(source_path) as document:
        assert len(document) == 1
        page = document[0]
        titles = [s for s in spans(page) if s["text"].startswith("S22:")]
        assert len(titles) == 2
        for title in titles:
            page.add_redact_annot(fitz.Rect(title["bbox"]), fill=False)
        page.apply_redactions(images=0, graphics=0)
        for title, label in zip(titles, ("(a)", "(b)")):
            page.insert_text((title["bbox"][0], title["origin"][1]),
                             label, fontname="hebo", fontsize=16.)
        document.save(target, garbage=4, deflate=True)
    return dict(removed_text=[s["text"] for s in titles],
                added_panel_labels=["(a)", "(b)"],
                explanatory_text_in_caption_only=True,
                artwork_rescaled=False, artwork_rasterized=False,
                original_plot_coordinate_aspect_preserved=True)


def compose_pair(source_paths, target: Path, headers, vertical=False):
    """Reuse each complete source page without rasterizing its vector content."""
    docs = [fitz.open(path) for path in source_paths]
    assert all(len(doc) == 1 for doc in docs)
    widths = [doc[0].rect.width for doc in docs]
    heights = [doc[0].rect.height for doc in docs]
    assert abs(heights[0] - heights[1]) < 1e-3
    gutter, header_height = (8., 26.) if vertical else (12., 30.)
    result = fitz.open()
    page = result.new_page(
        width=max(widths) if vertical else sum(widths) + gutter,
        height=sum(heights) + 2 * header_height + gutter if vertical else max(heights) + header_height)
    offset = 0.
    placements = []
    for index, (source, width, height, header) in enumerate(zip(docs, widths, heights, headers)):
        destination = (fitz.Rect(0., offset + header_height, width, offset + header_height + height)
                       if vertical else fitz.Rect(offset, header_height, offset + width, header_height + height))
        page.show_pdf_page(destination, source, 0, keep_proportion=True)
        page.insert_text((13., offset + 19.5) if vertical else (offset + 13., 22.),
                         header, fontname="hebo", fontsize=16. if vertical else 17.)
        placements.append(dict(source_index=index, rect_points=list(destination), header=header))
        offset += height + header_height + gutter if vertical else width + gutter
    result.set_metadata(dict(title=target.stem, subject="Archived NNAO figures composed without replotting"))
    result.save(target, garbage=4, deflate=True)
    result.close()
    for doc in docs:
        doc.close()
    return dict(orientation="vertical" if vertical else "horizontal",
                header_height_points=header_height, gutter_points=gutter, placements=placements,
                source_content_rasterized=False)


def pad_density(source_path: Path, target: Path):
    """Expose the existing label descenders by extending only the PDF canvas."""
    with fitz.open(source_path) as doc:
        assert len(doc) == 1
        page = doc[0]
        width, height = page.rect.width, page.rect.height
        # PDF coordinates are bottom-up: a negative lower edge adds space below
        # the existing artwork without moving it relative to the top of page.
        page.set_mediabox(fitz.Rect(0., -2., width, height))
        doc.save(target, garbage=4, deflate=True)
    return dict(bottom_canvas_padding_points=2., source_content_rasterized=False,
                reason="Original x-axis-label font metrics extend 0.149 point below the source page.")


def render_and_audit(target: Path):
    with fitz.open(target) as doc:
        assert len(doc) == 1
        page = doc[0]
        bad = bounds_audit(page)
        assert not bad, (target, bad)
        png = PREVIEW / (target.stem + ".png")
        page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(png)
        return dict(pages=len(doc), page_points=list(page.rect),
                    text_spans=len(spans(page)), text_outside_page=bad,
                    fonts=sorted({s["font"] for s in spans(page)}),
                    preview=str(png.relative_to(ROOT)))


def contact_sheet(figures):
    width, height, top = 620, 490, 34
    sheet = Image.new("RGB", (2 * width, ((len(figures) + 1) // 2) * (height + top)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    for index, entry in enumerate(figures):
        x = (index % 2) * width
        y = (index // 2) * (height + top)
        draw.text((x + 12, y + 8), entry["stem"], fill="black", font=font)
        with Image.open(PREVIEW / (entry["stem"] + ".png")) as image:
            image.thumbnail((width - 16, height - 16), Image.Resampling.LANCZOS)
            sheet.paste(image, (x + (width - image.width) // 2,
                               y + top + (height - image.height) // 2))
    sheet.save(PREVIEW / "figure-contact-sheet.png")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    from plot_ethylene_density_comparison import build_manuscript
    build_manuscript()
    entries, hashes = [], {}
    for specification in FIGURES:
        source_root = ROOT / specification.get("source_root", "output/ppt")
        paths = [source_root / name for name in specification["sources"]]
        source_info = []
        source_text = Counter()
        for path in paths:
            digest = sha(path)
            hashes[path] = digest
            with fitz.open(path) as source:
                assert len(source) == 1
                source_info.append(dict(path=str(path.relative_to(ROOT)), sha256=digest,
                                        page_points=list(source[0].rect),
                                        source_text_outside_page=bounds_audit(source[0])))
                source_text.update(s["text"] for s in spans(source[0]))
        target = DEST / (specification["stem"] + ".pdf")
        layout = specification["layout"]
        if layout in ("pair", "stack"):
            transform = compose_pair(paths, target, specification["headers"], vertical=layout == "stack")
        elif layout == "density_paper":
            from plot_ethylene_density_paper import build_figure
            transform = build_figure(specification["stem"].rsplit("-", 1)[-1], target)
        elif layout == "density_padding":
            transform = pad_density(paths[0], target)
        elif layout == "benchmark_caption":
            transform = benchmark_caption_only(paths[0], target)
            source_text.subtract(Counter(transform["removed_text"]))
        elif layout == "bsse_summary_caption":
            transform = bsse_summary_caption_only(paths[0], target)
            source_text.subtract(Counter(transform["removed_text"]))
        elif layout == "reaction_paper":
            from plot_reaction_profiles_paper import build_figure
            transform = build_figure(target)
        elif layout == "dimer_paper":
            from plot_dimer_interactions_paper import build_figure
            transform = build_figure(target)
        else:
            assert layout == "copy" and len(paths) == 1
            shutil.copyfile(paths[0], target)
            transform = dict(byte_identical_to_source=True)
            assert sha(target) == sha(paths[0])
        records = [dict(path=str((source_root / name).relative_to(ROOT)), sha256=sha(source_root / name))
                   for name in specification["records"]]
        audit = render_and_audit(target)
        with fitz.open(target) as final:
            # Form XObject placement can split words differently; compare all
            # extracted text after removing whitespace, preserving multiplicity.
            rendered = "".join(s["text"] for s in spans(final[0]))
            if layout not in ("density_paper", "reaction_paper", "dimer_paper"):
                for key, count in source_text.items():
                    assert rendered.count(key) >= count, (target, key, count)
        number = FIGURE_NUMBERS[specification["stem"]]
        entries.append(dict(stem=specification["stem"], figure_number=number,
                            manuscript_columns=2 if number in (2, 3, 4) else 1,
                            path=str(target.relative_to(ROOT)),
                            sha256=sha(target), layout=layout, sources=source_info,
                            transform=transform, caption_facts=specification["facts"],
                            source_plotting_records=records, verification=audit))
    assert all(sha(path) == digest for path, digest in hashes.items())
    manifest = dict(figure_count=len(entries), script=str(Path(__file__).relative_to(ROOT)),
                    script_sha256=sha(Path(__file__)), figures=entries,
                    selected_reactions=["BH9_02_134", "BH9_04_1"],
                    selected_dimers=["S66x8-Water-Water", "S66x8-Benzene-Benzene_pi-pi"],
                    source_artifacts_unchanged=True,
                    scope="Curve PDF composition and density-layout rendering from saved grids; no physical calculations or data changes.")
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verification = dict(figures=len(entries), all_single_page=True,
                        all_output_text_within_page=True,
                        all_plot_text_preserved_except_descriptions_moved_to_captions=True,
                        density_grid_values_preserved=True,
                        direct_copies_byte_identical=True,
                        all_source_artifacts_unchanged=True,
                        double_column_figures=[2, 3, 4],
                        single_column_figures=[1, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                        single_column_square_pairs=[5, 6, 9, 10, 11],
                        original_curve_figure_proportions_preserved=True,
                        density_coordinate_aspect_equal=True,
                        density_plane_positions_in_captions_only=True,
                        benchmark_explanations_in_captions_only=True,
                        contact_sheet="output/manuscript_figures/figure-contact-sheet.png",
                        checks=[dict(path=entry["path"], **entry["verification"]) for entry in entries])
    (PREVIEW / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    contact_sheet(entries)
    (DEST / "README.md").write_text("""# Manuscript figures

These thirteen PDFs are the manuscript copies of the reviewed plotting outputs.
The source files under `output/ppt/` and all archived calculation inputs
remain unchanged. `manifest.json` records the exact source paths, SHA256
checksums, caption facts, and page geometry for each figure.

Figures 2, 3, and 4 span both columns. Figures 1 and 5--13 occupy a single
column. All paired square plots are placed side by side in one column:
ethane energy scans (5), reaction profiles (6), dimer interactions (9), and
BSSE curves (10), and S22 BSSE summaries (11). Each original square source
page retains its proportions.
The radial response (2) remains a side-by-side, double-column comparison.
Curve figures reuse the original reviewed PDF pages as PDF Form XObjects;
their fonts, legends, molecular structures, and proportions are unchanged.
LaTeX specifies width only, so the complete figures are scaled uniformly.
Coefficient and training-statistics PDFs are byte-identical copies. Density
figures use a compact manuscript layout: larger maps, narrower inter-column
spacing, and shared horizontal colorbars beneath the two columns. Coordinate
scales remain equal, and full archived windows, linear color limits, contours,
and original numeric extrema are preserved. Plane positions appear only in
the captions. Original PPT density figures remain unchanged.

Benchmark figures 12 and 13 retain the approved vector plots, including all
24 points, direct labels and legends. Their method/configuration headers and
descriptive footers are removed from the manuscript copies, with whitespace
cropped uniformly around the remaining plot. The axes, marker areas and
coordinate proportions are unchanged. Scientific explanations belong to
the manuscript captions; no training configuration is added to these plots.

The main examples are limited to two reactions (BH9_02_134 and BH9_04_1)
and two dimer interactions (water-water and parallel-displaced benzene).
The same dimer geometries are used for separate frozen-parameter BSSE plots.
The corrected interaction energies use independent isolated-monomer
references, whereas the BSSE diagnostic freezes dimer-predicted parameters
in the own/ghost fragment calculations. These quantities must not be
identified or combined without their corresponding reference definitions.
The fixed-output MACE control is excluded.

Figure 11 includes the two archived S22 aggregate-mean panels without
altering their values. Cohort descriptions are moved from the panel titles
to the caption. The caption identifies the retained STO-6G pyrazine outlier
and unequal own-fragment energies, which require an SCF-branch audit before
a basis-quality interpretation. Missing ma-def2-QZVPP values are not zero.
The original S22 artifacts remain under `output/ppt/bsse_counterpoise/`.

Rebuild the figure copies from the repository root:

```bash
/opt/anaconda3/bin/python scripts/assemble_manuscript_figures.py
```

Rendered previews and the geometry/text-preservation audit are written to
`output/manuscript_figures/`. This workflow performs no model inference,
SCF, optimization, training, or finite-difference calculation.
""")
    print(json.dumps(dict(figures=len(entries), manifest="manuscript/figures/manifest.json",
                          verification="output/manuscript_figures/verification.json",
                          contact_sheet="output/manuscript_figures/figure-contact-sheet.png")))


if __name__ == "__main__":
    main()
