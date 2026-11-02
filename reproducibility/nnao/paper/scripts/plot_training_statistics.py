"""Count the archived training structures and NNAO-SZP-L spherical AOs.

Use --ppt for the presentation profile. This reads structures and shell
inventories only; no model inference or electronic calculations are needed.
"""
from __future__ import annotations

from collections import Counter
import csv
from html import escape
import json
from pathlib import Path
import zipfile

from plot_case08_reactions import ROOT, OUTPUT_ROOT, configure_style, sha, style_metadata
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter


SOURCE = ROOT / "data/training/nnao500-20261021"
OUT = OUTPUT_ROOT / "training_statistics"
MANUSCRIPT = ROOT / "manuscript/main.tex"
SYMBOLS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr".split()
ATOMIC_NUMBER = {symbol: z for z, symbol in enumerate(SYMBOLS, 1)}
# Element group, atomic numbers, fixed-core spherical AOs, active spherical AOs.
SHELL_GROUPS = [
    ("H, He", range(1, 3), 0, 4),
    ("Li, Be", range(3, 5), 1, 4),
    ("B--Ne", range(5, 11), 1, 9),
    ("Na, Mg", range(11, 13), 5, 4),
    ("Al--Ar", range(13, 19), 5, 9),
    ("K, Ca", range(19, 21), 9, 4),
    ("Ga--Kr", range(31, 37), 14, 9),
]
AO_PER_ATOM = {z: core + active for _, numbers, core, active in SHELL_GROUPS for z in numbers}
BLUE = "#0057B8"


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def xyz_frames(path):
    lines = path.read_text().splitlines()
    position = 0
    while position < len(lines):
        if not lines[position].strip():
            position += 1
            continue
        count = int(lines[position])
        metadata = json.loads(lines[position + 1])
        atoms = [line.split() for line in lines[position + 2:position + 2 + count]]
        assert len(atoms) == count and all(len(atom) == 4 for atom in atoms)
        yield metadata, [atom[0] for atom in atoms], [[float(x) for x in atom[1:]] for atom in atoms]
        position += count + 2


def check_manuscript_mapping():
    table = MANUSCRIPT.read_text().split(r"\label{tab:element-cores}", 1)[1].split(r"\end{tabular}", 1)[0]
    counts = {}
    for line in table.splitlines():
        columns = [x.strip() for x in line.split("&")]
        if len(columns) == 7 and columns[4].isdigit():
            counts[columns[0]] = int(columns[4])
    assert counts == {name: core + active for name, _, core, active in SHELL_GROUPS}


def describe(values):
    return dict(min=int(min(values)), max=int(max(values)), mean=float(np.mean(values)),
                median=float(np.median(values)), total=int(sum(values)),
                quartiles=np.percentile(values, [25, 75]).tolist())


def save_figure(figure, stem):
    for extension in ("png", "pdf", "svg"):
        figure.savefig(OUT / (stem + "." + extension), dpi=320)
    plt.close(figure)


def make_plots(elements, structures):
    figure = plt.figure(figsize=(9., 6.3))
    ax = figure.add_axes([.12, .13, .84, .81])
    x = np.arange(len(elements))
    ax.bar(x, [row["atom_count"] for row in elements], color=BLUE, width=.74, zorder=3)
    ax.set_xticks(x, [row["element"] for row in elements], fontsize=12.5)
    ax.set_xlabel("Element")
    ax.set_ylabel("Number of atoms (log scale)")
    ax.set_yscale("log")
    ax.set_ylim(.7, 8000)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2., 5.), numticks=100))
    ax.set_xlim(-.8, len(elements) - .2)
    ax.tick_params(which="both", top=True, right=True)
    ax.tick_params(which="minor", length=2.5, width=.8)
    ax.text(.98, .96, "5,402 atoms", transform=ax.transAxes, ha="right", va="top", fontsize=12.5)
    save_figure(figure, "element-distribution")

    heavy_counts = Counter(row["heavy_atoms"] for row in structures)
    heavy = [dict(heavy_atoms=x, structures=heavy_counts[x]) for x in range(max(heavy_counts) + 1)]
    write_csv("heavy-atom-distribution.csv", heavy)
    figure = plt.figure(figsize=(6.3, 6.3))
    ax = figure.add_axes([.15, .13, .81, .81])
    ax.bar([r["heavy_atoms"] for r in heavy], [r["structures"] for r in heavy],
           color=BLUE, width=.86)
    ax.set_xlabel("Number of heavy atoms")
    ax.set_ylabel("Number of structures")
    ax.set_xlim(-.75, max(heavy_counts) + .75)
    ax.set_xticks(range(0, max(heavy_counts) + 1, 2))
    ax.set_ylim(0, max(heavy_counts.values()) * 1.2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.tick_params(top=True, right=True)
    ax.text(.97, .96, "500 structures", transform=ax.transAxes, ha="right", va="top", fontsize=12.5)
    save_figure(figure, "heavy-atom-distribution")

    nao = [row["nao_szp_l"] for row in structures]
    edges = np.arange(0, max(nao) // 10 * 10 + 11, 10)
    counts, _ = np.histogram(nao, bins=edges)
    assert int(counts.sum()) == len(structures)
    bins = [dict(left_inclusive=int(left), right_exclusive=int(right), structures=int(count))
            for left, right, count in zip(edges[:-1], edges[1:], counts)]
    write_csv("ao-histogram-bins.csv", bins)
    exact = Counter(nao)
    write_csv("ao-count-distribution.csv", [dict(nao_szp_l=value, structures=exact[value]) for value in sorted(exact)])
    figure = plt.figure(figsize=(6.3, 6.3))
    ax = figure.add_axes([.15, .13, .81, .81])
    ax.bar(edges[:-1], counts, align="edge", width=9.6, color=BLUE)
    ax.set_xlabel("Number of AOs (NNAO-SZP-L)")
    ax.set_ylabel("Number of structures")
    ax.set_xlim(0, edges[-1])
    ax.set_xticks(range(0, int(edges[-1]) + 1, 50))
    ax.set_ylim(0, max(counts) * 1.25)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.tick_params(top=True, right=True)
    ax.text(.97, .96, "500 structures", transform=ax.transAxes, ha="right", va="top", fontsize=12.5)
    save_figure(figure, "ao-count-distribution")
    return heavy, bins


def write_gallery():
    items = [("element-distribution", "Element distribution"),
             ("heavy-atom-distribution", "Heavy-atom count distribution"),
             ("ao-count-distribution", "NNAO-SZP-L AO count distribution")]
    cards = []
    for stem, title in items:
        cards.append('<section><h2>' + escape(title) + '</h2><a href="' + stem + '.png">'
            '<img src="' + stem + '.png" alt="' + escape(title) + '"></a><p>' +
            ' | '.join('<a href="' + stem + '.' + ext + '">' + ext.upper() + '</a>'
                       for ext in ("png", "pdf", "svg")) + '</p></section>')
    (OUT / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Training-set statistics</title>'
        '<style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:16px/1.5 Arial;color:#171717}'
        'img{max-width:100%;height:auto}section{margin:35px 0}a{color:#0057b8}</style>'
        '<h1>Training-set statistics</h1><p>500 structures; 5,402 atoms; 25 observed elements. '
        '400 GMTKN55-v2 structures and 100 QH9Stable structures, each counted once.</p>'
        '<p>Elements are counted by atomic occurrences. Heavy atoms have Z &gt; 1. '
        'AO counts include fixed cores and active valence/polarization shells in the real spherical basis. '
        'The AO histogram has 10-AO bins; exact integer frequencies are supplied separately.</p>'
        '<p><a href="training-statistics-pngs.zip">All PNGs</a> | '
        '<a href="training-statistics.pdf">Combined PDF</a> | '
        '<a href="element-distribution.csv">Element counts and structure coverage</a> | '
        '<a href="per-structure-statistics.csv">All 500 structure counts</a></p>' + ''.join(cards) + '</html>')
    book = PdfWriter()
    with zipfile.ZipFile(OUT / "training-statistics-pngs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for stem, _ in items:
            archive.write(OUT / (stem + ".png"), stem + ".png")
            book.append(PdfReader(OUT / (stem + ".pdf")))
    book.write(OUT / "training-statistics.pdf")
    overview = Image.new("RGB", (2700, 3000), "white")
    for number, (stem, _) in enumerate(items):
        with Image.open(OUT / (stem + ".png")) as image:
            image.thumbnail((1350, 1500), Image.Resampling.LANCZOS)
            x = (2700 - image.width) // 2 if number == 0 else (number - 1) * 1350 + (1350 - image.width) // 2
            y = (1500 - image.height) // 2 if number == 0 else 1500 + (1500 - image.height) // 2
            overview.paste(image, (x, y))
    overview.save(OUT / "overview.png")


def main():
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    check_manuscript_mapping()
    references = json.loads((SOURCE / "references.json").read_text())
    source_rows = read_csv(SOURCE / "structure-index.csv")
    frames = list(xyz_frames(SOURCE / "all-structures.xyz"))
    assert len(source_rows) == len(frames) == len(references["structures"]) == 500
    elements, presence, structures = Counter(), Counter(), []
    for i, (index, frame, reference) in enumerate(zip(source_rows, frames, references["structures"])):
        metadata, symbols, coords = frame
        assert int(index["training_index_zero_based"]) == metadata["training_index_zero_based"] == i
        assert index["structure_id"] == metadata["structure_id"] == reference["structure_id"]
        assert symbols == reference["symbols"] and len(symbols) == int(index["natoms"])
        assert np.array_equal(coords, reference["coords_angstrom"])
        assert sha(SOURCE / index["xyz_file"]) == index["xyz_sha256"]
        numbers = [ATOMIC_NUMBER[symbol] for symbol in symbols]
        assert numbers == reference["atomic_numbers"]
        composition = Counter(symbols)
        elements.update(composition)
        presence.update(composition.keys())
        structures.append(dict(training_index_zero_based=i, structure_id=index["structure_id"],
            source_dataset=index["source_dataset"], formula=index["formula"], natoms=len(symbols),
            hydrogen_atoms=composition["H"], heavy_atoms=sum(z > 1 for z in numbers),
            nao_szp_l=sum(AO_PER_ATOM[z] for z in numbers),
            element_counts=json.dumps(dict(sorted(composition.items(), key=lambda x: ATOMIC_NUMBER[x[0]]))),
            xyz_file=index["xyz_file"]))
    coverage = {row["element"]: row for row in read_csv(SOURCE / "element-coverage.csv")}
    assert len(elements) == 25 and sum(elements.values()) == 5402
    assert all(elements[symbol] == int(coverage[symbol]["atoms"])
               and presence[symbol] == int(coverage[symbol]["structures"]) for symbol in elements)
    assert sum(int(row["atomic_number"]) > 1 for row in read_csv(SOURCE / "atoms.csv")) == sum(r["heavy_atoms"] for r in structures)
    element_rows = [dict(atomic_number=ATOMIC_NUMBER[symbol], element=symbol, atom_count=elements[symbol],
        atom_percent=100 * elements[symbol] / sum(elements.values()), structure_count=presence[symbol],
        structure_percent=100 * presence[symbol] / len(structures),
        nao_per_atom_szp_l=AO_PER_ATOM[ATOMIC_NUMBER[symbol]]) for symbol in sorted(elements, key=ATOMIC_NUMBER.get)]
    mapping = [dict(atomic_number=z, element=SYMBOLS[z - 1], fixed_core_ao=core,
        active_valence_and_polarization_ao=active, total_spherical_ao=core + active)
        for _, numbers, core, active in SHELL_GROUPS for z in numbers]
    write_csv("element-distribution.csv", element_rows)
    write_csv("basis-ao-counts.csv", mapping)
    write_csv("per-structure-statistics.csv", structures)
    make_plots(element_rows, structures)
    write_gallery()
    report = dict(structures=500, source_counts=dict(Counter(r["source_dataset"] for r in structures)),
        total_atoms=sum(elements.values()), observed_elements=len(elements),
        atom_counts=describe([r["natoms"] for r in structures]),
        heavy_atom_counts=describe([r["heavy_atoms"] for r in structures]),
        nao_szp_l_counts=describe([r["nao_szp_l"] for r in structures]),
        definitions=dict(element_distribution="Atomic occurrences across all 5,402 atoms; structure presence also exported.",
            heavy_atom="Atomic number Z > 1, including He and other noble gases.",
            nao="Contracted real spherical AOs per structure, including fixed core, valence and polarization; d=5.",
            basis="NNAO-SZP-L", weighting="Each of the 500 ordered training entries is counted once.",
            ao_histogram_bin_width=10, ao_histogram_intervals="[left, right); no data on the final right edge."),
        elements=element_rows, style=style_metadata(),
        ao_mapping_source="manuscript/main.tex, Table element-cores (SZP column), checked by this script.",
        source_sha256={str(p): sha(p) for p in [SOURCE / "references.json", SOURCE / "structure-index.csv",
            SOURCE / "all-structures.xyz", SOURCE / "atoms.csv", SOURCE / "element-coverage.csv", MANUSCRIPT,
            ROOT / "scripts/plot_case08_reactions.py", Path(__file__)]},
        individual_xyz_hashes_verified=500,
        scope="Descriptive structure and basis-dimension statistics. No model inference, SCF or training. "
              "All 500 entries, including one isolated He reference, are retained. "
              "These are training structures, not a held-out validation set.")
    (OUT / "statistics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_sha256", "elements", "scope")}, indent=2))


if __name__ == "__main__":
    main()
