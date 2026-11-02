#!/usr/bin/env python3
"""Compose the case 08 gallery from existing xyzrender images.

Run with /opt/anaconda3/bin/python scripts/build_case08_gallery.py after
output/case08_xyzrender/manifest.json and all of its renders are complete.
This script only reads the manifest, source geometries, and rendered images;
it does not infer bonds, transform geometries, or run electronic calculations.
"""

from collections import OrderedDict
from functools import lru_cache
from html import escape
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote

from PIL import Image, ImageChops, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "output/case08_xyzrender"
MANIFEST = GALLERY / "manifest.json"
PDF = ROOT / "output/pdf/case08-structures.pdf"
DATASETS = ("S66x8", "HB300SPXx10", "BH9")
EXPECTED_GROUPS = {"S66x8": 6, "HB300SPXx10": 6, "BH9": 9}
FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
BOLD_FONT = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
INK = "#18212b"
MUTED = "#4d5966"
BORDER = "#dce1e6"
WIDTH, HEIGHT = 2400, 1697
MARGIN, GAP = 48, 22

DIMER_NOTE = (
    "Dimer renders share a camera and physical pixel scale within each group. "
    "Distance scale factors are dimensionless."
)
BH9_NOTE = (
    "Reactant and product fragments are separate source XYZ structures. "
    "Species are independently oriented, with a common scale within each reaction."
)
VISUAL_NOTE = "Inferred bonds are visual guides only. No combined fragment geometry is constructed."
GROUP_CROP_NOTE = "One common foreground crop is used per group; original PNG/SVG files are unchanged."
OVERVIEW_NOTE = "Representatives are framed individually; displayed sizes are not comparable across groups."


@lru_cache(maxsize=None)
def font(size, bold=False):
    return ImageFont.truetype(str(BOLD_FONT if bold else FONT), size)


def slug(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-") or "group"


def link(path):
    relative = Path(os.path.relpath(path, GALLERY)).as_posix()
    return quote(relative, safe="/._-")


def input_path(value):
    path = Path(value)
    return path if path.is_absolute() else GALLERY / path


def phase(record):
    role = str(record["role"]).strip().upper()
    if role in {"R", "R1", "R2"}:
        return 0
    if role == "TS":
        return 1
    if role in {"P", "P1", "P2"}:
        return 2
    raise ValueError(f"Unknown BH9 role for {record['id']}: {record['role']!r}")


def role_label(record):
    role = str(record["role"])
    if record["dataset"] == "BH9":
        return {
            "R": "Reactant", "R1": "Reactant fragment 1", "R2": "Reactant fragment 2",
            "TS": "Transition state", "P": "Product", "P1": "Product fragment 1",
            "P2": "Product fragment 2",
        }[role.strip().upper()]
    if record.get("scale") is not None:
        return f"Scale = {float(record['scale']):.2f} (dimensionless)"
    return role[:1].upper() + role[1:]


def metadata(record):
    return f"Atoms: {record['natoms']} | Charge: {record['charge']} | 2S: {record['spin_2S']}"


def load_inputs():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = manifest["records"]
    if len(records) != 166:
        raise ValueError(f"Expected all 166 structures; manifest contains {len(records)}")
    groups = OrderedDict()
    ids = set()
    image_paths = {"png": set(), "svg": set()}
    required = {"id", "dataset", "group_id", "group_title", "role", "natoms",
                "charge", "spin_2S", "png", "svg", "source_xyz", "representative"}
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Manifest record lacks fields: {sorted(missing)}")
        if record["id"] in ids:
            raise ValueError(f"Duplicate record ID: {record['id']}")
        ids.add(record["id"])
        if record["dataset"] not in DATASETS:
            raise ValueError(f"Unknown dataset: {record['dataset']}")
        for key in ("png", "svg", "source_xyz"):
            path = input_path(record[key])
            if not path.is_file():
                raise FileNotFoundError(f"Missing {key} for {record['id']}: {path}")
            if key in image_paths:
                resolved = path.resolve()
                if resolved in image_paths[key]:
                    raise ValueError(f"Duplicate {key} path for {record['id']}: {path}")
                image_paths[key].add(resolved)
        with Image.open(input_path(record["png"])) as image:
            image.verify()
        key = (record["dataset"], str(record["group_id"]))
        groups.setdefault(key, []).append(record)
    counts = {dataset: sum(key[0] == dataset for key in groups) for dataset in DATASETS}
    if counts != EXPECTED_GROUPS:
        raise ValueError(f"Expected 21 groups {EXPECTED_GROUPS}; received {counts}")
    for (dataset, group_id), members in groups.items():
        if len({r["group_title"] for r in members}) != 1:
            raise ValueError(f"Inconsistent title for {dataset}/{group_id}")
        representatives = [r for r in members if r["representative"]]
        if len(representatives) != 1:
            raise ValueError(f"Expected one representative for {dataset}/{group_id}")
        if dataset == "BH9":
            members.sort(key=phase)
            if {phase(r) for r in members} != {0, 1, 2}:
                raise ValueError(f"Missing reactant, TS, or product for {group_id}")
            if phase(representatives[0]) != 1:
                raise ValueError(f"BH9 representative must be the TS: {group_id}")
        elif not math.isclose(float(representatives[0]["scale"]), 1.0):
            raise ValueError(f"Dimer representative must have scale 1.00: {group_id}")
    ordered = OrderedDict((key, groups[key]) for dataset in DATASETS
                          for key in groups if key[0] == dataset)
    return manifest, records, ordered


def lines_for(draw, text, face, width):
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if line and draw.textlength(candidate, font=face) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def fitted_lines(draw, text, width, max_lines=2, size=34):
    for current in range(size, 21, -1):
        face = font(current, True)
        lines = lines_for(draw, text, face, width)
        if len(lines) <= max_lines and all(draw.textlength(s, font=face) <= width for s in lines):
            return lines, face, current
    raise ValueError(f"Label is too long for the contact sheet: {text}")


def white_render(path):
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(white, rgba).convert("RGB")


@lru_cache(maxsize=None)
def image_size(path):
    with Image.open(path) as image:
        return image.size


@lru_cache(maxsize=None)
def foreground_box(path):
    image = white_render(path)
    difference = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
    red, green, blue = difference.split()
    mask = ImageChops.lighter(red, ImageChops.lighter(green, blue)).point(lambda value: 255 if value > 8 else 0)
    box = mask.getbbox()
    if box is None:
        raise ValueError(f"No visible foreground in {path}")
    return (max(0, box[0] - 20), max(0, box[1] - 20),
            min(image.width, box[2] + 20), min(image.height, box[3] + 20))


def group_crop(members):
    paths = [input_path(record["png"]) for record in members]
    if len({image_size(path) for path in paths}) != 1:
        raise ValueError(f"Common group crop requires equal original canvas sizes: {members[0]['group_id']}")
    boxes = [foreground_box(path) for path in paths]
    return (min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes))


def common_image_scale(crop_box, width, height):
    return min((width - 28) / (crop_box[2] - crop_box[0]),
               (height - 160) / (crop_box[3] - crop_box[1]))


def write_thumbnails(groups, crop_boxes):
    directory = GALLERY / "thumbnails"
    directory.mkdir(parents=True, exist_ok=True)
    thumbnails = {}
    for key, members in groups.items():
        for record in members:
            destination = directory / f"{len(thumbnails) + 1:03d}-{slug(record['id'])}.png"
            white_render(input_path(record["png"])).crop(crop_boxes[key]).save(destination)
            thumbnails[record["id"]] = destination
    return thumbnails


def draw_card(sheet, record, box, title=None, representative=False, image_scale=None, crop_box=None):
    draw = ImageDraw.Draw(sheet)
    x0, y0, x1, y1 = map(int, box)
    draw.rectangle((x0, y0, x1, y1), outline=BORDER, width=2)
    text_width = x1 - x0 - 32
    heading = title if title is not None else role_label(record)
    lines, face, size = fitted_lines(draw, heading, text_width)
    y = y0 + 13
    for line in lines:
        draw.text((x0 + 16, y), line, font=face, fill=INK)
        y += size + 6
    if representative:
        draw.text((x0 + 16, y + 2), role_label(record), font=font(26), fill=MUTED)
        y += 35
    draw.text((x0 + 16, y + 2), metadata(record), font=font(26), fill=MUTED)
    y += 40
    available = (int(x1 - x0 - 28), int(y1 - y - 16))
    if min(available) <= 0:
        raise ValueError(f"No image space for {record['id']}")
    image = white_render(input_path(record["png"]))
    if crop_box is not None:
        image = image.crop(crop_box)
    if image_scale is None:
        image_scale = min(available[0] / image.width, available[1] / image.height)
    size = (max(1, int(image.width * image_scale)), max(1, int(image.height * image_scale)))
    if size[0] > available[0] or size[1] > available[1]:
        raise ValueError(f"Common group scale does not fit {record['id']}")
    image = image.resize(size, Image.Resampling.LANCZOS)
    sheet.paste(image, (int(x0 + (x1 - x0 - image.width) / 2),
                        int(y + (y1 - y - image.height) / 2)))


def sheet_base(title, subtitle, note, height=HEIGHT):
    sheet = Image.new("RGB", (WIDTH, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((MARGIN, 28), title, font=font(54, True), fill=INK)
    draw.text((MARGIN, 93), subtitle, font=font(28), fill=MUTED)
    footer_lines = lines_for(draw, note, font(25), WIDTH - 2 * MARGIN)
    footer_top = height - 30 - 33 * len(footer_lines)
    for index, line in enumerate(footer_lines):
        draw.text((MARGIN, footer_top + 33 * index), line, font=font(25), fill=MUTED)
    return sheet, 150, footer_top - 24


def overview(groups, destination, dataset=None):
    selected = [(key, rows) for key, rows in groups.items() if dataset is None or key[0] == dataset]
    representatives = [(key, next(r for r in rows if r["representative"])) for key, rows in selected]
    title = f"{dataset}: representative structures" if dataset else "Case 08: 21 structure groups"
    note = OVERVIEW_NOTE + " "
    note += ("BH9 panels show individual transition states. " if dataset == "BH9" else
             "Dimer scale factors are dimensionless. ")
    if dataset is None:
        note += "BH9 panels show individual transition states. "
    note += VISUAL_NOTE
    columns = 3
    rows = math.ceil(len(representatives) / columns)
    height = HEIGHT if dataset else 150 + rows * 510 + (rows - 1) * GAP + 140
    sheet, top, bottom = sheet_base(title, "One representative per group, with individual foreground framing.", note, height)
    cell_width = (WIDTH - 2 * MARGIN - (columns - 1) * GAP) / columns
    cell_height = (bottom - top - (rows - 1) * GAP) / rows
    for index, ((name, group_id), record) in enumerate(representatives):
        row, column = divmod(index, columns)
        x, y = MARGIN + column * (cell_width + GAP), top + row * (cell_height + GAP)
        heading = f"{group_id}: {record['group_title']}"
        if dataset is None:
            heading = f"{name} | {heading}"
        draw_card(sheet, record, (x, y, x + cell_width, y + cell_height), heading, True,
                  crop_box=foreground_box(input_path(record["png"])))
    sheet.save(destination, dpi=(220, 220))


def group_sheet(key, members, destination, crop_box):
    dataset, group_id = key
    note = (BH9_NOTE if dataset == "BH9" else DIMER_NOTE) + " " + GROUP_CROP_NOTE + " " + VISUAL_NOTE
    title = f"{dataset} | {group_id}"
    sheet, top, bottom = sheet_base(title, members[0]["group_title"], note)
    if dataset == "BH9":
        draw = ImageDraw.Draw(sheet)
        width = (WIDTH - 2 * MARGIN - 2 * GAP) / 3
        rows = max(sum(phase(r) == column for r in members) for column in range(3))
        card_top = top + 48
        height = (bottom - card_top - (rows - 1) * GAP) / rows
        image_scale = common_image_scale(crop_box, width, height)
        for column, heading in enumerate(("Reactants", "Transition state", "Products")):
            x = MARGIN + column * (width + GAP)
            draw.text((x + 10, top), heading, font=font(32, True), fill=INK)
            species = [r for r in members if phase(r) == column]
            for row, record in enumerate(species):
                y = card_top + (row + (rows - len(species)) / 2) * (height + GAP)
                draw_card(sheet, record, (x, y, x + width, y + height),
                          image_scale=image_scale, crop_box=crop_box)
    else:
        columns = 4
        rows = math.ceil(len(members) / columns)
        width = (WIDTH - 2 * MARGIN - (columns - 1) * GAP) / columns
        height = (bottom - top - (rows - 1) * GAP) / rows
        image_scale = common_image_scale(crop_box, width, height)
        for index, record in enumerate(members):
            row, column = divmod(index, columns)
            x, y = MARGIN + column * (width + GAP), top + row * (height + GAP)
            draw_card(sheet, record, (x, y, x + width, y + height),
                      image_scale=image_scale, crop_box=crop_box)
    sheet.save(destination, dpi=(220, 220))


def write_html(groups, sheets, thumbnails):
    navigation, sections = [], []
    for key, members in groups.items():
        dataset, group_id = key
        anchor = f"{slug(dataset)}-{slug(group_id)}"
        title = f"{dataset} | {group_id}: {members[0]['group_title']}"
        navigation.append(f'<a href="#{anchor}">{escape(title)}</a>')
        cards = []
        max_width = max(image_size(thumbnails[r["id"]])[0] for r in members)
        max_height = max(image_size(thumbnails[r["id"]])[1] for r in members)
        for record in members:
            alt = f"{title}; {role_label(record)}; {metadata(record)}"
            links = " | ".join(f'<a href="{escape(link(input_path(record[k])))}">{name}</a>'
                               for k, name in (("png", "PNG"), ("svg", "SVG"), ("source_xyz", "Source XYZ")))
            width = 100 * image_size(thumbnails[record["id"]])[0] / max_width
            cards.append(f'''<figure><a class="imageframe" style="aspect-ratio:{max_width}/{max_height}"
                href="{escape(link(input_path(record['png'])))}"><img style="width:{width:.6f}%;height:auto"
                src="{escape(link(thumbnails[record['id']]))}" alt="{escape(alt)}" loading="lazy"></a>
                <figcaption><strong>{escape(role_label(record))}</strong>
                <span>{escape(metadata(record))}</span><small>{escape(str(record['id']))}</small>
                <div class="links">{links}</div></figcaption></figure>''')
        if dataset == "BH9":
            content = "".join(f'<div class="phase"><h3>{heading}</h3>'
                              + "".join(card for record, card in zip(members, cards) if phase(record) == column)
                              + "</div>" for column, heading in enumerate(("Reactants", "Transition state", "Products")))
        else:
            content = "".join(cards)
        note = (BH9_NOTE if dataset == "BH9" else DIMER_NOTE) + " " + GROUP_CROP_NOTE
        sections.append(f'''<section id="{anchor}"><h2>{escape(title)}</h2><p>{escape(note)}</p>
            <p><a href="{escape(link(sheets[key]))}">Group contact sheet</a> | <a href="#groups">Group navigation</a></p>
            <div class="grid">{content}</div></section>''')
    dataset_links = " | ".join(f'<a href="{dataset}-overview.png">{dataset} overview</a>' for dataset in DATASETS)
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Case 08 molecular structures</title><style>
*{{box-sizing:border-box}} body{{margin:0;background:white;color:#18212b;font:16px/1.5 Arial,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}} h1{{font-size:32px;margin-bottom:8px}} h2{{font-size:23px}}
a{{color:#175b9a}} p{{max-width:1100px}} .nav{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px 18px}}
.nav a{{padding:7px 10px;border:1px solid #dce1e6;text-decoration:none}} section{{padding-top:18px;margin-top:30px;border-top:2px solid #dce1e6}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}} figure{{margin:0;border:1px solid #dce1e6;background:white}}
.imageframe{{display:flex;width:100%;align-items:center;justify-content:center;background:white}}
figure img{{display:block;max-width:100%;background:white}} .phase figure{{margin-bottom:18px}}
figcaption{{padding:12px 15px}} figcaption strong,figcaption span,figcaption small{{display:block}} small{{overflow-wrap:anywhere;color:#4d5966}}
.links{{margin-top:8px}} .overview{{width:100%;height:auto}} summary{{cursor:pointer;font-weight:bold}}
@media(max-width:1000px){{.grid,.nav{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:600px){{main{{padding:16px}}.grid,.nav{{grid-template-columns:1fr}}}}
</style></head><body><main><header><h1>Case 08 molecular structures</h1>
<p>166 existing xyzrender images in 21 groups: six S66x8 dimers, six HB300SPXx10 dimers, and nine BH9 reactions.
Charge and 2S labels follow the source manifest. This gallery contains structures, not new calculations.</p>
<p>{escape(DIMER_NOTE)} {escape(BH9_NOTE)} {escape(VISUAL_NOTE)}</p>
<p>{escape(GROUP_CROP_NOTE)} {escape(OVERVIEW_NOTE)}</p>
<p><a href="{escape(link(PDF))}">24-page PDF</a> | <a href="overview.png">All 21 representatives</a> |
{dataset_links} | <a href="manifest.json">Render manifest</a> | <a href="gallery-report.json">Gallery report</a></p></header>
<details><summary>All group representatives</summary><img class="overview" src="overview.png" alt="Representative structures for all 21 groups"></details>
<h2 id="groups">Group navigation</h2><nav class="nav" aria-label="Structure groups">{''.join(navigation)}</nav>
{''.join(sections)}<footer><p>Source XYZ, PNG, and SVG links point to unchanged originals. Thumbnails use common group crops;
overview representatives are individually framed from the original renders.
No reactant or product fragments have been combined into a new geometry.</p></footer></main></body></html>'''
    (GALLERY / "index.html").write_text(html, encoding="utf-8")


def write_pdf(dataset_sheets, group_sheets):
    pdfmetrics.registerFont(TTFont("GalleryArial", str(FONT)))
    PDF.parent.mkdir(parents=True, exist_ok=True)
    document = canvas.Canvas(str(PDF), pagesize=landscape(A4))
    document.setTitle("Case 08 molecular structures")
    document.setAuthor("NNAO paper structure gallery")
    pages = [(f"{dataset}: representative structures", dataset_sheets[dataset]) for dataset in DATASETS]
    pages.extend((f"{key[0]} | {key[1]}", path) for key, path in group_sheets.items())
    page_width, page_height = landscape(A4)
    for number, (title, path) in enumerate(pages, 1):
        document.bookmarkPage(f"page-{number}")
        document.addOutlineEntry(title, f"page-{number}", level=0)
        with Image.open(path) as image:
            ratio = min((page_width - 28) / image.width, (page_height - 42) / image.height)
            width, height = image.width * ratio, image.height * ratio
            document.drawImage(ImageReader(image), (page_width - width) / 2,
                               29 + (page_height - 42 - height) / 2, width, height)
        document.setFont("GalleryArial", 8)
        document.setFillColorRGB(0.30, 0.35, 0.40)
        document.drawString(20, 13, "Case 08 | Structures from unchanged source XYZ files")
        document.drawRightString(page_width - 20, 13, f"{number} / {len(pages)}")
        document.showPage()
    document.save()
    return len(pages)


def main():
    if not FONT.is_file() or not BOLD_FONT.is_file():
        raise FileNotFoundError("Arial regular and bold fonts are required in the macOS Supplemental font directory")
    manifest, records, groups = load_inputs()
    crop_boxes = {key: group_crop(members) for key, members in groups.items()}
    thumbnails = write_thumbnails(groups, crop_boxes)
    contacts = GALLERY / "contact-sheets"
    contacts.mkdir(parents=True, exist_ok=True)
    dataset_sheets = {}
    for dataset in DATASETS:
        destination = GALLERY / f"{dataset}-overview.png"
        overview(groups, destination, dataset)
        dataset_sheets[dataset] = destination
    overview(groups, GALLERY / "overview.png")
    group_sheets = OrderedDict()
    for key, members in groups.items():
        destination = contacts / f"{slug(key[0])}-{slug(key[1])}.png"
        group_sheet(key, members, destination, crop_boxes[key])
        group_sheets[key] = destination
    write_html(groups, group_sheets, thumbnails)
    pages = write_pdf(dataset_sheets, group_sheets)
    generated = [GALLERY / "index.html", GALLERY / "overview.png", *dataset_sheets.values(),
                 *group_sheets.values(), *thumbnails.values(), PDF]
    if not all(path.is_file() for path in generated):
        raise RuntimeError("A requested gallery output was not written")
    report = {
        "status": "complete",
        "renderer": manifest["renderer"],
        "input_manifest": str(MANIFEST.relative_to(ROOT)),
        "counts": {"structures": len(records), "groups": len(groups), "source_pngs_present_and_decoded": len(records),
                   "source_svgs_present": len(records), "source_xyz_files_present": len(records),
                   "unique_source_png_paths": len({input_path(r["png"]).resolve() for r in records}),
                   "unique_source_svg_paths": len({input_path(r["svg"]).resolve() for r in records}),
                   "representative_overviews": 4, "group_contact_sheets": len(group_sheets), "pdf_pages": pages,
                   "derived_thumbnail_pngs": len(thumbnails),
                   "generated_pngs": 4 + len(group_sheets) + len(thumbnails),
                   "generated_files_including_report": len(generated) + 1},
        "datasets": {dataset: {"groups": sum(k[0] == dataset for k in groups),
                                "structures": sum(r["dataset"] == dataset for r in records)} for dataset in DATASETS},
        "groups": [{"dataset": key[0], "group_id": key[1], "group_title": members[0]["group_title"],
                    "structures": len(members), "ordered_record_ids": [r["id"] for r in members],
                    "representative_id": next(r["id"] for r in members if r["representative"]),
                    "common_crop_box_pixels": crop_boxes[key],
                    "contact_sheet": str(group_sheets[key].relative_to(GALLERY))} for key, members in groups.items()],
        "outputs": [str(path.relative_to(ROOT)) for path in generated],
        "captions": {"dimers": DIMER_NOTE, "BH9": BH9_NOTE, "group_framing": GROUP_CROP_NOTE,
                     "overview_framing": OVERVIEW_NOTE, "visual_bonds": VISUAL_NOTE},
        "crop_box_convention": "[left, top, right, bottom), in original PNG pixels; foreground union plus 20-pixel margin",
        "verification_scope": "Unique PNG/SVG paths, input presence and PNG decoding; output presence. Visual PDF and HTML inspection is separate.",
    }
    (GALLERY / "gallery-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gallery": str(GALLERY / "index.html"), "pdf": str(PDF), "counts": report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
