"""Render the four repaired cases from their exact training geometries."""
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.cache'))
os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = '/opt/homebrew/lib:/usr/lib'
import cairosvg
import numpy as np
from xyzrender import load, render

NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
SOURCE = ROOT / 'artifacts/nnao-stage2-20261007/selection/train.jsonl'
IDS = ['MCONF/21', 'MCONF/44', 'Amino20x4/ARG_xci', 'S66/33']
rows = {row['structure_id']: row for row in map(json.loads, SOURCE.read_text().splitlines())}


def node(parent, tag, **attributes):
    return ET.SubElement(parent, f'{{{NS}}}{tag}', {k.replace('_', '-'): str(v) for k, v in attributes.items()})


def text(parent, x, y, value, size=24, weight='normal', color='#526070'):
    element = node(parent, 'text', x=x, y=y, font_size=size, font_weight=weight,
                   fill=color, font_family='Arial, sans-serif')
    element.text = value


def formula(symbols):
    counts = Counter(symbols)
    order = ['C', 'H'] + sorted(set(counts) - {'C', 'H'})
    return ''.join(s + (str(counts[s]) if counts[s] > 1 else '') for s in order if counts[s])


def unique_svg_ids(svg, prefix):
    identifiers = {e.attrib['id']: prefix + e.attrib['id'] for e in svg.iter() if 'id' in e.attrib}
    for element in svg.iter():
        for key, value in list(element.attrib.items()):
            if key == 'id':
                element.set(key, identifiers[value])
            else:
                for original, replacement in identifiers.items():
                    value = value.replace(f'url(#{original})', f'url(#{replacement})')
                    if value == f'#{original}':
                        value = f'#{replacement}'
                element.set(key, value)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    board = ET.Element(f'{{{NS}}}svg', width='2200', height='2100', viewBox='0 0 2200 2100')
    node(board, 'rect', width=2200, height=2100, fill='#f6f8fb')
    text(board, 60, 72, 'NNAO: four formerly failing structures', size=46, weight='bold', color='#1f2937')
    text(board, 60, 118, 'Original GMTKN55 input geometries  |  all hydrogen atoms shown  |  xyzrender', size=25)
    manifest = dict(source=str(SOURCE.relative_to(ROOT)),
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    xyzrender_version=version('xyzrender'), coordinate_unit='angstrom',
                    geometry_optimized=False, panels_scaled_independently=True, structures=[])
    reference = OUT / 'MCONF_oriented_reference.xyz'
    for index, identifier in enumerate(IDS):
        row = rows[identifier]
        stem = identifier.replace('/', '_')
        xyz = OUT / f'{stem}.xyz'
        lines = [str(len(row['symbols'])), f'{identifier}; coordinates in angstrom; charge={row["charge"]}; spin={row["spin"]}']
        lines.extend(s + ' ' + ' '.join(format(float(x), '.17g') for x in p)
                     for s, p in zip(row['symbols'], row['coords_angstrom'], strict=True))
        xyz.write_text('\n'.join(lines) + '\n')
        restored = np.array([[float(x) for x in line.split()[1:]] for line in lines[2:]])
        np.testing.assert_array_equal(restored, np.asarray(row['coords_angstrom']))
        mol = load(xyz, charge=row['charge'], multiplicity=row['spin'] + 1, bohr=False)
        options = dict(config='flat', hy=True, canvas_size=1000, transparent=True, fog=False)
        if identifier.startswith('MCONF/'):
            options['ref'] = reference
        elif identifier == 'S66/33':
            # The global PCA view hides the ring face in this two-fragment case.
            # Rotate the view rigidly; leave the exported input geometry intact.
            theta = np.deg2rad(35.)
            rotation = np.array([[1., 0., 0.], [0., np.cos(theta), -np.sin(theta)],
                                 [0., np.sin(theta), np.cos(theta)]])
            before = np.array([data['position'] for _, data in mol.graph.nodes(data=True)])
            after = (before - before.mean(axis=0)) @ rotation.T
            np.testing.assert_allclose(np.linalg.norm(before[:, None] - before[None, :], axis=-1),
                                       np.linalg.norm(after[:, None] - after[None, :], axis=-1),
                                       atol=2e-14, rtol=2e-14)
            for (_, data), position in zip(mol.graph.nodes(data=True), after, strict=True):
                data['position'] = tuple(position)
            options['orient'] = False
        result = render(mol, output=OUT / f'{stem}.svg', **options)
        cairosvg.svg2png(bytestring=str(result).encode(), write_to=str(OUT / f'{stem}.png'))
        x, y = 60 + (index % 2) * 1080, 175 + (index // 2) * 910
        node(board, 'rect', x=x, y=y, width=1000, height=860, rx=20, fill='white',
             stroke='#dbe2ec', stroke_width=2)
        text(board, x + 30, y + 53, f'({chr(97 + index)})  {identifier}', size=33,
             weight='bold', color='#1f2937')
        sizes = row['size_estimate']
        text(board, x + 30, y + 91,
             f'{formula(row["symbols"])}   |   {row["natoms"]} atoms   |   {sizes["nao"]} AOs   |   {sizes["primitive_nao"]} primitive AOs', size=24)
        svg = ET.fromstring(str(result))
        unique_svg_ids(svg, f'panel{index}_')
        svg.set('x', str(x + 30)); svg.set('y', str(y + 116))
        svg.set('width', '940'); svg.set('height', '690')
        board.append(svg)
        problem = 'Earlier failure: GPU out of memory' if identifier != 'S66/33' else 'Earlier failure: SCF charge oscillation'
        text(board, x + 30, y + 834, problem, size=23)
        manifest['structures'].append(dict(structure_id=identifier, natoms=row['natoms'],
            formula=formula(row['symbols']), source_coordinate_signature=row['coordinate_signature_sha256'],
            xyz_sha256=hashlib.sha256(xyz.read_bytes()).hexdigest(),
            nao=sizes['nao'], primitive_nao=sizes['primitive_nao'],
            xyz=xyz.name, svg=f'{stem}.svg', png=f'{stem}.png',
            orientation=('Kabsch-aligned common MCONF view' if identifier.startswith('MCONF/')
                         else 'rigid x-axis rotation, 35 degrees' if identifier == 'S66/33' else 'PCA view')))
    for index, (symbol, color) in enumerate([('C', '#909090'), ('H', '#ffffff'), ('N', '#3050f8'), ('O', '#ff0d0d')]):
        x = 80 + index * 110
        node(board, 'circle', cx=x, cy=2010, r=15, fill=color, stroke='#526070', stroke_width=1.5)
        text(board, x + 24, 2019, symbol, size=27, color='#1f2937')
    text(board, 590, 2019, 'AOs: contracted basis functions used in SCF', size=24)
    text(board, 60, 2066, 'View rotation only; no geometry optimization. Panels are scaled independently.', size=23)
    svg_path = OUT / 'four_structures.svg'
    svg_path.write_bytes(ET.tostring(board, encoding='utf-8', xml_declaration=True))
    cairosvg.svg2png(url=str(svg_path), write_to=str(OUT / 'four_structures.png'))
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(OUT / 'four_structures.pdf'))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
