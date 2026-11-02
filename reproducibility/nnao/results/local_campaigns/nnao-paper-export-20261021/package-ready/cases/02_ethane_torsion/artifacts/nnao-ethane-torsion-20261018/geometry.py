"""Rigid methyl rotation, with all bond lengths and valence angles preserved."""
from copy import deepcopy
import numpy as np


def torsion_rows(reference, angles):
    symbols = reference['symbols']
    assert symbols.count('C') == 2 and symbols.count('H') == 6
    xyz = np.asarray(reference['coords_angstrom'], dtype=float)
    carbon = np.flatnonzero(np.asarray(symbols) == 'C')
    hydrogen = np.flatnonzero(np.asarray(symbols) == 'H')
    parent = carbon[np.linalg.norm(xyz[hydrogen, None]-xyz[carbon], axis=2).argmin(axis=1)]
    moving = hydrogen[parent == carbon[1]]
    assert len(moving) == 3
    fixed = [i for i in range(len(xyz)) if i not in moving]
    axis = xyz[carbon[1]]-xyz[carbon[0]]
    axis /= np.linalg.norm(axis)
    pivot = xyz[carbon[1]]
    distances = lambda points: np.linalg.norm(points[:, None]-points[None], axis=2)
    rows = []
    for angle in angles:
        a = np.deg2rad(angle)
        relative = xyz[moving]-pivot
        rotated = (relative*np.cos(a)+np.cross(axis, relative)*np.sin(a)
                   +np.outer(relative@axis, axis)*(1-np.cos(a)))
        coordinates = xyz.copy()
        coordinates[moving] = rotated+pivot
        np.testing.assert_allclose(coordinates[fixed], xyz[fixed], atol=1e-13, rtol=0)
        group = [int(carbon[0]), int(carbon[1]), *map(int, moving)]
        np.testing.assert_allclose(distances(coordinates[group]), distances(xyz[group]), atol=1e-12, rtol=0)
        row = {key: deepcopy(reference[key]) for key in
               ['symbols', 'atomic_numbers', 'natoms', 'charge', 'spin', 'nelectron']}
        row.update(structure_id=f'ethane-torsion/{int(angle):03d}',
                   source_structure_id=reference['structure_id'], angle_deg=float(angle),
                   coords_angstrom=coordinates.tolist(), rotation_axis_atoms=carbon.tolist(),
                   rotating_hydrogen_indices=moving.tolist(), geometry_optimized=False)
        rows.append(row)
    first, last = rows[0], rows[-1]
    if last['angle_deg']-first['angle_deg'] == 120:
        np.testing.assert_allclose(np.sort(distances(np.asarray(first['coords_angstrom'])).ravel()),
            np.sort(distances(np.asarray(last['coords_angstrom'])).ravel()), atol=1e-11, rtol=0)
    return rows
