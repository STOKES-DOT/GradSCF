"""Rigid, symmetric separation of the two ethane methyl groups."""
from copy import deepcopy
import numpy as np


def stretch_rows(reference, distances):
    xyz = np.asarray(reference['coords_angstrom'], dtype=np.float64)
    symbols = np.asarray(reference['symbols'])
    carbon, hydrogen = np.flatnonzero(symbols == 'C'), np.flatnonzero(symbols == 'H')
    assert len(carbon) == 2 and len(hydrogen) == 6
    owners = np.argmin(np.linalg.norm(xyz[hydrogen, None]-xyz[carbon], axis=2), axis=1)
    groups = [np.r_[carbon[k], hydrogen[owners == k]] for k in (0, 1)]
    assert all(len(group) == 4 for group in groups)
    axis = xyz[carbon[1]]-xyz[carbon[0]]
    original_distance = float(np.linalg.norm(axis))
    axis /= original_distance
    rows = []
    for distance in distances:
        distance = float(distance)
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError('C-C distances must be positive and finite.')
        coordinates = xyz.copy()
        for k, group in enumerate(groups):
            coordinates[group] += (2*k-1)*(distance-original_distance)*axis/2
            np.testing.assert_allclose(coordinates[group]-coordinates[group[0]],
                                       xyz[group]-xyz[group[0]], atol=2e-15, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(coordinates[carbon[1]]-coordinates[carbon[0]]),
                                   distance, atol=2e-15, rtol=0)
        row = {key: deepcopy(reference[key]) for key in
               ['symbols', 'atomic_numbers', 'natoms', 'charge', 'spin', 'nelectron']}
        row.update(structure_id=f'ethane-cc-stretch/{distance:.12f}',
            source_structure_id=reference.get('source_structure_id', reference['structure_id']),
            distance_angstrom=distance, reference_distance_angstrom=original_distance,
            coords_angstrom=coordinates.tolist(), methyl_groups=[group.tolist() for group in groups],
            geometry_optimized=False, electronic_branch='restricted closed-shell singlet RHF')
        rows.append(row)
    return rows
