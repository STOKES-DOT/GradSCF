"""Rigid benchmark geometry rules; standard library only, no optimization."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def archived_water(repo):
    path = Path("reports/tddft_perf/20261027-33a7cab/cases/water-pbe-tda.json")
    raw = (repo/path).read_bytes()
    case = json.loads(raw)
    settings = dict(unit=case.get("unit", "Angstrom"),
                    charge=case.get("charge", 0), spin=case.get("spin", 0))
    assert tuple(settings.values()) == ("Angstrom", 0, 0)
    atoms = [(p[0], *map(float, p[1:])) for p in (row.split() for row in case["atom"].split(";"))]
    assert [a[0] for a in atoms] == ["O", "H", "H"]
    return atoms, dict(water_case_file=str(path), water_case_settings=settings,
        water_case_sha256=hashlib.sha256(raw).hexdigest())


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _unit(a):
    norm = math.sqrt(_dot(a, a))
    assert norm > 1e-12
    return tuple(x/norm for x in a)


def orient_water(water, oxygen, outward):
    """Map the archived H--O--H bisector onto outward by a proper rotation."""
    vectors = [tuple(x-y for x, y in zip(a[1:], water[0][1:])) for a in water[1:]]
    z = _unit(tuple(a+b for a, b in zip(*vectors)))
    x = _unit(tuple(a-_dot(vectors[0], z)*b for a, b in zip(vectors[0], z)))
    local = (x, _cross(z, x), z)
    z_new = _unit(outward)
    anchor = (0., 0., 1.) if abs(z_new[2]) < .9 else (0., 1., 0.)
    x_new = _unit(_cross(anchor, z_new))
    target = (x_new, _cross(z_new, x_new), z_new)
    transformed = [("O", *oxygen)]
    for vector in vectors:
        components = [_dot(vector, axis) for axis in local]
        transformed.append(("H", *(oxygen[j]+sum(c*axis[j] for c, axis in zip(components, target)) for j in range(3))))
    error = max(abs(math.dist(a[1:], b[1:])-math.dist(c[1:], d[1:]))
        for i, (a, c) in enumerate(zip(water, transformed)) for b, d in zip(water[:i], transformed[:i]))
    assert error < 1e-12
    bisector = tuple(sum(h[j+1]-oxygen[j] for h in transformed[1:]) for j in range(3))
    assert _dot(_unit(bisector), z_new) > 1.-1e-12
    return transformed, error


def metal_water(seed, repo):
    """Fixed2.4 Angstrom diaqua/tetraqua cation input, not a bond-length claim."""
    water, provenance = archived_water(repo)
    count, distance = seed["waters"], 2.4
    if count == 2:
        directions = [(1., 0., 0.), (-1., 0., 0.)]
    elif count == 4:
        directions = [_unit(d) for d in ((1.,1.,1.), (1.,-1.,-1.), (-1.,1.,-1.), (-1.,-1.,1.))]
    else:
        raise ValueError("Only the declared diaqua/tetraqua benchmark rules are defined")
    atoms, errors = [(seed["ideal_metal"], 0., 0., 0.)], []
    for direction in directions:
        fragment, error = orient_water(water, tuple(distance*x for x in direction), direction)
        atoms.extend(fragment)
        errors.append(error)
    minimum = min(math.dist(a[1:], b[1:]) for i, a in enumerate(atoms) for b in atoms[:i])
    assert minimum > .2
    return atoms, dict(**provenance, metal=seed["ideal_metal"], waters=count,
        metal_oxygen_distance_angstrom=distance,
        oxygen_directions=directions, hydrogen_bisectors="radially outward from the cation",
        water_internal_distance_error_angstrom=max(errors), min_pair_distance_angstrom=minimum,
        geometry_rule="opposite axial diaqua or tetrahedral tetraqua; rigid waters",
        optimization="none", experimental_structure_claim=False)


def validate_coordination(atoms, seed, repo):
    """Check the serialized geometry, not only the unrounded construction."""
    water, _ = archived_water(repo)
    assert atoms[0][0] == seed["ideal_metal"] and len(atoms) == 1+3*seed["waters"]
    errors, distances, alignments = [], [], []
    for offset in range(1, len(atoms), 3):
        fragment = atoms[offset:offset+3]
        assert [a[0] for a in fragment] == ["O", "H", "H"]
        errors.extend(abs(math.dist(a[1:], b[1:])-math.dist(c[1:], d[1:]))
            for i, (a, c) in enumerate(zip(water, fragment)) for b, d in zip(water[:i], fragment[:i]))
        oxygen = fragment[0][1:]
        direction = tuple(x-y for x, y in zip(oxygen, atoms[0][1:]))
        bisector = tuple(sum(h[j+1]-oxygen[j] for h in fragment[1:]) for j in range(3))
        distances.append(abs(math.sqrt(_dot(direction, direction))-2.4))
        alignments.append(_dot(_unit(direction), _unit(bisector)))
    minimum = min(math.dist(a[1:], b[1:]) for i, a in enumerate(atoms) for b in atoms[:i])
    assert max(errors) < 1e-10 and max(distances) < 1e-10 and min(alignments) > 1.-1e-10 and minimum > .2
    return dict(water_internal_distance_error_angstrom=max(errors),
        metal_oxygen_distance_error_angstrom=max(distances),
        outward_bisector_cosine_min=min(alignments), min_pair_distance_angstrom=minimum)
