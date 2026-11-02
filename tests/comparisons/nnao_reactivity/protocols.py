"""Explicit states, signed stoichiometry and electronic energy conventions."""
from collections import Counter
import re

import numpy as np

HARTREE_TO_KCAL_MOL = 627.5094740631


def bse_element_shells(element):
    """Preserve BSE radial contractions; the benchmark requests spherical AOs."""
    if element.get('ecp_electrons', 0) or element.get('ecp_potentials'):
        raise ValueError('ECP basis data cannot enter the all-electron benchmark.')
    result = []
    for shell in element['electron_shells']:
        exponents = np.asarray(shell['exponents'], dtype=float)
        coefficients = np.asarray(shell['coefficients'], dtype=float).T
        angular = shell['angular_momentum']
        if not np.isfinite(exponents).all() or np.any(exponents <= 0) or not np.isfinite(coefficients).all():
            raise ValueError('Invalid BSE exponents or coefficients.')
        if coefficients.shape[0] != len(exponents):
            raise ValueError('BSE contraction length differs from its exponent list.')
        if len(angular) == 1:
            result.append([angular[0]]+np.column_stack([exponents, coefficients]).tolist())
        elif len(angular) == coefficients.shape[1] and len(set(angular)) == len(angular):
            for column, momentum in enumerate(angular):
                result.append([momentum]+np.column_stack([exponents, coefficients[:, column]]).tolist())
        else:
            raise ValueError('Ambiguous combined-angular BSE contraction layout.')
    if not result:
        raise ValueError('No orbital shells in BSE element data.')
    return result


def parse_din(text):
    lines = [s.strip() for s in text.splitlines() if s.strip() and not s.lstrip().startswith('#')]
    rows, position = [], 0
    while position < len(lines):
        terms = {}
        while True:
            if position >= len(lines):
                raise ValueError('Unterminated DIN reaction.')
            coefficient = int(lines[position])
            position += 1
            if coefficient == 0:
                break
            if position >= len(lines):
                raise ValueError('Missing DIN species.')
            name = lines[position]
            terms[name] = terms.get(name, 0) + coefficient
            position += 1
        if position >= len(lines) or not terms:
            raise ValueError('Missing DIN reference energy or terms.')
        fields = lines[position].split(maxsplit=1)
        value = float(fields[0])
        if not np.isfinite(value):
            raise ValueError('Nonfinite reference energy.')
        rows.append(dict(terms=terms, reference_kcal_mol=value,
                         label=fields[1] if len(fields) == 2 else ''))
        position += 1
    return rows


def parse_xyz(text, identifier):
    from gradscf.data.molecule import atomic_number
    lines = text.splitlines()
    count = int(lines[0])
    state = re.match(r'^\s*(-?\d+)\s+(\d+)(?:\s|$)', lines[1])
    if state is None:
        raise ValueError('XYZ must explicitly provide charge and multiplicity: ' + identifier)
    charge, multiplicity = map(int, state.groups())
    if len(lines[2:]) < count:
        raise ValueError('Truncated XYZ: ' + identifier)
    values = [line.split() for line in lines[2:2+count]]
    symbols = [v[0] for v in values]
    numbers = [atomic_number(s) for s in symbols]
    coordinates = np.asarray([[float(x) for x in v[1:4]] for v in values])
    nelectron, spin = sum(numbers)-charge, multiplicity-1
    if coordinates.shape != (count, 3) or not np.isfinite(coordinates).all():
        raise ValueError('Invalid XYZ coordinates.')
    if spin < 0 or spin > nelectron or (nelectron-spin) % 2:
        raise ValueError('XYZ charge/multiplicity violates electron parity.')
    return dict(structure_id=identifier, label=identifier, symbols=symbols,
                atomic_numbers=numbers, coords_angstrom=coordinates.tolist(),
                natoms=count, charge=charge, spin=spin, nelectron=nelectron,
                source_state_comment=lines[1])


def validate_reaction(reactants, transition, products, structures):
    def signature(terms):
        elements, charge = Counter(), 0
        for name, coefficient in terms.items():
            if not isinstance(coefficient, int) or coefficient <= 0:
                raise ValueError('Stoichiometry must be positive integers.')
            row = structures[name]
            elements.update({s: n*coefficient for s, n in Counter(row['symbols']).items()})
            charge += row['charge']*coefficient
        return elements, charge
    if not signature(reactants) == signature(products) == signature({transition: 1}):
        raise ValueError('Reaction elements and charge must be balanced at R, TS and P.')


def reaction_profile(reactants, transition, products, energies):
    er = sum(count*energies[name] for name, count in reactants.items())
    ep = sum(count*energies[name] for name, count in products.items())
    ets = energies[transition]
    relative = np.array([0., ets-er, ep-er])
    if not np.isfinite(relative).all():
        raise ValueError('Nonfinite reaction energy.')
    kcal = relative*HARTREE_TO_KCAL_MOL
    return dict(relative_hartree=relative.tolist(), relative_kcal_mol=kcal.tolist(),
                forward_kcal_mol=float(kcal[1]), reverse_kcal_mol=float(kcal[1]-kcal[2]),
                reaction_kcal_mol=float(kcal[2]))
