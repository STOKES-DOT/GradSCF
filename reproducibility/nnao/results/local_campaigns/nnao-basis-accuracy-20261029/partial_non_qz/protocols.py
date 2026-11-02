"""Size and scoring conventions for the frozen 27-basis comparison."""
import numpy as np


def basis_counts(atom_shells):
    nao, primitives = 0, set()
    for atom, shells in enumerate(atom_shells):
        for shell in shells:
            l, rows = int(shell[0]), np.asarray(shell[1:], dtype=float)
            if rows.ndim != 2 or rows.shape[1] < 2 or not np.isfinite(rows).all():
                raise ValueError("Malformed shell")
            nao += (2*l+1)*(rows.shape[1]-1)
            for row in rows:
                if row[0] <= 0:
                    raise ValueError("Primitive exponents must be positive")
                if np.any(row[1:] != 0):
                    primitives.add((atom, l, float(row[0])))
    return dict(nao_spherical=nao, primitive_radial_unique=len(primitives),
                primitive_spherical_unique=sum(2*l+1 for _, l, _ in primitives))


def frozen_core_count(nuclear_charges):
    z = np.asarray(nuclear_charges)
    if any(int(x) not in {0, 1, 6, 7, 8} for x in z):
        raise ValueError("This benchmark's frozen-core policy covers H/C/N/O and ghosts only")
    return int(np.count_nonzero(z > 2))


def balanced_mae(isomer_errors, dimer_errors):
    if len(isomer_errors) != 6 or len(dimer_errors) != 6:
        raise ValueError("Both categories require all six cases")
    values = np.asarray([isomer_errors, dimer_errors], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("A partial or nonfinite score cannot be published")
    return float(np.mean(abs(values)))


def cp_estimate(total, isolated, conditioned, ghosts):
    if not np.isfinite([total, *isolated, *conditioned, *ghosts]).all():
        raise ValueError("Nonfinite CP energies")
    model = float(total-sum(isolated))
    bsse = float(sum(conditioned)-sum(ghosts))
    return dict(model_interaction_hartree=model, bsse_estimate_hartree=bsse,
                cp_estimate_hartree=model+bsse, conditional_cp_hartree=float(total-sum(ghosts)))


def fragment_indices(row):
    """Identify the two bonded components, independent of file atom ordering."""
    radii={'H':.31,'C':.76,'N':.71,'O':.66}
    positions=np.asarray(row['coords_angstrom'])
    unseen=set(range(len(positions)));groups=[]
    while unseen:
        group=set();stack=[min(unseen)]
        while stack:
            i=stack.pop()
            if i not in unseen:continue
            unseen.remove(i);group.add(i)
            stack.extend(j for j in unseen if np.linalg.norm(positions[i]-positions[j]) <
                         1.2*(radii[row['symbols'][i]]+radii[row['symbols'][j]]))
        groups.append(sorted(group))
    if len(groups)!=2:raise ValueError(f'Expected two molecular fragments, found {groups}')
    return groups
