"""Host-only physical observables and conventions; no electronic solvers."""
import math
import numpy as np

HARTREE_KCAL_MOL = 627.5094740631


def normalized_radial_shell(shell):
    l = int(shell[0])
    rows = np.asarray(shell[1:], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 2 or np.any(rows[:, 0] <= 0) or not np.isfinite(rows).all():
        raise ValueError('Expected finite positive Gaussian exponents and coefficients.')
    alpha, c = rows[:, 0], rows[:, 1:]
    power = l+1.5
    weights = c*np.sqrt(2*(2*alpha)**power/math.gamma(power))[:, None]
    metric = math.gamma(power)/(2*(alpha[:, None]+alpha[None, :])**power)
    norm = np.einsum('pi,pq,qi->i', weights, metric, weights)
    if np.any(norm <= 0):
        raise ValueError('A radial contraction has nonpositive norm.')
    return l, alpha, weights/np.sqrt(norm)[None, :]


def atomic_radial_density(shells, density, radius):
    """P(r)=r^2 integral rho(r,Omega)dOmega; integral P dr is electron number.

    One center, real spherical AOs, primitive/contraction/m ordering as in the
    integral layer. Angular integration keeps only equal l and equal m. Both
    spin-summed densities and an individual spin density are accepted.
    """
    density, radius = np.asarray(density), np.asarray(radius)
    nao = sum((2*int(s[0])+1)*(len(s[1])-1) for s in shells)
    if density.shape != (nao, nao) or radius.ndim != 1 or np.any(radius < 0):
        raise ValueError('Atomic density, spherical basis and radial grid do not match.')
    if not np.isfinite(density).all() or not np.isfinite(radius).all():
        raise ValueError('Density and radial grid must be finite.')
    if np.max(abs(density-density.conj().T)) > 1e-9:
        raise ValueError('Atomic density must be Hermitian.')
    channels, offset = {}, 0
    for shell in shells:
        l, alpha, weights = normalized_radial_shell(shell)
        radial = radius[:, None]**l*(np.exp(-radius[:, None]**2*alpha)@weights)
        for contraction in range(weights.shape[1]):
            indices = np.arange(offset, offset+2*l+1)
            channels.setdefault(l, []).append((indices, radial[:, contraction]))
            offset += 2*l+1
    result = np.zeros_like(radius, dtype=np.float64)
    for members in channels.values():
        for indices, first in members:
            for other, second in members:
                result += np.sum(density[indices, other]).real*first*second*radius**2
    return result


def counterpoise_energies(dimer, own_a, own_b, ghost_a, ghost_b, *, tolerance=1e-8):
    """Frozen orbital basis and common auxiliary metric in all five energies."""
    values = np.asarray([dimer, own_a, own_b, ghost_a, ghost_b], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Counterpoise energies must be finite.')
    if ghost_a > own_a+tolerance or ghost_b > own_b+tolerance:
        raise ValueError('Counterpoise violated nested variational fragment spaces.')
    uncorrected = dimer-own_a-own_b
    corrected = dimer-ghost_a-ghost_b
    bsse = own_a+own_b-ghost_a-ghost_b
    return dict(uncorrected_interaction_hartree=uncorrected,
                cp_interaction_hartree=corrected, bsse_hartree=bsse,
                bsse_kcal_mol=bsse*HARTREE_KCAL_MOL,
                cp_interaction_kcal_mol=corrected*HARTREE_KCAL_MOL,
                uncorrected_interaction_kcal_mol=uncorrected*HARTREE_KCAL_MOL)


def subset_packed_factors(factors, indices, *, nao):
    """Slice lower-triangular pair factors without creating a four-index ERI."""
    factors = np.asarray(factors)
    indices = np.asarray(indices, dtype=int)
    if (factors.ndim != 2 or factors.shape[1] != nao*(nao+1)//2
            or indices.ndim != 1 or len(set(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= nao)):
        raise ValueError('Invalid packed factors or fragment AO indices.')
    i, j = np.tril_indices(len(indices))
    hi, lo = np.maximum(indices[i], indices[j]), np.minimum(indices[i], indices[j])
    return factors[:, hi*(hi+1)//2+lo]


def polarizability_from_response(position, density_response, *, tolerance=1e-6):
    """alpha_ij=-Tr[r_i dD/dF_j], in Bohr^3; keep the measured asymmetry."""
    position, density_response = np.asarray(position), np.asarray(density_response)
    if position.ndim != 3 or position.shape[0] != 3 or density_response.shape != position.shape:
        raise ValueError('Position and density response require (3,nao,nao) arrays.')
    tensor = -np.einsum('ipq,jqp->ij', position, density_response).real
    if not np.isfinite(tensor).all() or np.max(abs(tensor-tensor.T)) > tolerance:
        raise ValueError('Polarizability must be finite and symmetric within the response tolerance.')
    return tensor
