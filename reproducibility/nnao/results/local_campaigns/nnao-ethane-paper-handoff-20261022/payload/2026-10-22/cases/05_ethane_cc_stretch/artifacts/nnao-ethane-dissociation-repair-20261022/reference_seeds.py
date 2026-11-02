"""Independent NumPy atomic-block Lowdin seed for the H/C comparison cases."""
import numpy as np


def atomic_population_seed(mol):
    if any(int(z) not in (1, 6) for z in mol.atom_charges()):
        raise ValueError('This explicit reference occupation is defined only for H and C.')
    owners, occupation = [], []
    seen = {}
    for shell in range(mol.nbas):
        atom, ell = mol.bas_atom(shell), mol.bas_angular(shell)
        charge = int(mol.atom_charge(atom))
        for _ in range(mol.bas_nctr(shell)):
            radial = seen.get((atom, ell), 0)
            number = (1. if charge == 1 else 2.) if ell == 0 and radial < (1 if charge == 1 else 2) else 0.
            if charge == 6 and ell == 1 and radial == 0:
                number = 2. / 3.
            owners.extend([atom] * (2*ell+1))
            occupation.extend([number] * (2*ell+1))
            seen[atom, ell] = radial + 1
    owners, occupation = np.asarray(owners), np.asarray(occupation)
    assert len(owners) == mol.nao_nr() and abs(occupation.sum()-mol.nelectron) < 1e-12
    mask = owners[:, None] == owners[None, :]
    overlap = mol.intor_symmetric('int1e_ovlp')
    values, vectors = np.linalg.eigh(np.where(mask, overlap, 0.))
    if values.min() <= 1e-10:
        raise ValueError('Atomic reference metric is singular; do not clip silently.')
    x = (vectors * values**-.5) @ vectors.T
    return np.where(mask, (x * occupation) @ x.T, 0.)
