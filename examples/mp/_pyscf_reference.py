"""Independent full-space RS reference for MP comparison examples only.

Uses PySCF integrals/actions and NumPy recurrence, without GradSCF or Taylor AD.
``frozen`` is a nonnegative core count applied to both spin channels. This
reference still allocates full determinant vectors, so use small systems.
"""
import numpy as np
from pyscf import ao2mo, fci


def mp_coefficients(mf, order, frozen=0):
    """Return individual E2 through Eorder in Hartree for a converged RHF/UHF."""
    unrestricted = np.asarray(mf.mo_coeff).ndim == 3
    coeff = mf.mo_coeff if unrestricted else (mf.mo_coeff, mf.mo_coeff)
    ca, cb = coeff
    nmo = ca.shape[1]
    na, nb = mf.mol.nelec
    if not mf.converged or not 0 <= frozen <= min(na, nb):
        raise ValueError("Use a converged reference and a valid frozen-core count")
    h = tuple(c.T @ mf.get_hcore() @ c for c in (coeff if unrestricted else (ca,)))
    frames = ((ca, ca, ca, ca), (ca, ca, cb, cb), (cb, cb, cb, cb)) if unrestricted else ((ca,)*4,)
    g = tuple(ao2mo.general(mf.mol, cs, compact=False).reshape((nmo,)*4)
              for cs in frames)
    ea, eb = mf.mo_energy if unrestricted else (mf.mo_energy, mf.mo_energy)
    aocc = fci.cistring.gen_occslst(range(nmo), na)
    bocc = fci.cistring.gen_occslst(range(nmo), nb)
    phi = np.zeros((len(aocc), len(bocc)))
    phi[0, 0] = 1.
    engine = fci.direct_uhf if unrestricted else fci.direct_spin1
    tensor = engine.absorb_h1e(h if unrestricted else h[0],
        g if unrestricted else g[0], nmo, (na, nb), .5)
    action = lambda c: np.asarray(engine.contract_2e(tensor, c, nmo, (na, nb)))
    e0 = action(phi)[0, 0]
    gaps = (np.sum(ea[aocc], axis=1)[:, None]+np.sum(eb[bocc], axis=1)[None, :]
            -np.sum(ea[:na])-np.sum(eb[:nb]))
    amask = np.array([all(i in occ for i in range(frozen)) for occ in aocc])
    bmask = np.array([all(i in occ for i in range(frozen)) for occ in bocc])
    mask = amask[:, None] & bmask[None, :]
    mask[0, 0] = False
    waves, energies = [phi], [e0]
    for degree in range(1, order+1):
        source = action(waves[-1])-(e0+gaps)*waves[-1]
        energies.append(source[0, 0])
        for j in range(1, degree):
            source -= energies[j]*waves[degree-j]
        waves.append(np.where(mask, -source/np.where(mask, gaps, 1.), 0.))
    return np.asarray(energies[2:])
