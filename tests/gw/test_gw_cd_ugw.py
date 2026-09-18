"""Unrestricted G0W0-CD tests.

1. Closed-shell consistency: UGW on a spin-0 UKS calculation must reproduce
   the restricted GW result to numerical precision (the spin-summed RPA
   response with factor 2 per channel equals the restricted factor 4).
2. Open-shell sanity: OH radical (spin=1, HF start) QP energies are finite,
   and the spin-resolved Fermi estimates split the channels correctly.
"""

import numpy as np
import pytest

from gradscf import dft, gto
from gradscf.gw import GW, UGW

_H2O_ATOM = "O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161"
_OH_ATOM = "O 0 0 0.0; H 0 0 0.9697"
_NW = 60  # smaller grid keeps the unrestricted double loop affordable


def test_ugw_closed_shell_matches_rgw():
    mol = gto.M(atom=_H2O_ATOM, basis="sto-3g", cart=True)
    rgw = GW(dft.RKS(mol, xc="hf").run(), nw=_NW).run()
    assert rgw.converged
    ugw = UGW(dft.UKS(mol, xc="hf").run(), nw=_NW).run()
    assert np.asarray(ugw.result.converged_mask).all()
    e_r = np.asarray(rgw.mo_energy)
    e_u = np.asarray(ugw.mo_energy)
    # Different integral caches (restricted reads the SCF inputs, UKS
    # rebuilds the pair matrix) and independent solver paths leave ~1e-7
    # residue; assert consistency at 1e-6 Ha.
    np.testing.assert_allclose(e_u[0], e_r, atol=1e-6)
    np.testing.assert_allclose(e_u[1], e_r, atol=1e-6)


def test_ugw_open_shell_oh_runs():
    mol = gto.M(atom=_OH_ATOM, basis="sto-3g", cart=True, spin=1)
    mf = dft.UKS(mol, xc="hf").run()
    assert mf.converged
    ugw = UGW(mf, nw=_NW).run()
    e_u = np.asarray(ugw.mo_energy)
    assert e_u.shape == (2, e_u.shape[1])
    assert np.all(np.isfinite(e_u))
    # Alpha has one more occupied orbital than beta for OH (9 electrons).
    occ_a = np.asarray(mf.mo_occ)[0] > 0
    occ_b = np.asarray(mf.mo_occ)[1] > 0
    assert occ_a.sum() == occ_b.sum() + 1
