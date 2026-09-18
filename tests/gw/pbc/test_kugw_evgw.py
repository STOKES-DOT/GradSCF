"""KUGW (Gamma unrestricted) and evGW-Gamma tests."""

import numpy as np

from gradscf.gw.pbc import KRGW, KUGW, evgw_cd_gamma
from gradscf.pbc import gto, scf

_MESH = (17, 17, 17)
_NW = 40
_ATOM = "H 3.0 3.0 2.3; H 3.0 3.0 3.7"


def _mf(unrestricted=False):
    cell = gto.M(
        atom=_ATOM,
        a=np.eye(3) * 6.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-pade",
        mesh=_MESH,
        precision=1e-10,
    )
    cls = scf.UHF if unrestricted else scf.RHF
    mf = cls(cell).run()
    assert mf.converged
    return mf


def test_kugw_closed_shell_matches_krgw():
    rgw = KRGW(_mf(), nw=_NW).run()
    ugw = KUGW(_mf(unrestricted=True), nw=_NW).run()
    e_r = np.asarray(rgw.mo_energy)
    e_u = np.asarray(ugw.mo_energy)
    np.testing.assert_allclose(e_u[0], e_r, atol=1e-5)
    np.testing.assert_allclose(e_u[1], e_r, atol=1e-5)


def test_evgw_gamma_fixed_point():
    mf = _mf()
    result = mf.result
    res = evgw_cd_gamma(
        inputs=mf.inputs,
        mo_energy=result.mo_energy_spin[0, 0],
        mo_coeff=result.mo_coeff_spin[0, 0],
        nocc=1,
        fock_matrix=result.fock_spin[0, 0],
        hcore_matrix=mf.inputs.hcore[0],
        density_spin=result.density_spin[:, 0],
        mesh=_MESH,
        nw=_NW,
        max_iter=20,
        tol=1e-6,
    )
    assert res.converged
    g0 = KRGW(mf, nw=_NW).run()
    delta = np.asarray(res.mo_energy) - np.asarray(g0.mo_energy)
    assert np.all(np.isfinite(delta))
    assert np.max(np.abs(delta)) < 0.1
