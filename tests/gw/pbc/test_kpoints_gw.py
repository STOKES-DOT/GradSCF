"""k-point KRGW tests (Stage 2, k sampling).

Key consistency anchor: the k-point driver on a Gamma-only mesh must
reproduce the dedicated Gamma driver bit-for-bit-ish.
"""

import numpy as np
import jax.numpy as jnp

from gradscf.gw.pbc import KRGW
from gradscf.gw.pbc.ksigma import g0w0_cd_kpoints
from gradscf.pbc import gto, scf

_MESH = (13, 13, 13)
_NW = 30
_ATOM = "H 3.0 3.0 2.3; H 3.0 3.0 3.7"


def _cell():
    return gto.M(
        atom=_ATOM,
        a=np.eye(3) * 6.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-pade",
        mesh=_MESH,
        precision=1e-10,
    )


def test_kpoints_driver_matches_gamma_on_single_k():
    mf = scf.KRHF(_cell(), kpts=np.zeros((1, 3))).run()
    assert mf.converged
    result = mf.result
    res_k = g0w0_cd_kpoints(
        inputs=mf.inputs,
        kpts_frac=np.zeros((1, 3)),
        mo_energy_k=jnp.asarray(result.mo_energy_spin[0]),
        mo_coeff_k=jnp.asarray(result.mo_coeff_spin[0]),
        nocc=1,
        fock_k=jnp.asarray(result.fock_spin[0]),
        hcore_k=jnp.asarray(mf.inputs.hcore),
        density_spin=jnp.asarray(result.density_spin),
        mesh=_MESH,
        nw=_NW,
    )
    res_g = KRGW(mf, nw=_NW).run()
    np.testing.assert_allclose(res_k.mo_energy[0], np.asarray(res_g.mo_energy), atol=1e-8)


def test_kpoints_two_k_mesh_runs():
    # 2x1x1 mesh: exercises the momentum transfer machinery (q = 0, 1)
    cell = _cell()
    kpts = cell.make_kpts([2, 1, 1])
    mf = scf.KRHF(cell, kpts=kpts).run()
    assert mf.converged
    result = mf.result
    kpts_frac = np.asarray(kpts) @ np.asarray(mf.cell.lattice).T / (2.0 * np.pi)
    res = g0w0_cd_kpoints(
        inputs=mf.inputs,
        kpts_frac=kpts_frac,
        mo_energy_k=jnp.asarray(result.mo_energy_spin[0]),
        mo_coeff_k=jnp.asarray(result.mo_coeff_spin[0]),
        nocc=1,
        fock_k=jnp.asarray(result.fock_spin[0]),
        hcore_k=jnp.asarray(mf.inputs.hcore),
        density_spin=jnp.asarray(result.density_spin),
        mesh=_MESH,
        nw=_NW,
    )
    e_qp = np.asarray(res.mo_energy)
    assert e_qp.shape == (2, e_qp.shape[1])
    assert np.all(np.isfinite(e_qp))
    # the two k points of an isolated molecule in a box are nearly
    # degenerate; require only sanity of the spread
    assert np.abs(e_qp[0] - e_qp[1]).max() < 0.2
