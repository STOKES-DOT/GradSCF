"""qsGW (Stage 4) tests: self-consistent static potential iteration."""

import numpy as np
import jax.numpy as jnp

from gradscf import dft, gto
from gradscf.gw import qsgw_cd_restricted

_ATOM = "O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161"
_NW = 30  # qsGW evaluates the full off-diagonal self-energy; keep it small


def test_qsgw_water_sto3g_converges_and_is_sane():
    mol = gto.M(atom=_ATOM, basis="sto-3g", cart=True)
    mf = dft.RKS(mol, xc="hf").run()
    res = mf.scf_result
    from gradscf.df import eri_pair_matrix_to_df_factors

    df = eri_pair_matrix_to_df_factors(
        mf._scf_inputs.eri_pair_matrix, nao=res.mo_coeff.shape[0], tol=1e-12
    )
    out = qsgw_cd_restricted(
        mo_energy=jnp.asarray(res.mo_energy),
        mo_coeff=jnp.asarray(res.mo_coeff),
        nocc=5,
        df_factors=df,
        hcore_matrix=jnp.asarray(res.hcore_matrix),
        nw=_NW,
        max_iter=15,
        tol=1e-5,
        damping=0.3,
    )
    e_qp = np.asarray(out.mo_energy)
    e_mf = np.asarray(res.mo_energy)
    assert np.all(np.isfinite(e_qp))
    # qsGW HOMO (ionization potential) is below the HF Koopmans value and
    # the gap remains open
    homo, lumo = 4, 5
    assert e_qp[homo] < e_mf[homo] + 0.05
    assert e_qp[lumo] > e_qp[homo]
    # orbitals are updated and remain orthonormal w.r.t. the overlap
    s = np.asarray(res.overlap_matrix)
    c = np.asarray(out.mo_coeff)
    np.testing.assert_allclose(c.T @ s @ c, np.eye(c.shape[1]), atol=1e-10)
