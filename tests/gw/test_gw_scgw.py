"""scGW (Stage 5) tests: matrix self-consistent GW on H2."""

import numpy as np
import jax.numpy as jnp

from gradscf import dft, gto
from gradscf.gw import scgw_cd_restricted

_NW = 40


def test_scgw_h2_sto3g_converges_with_sane_energy():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", cart=True)
    mf = dft.RKS(mol, xc="hf").run()
    res = mf.scf_result
    from gradscf.df import eri_pair_matrix_to_df_factors

    df = eri_pair_matrix_to_df_factors(
        mf._scf_inputs.eri_pair_matrix, nao=res.mo_coeff.shape[0], tol=1e-12
    )
    out = scgw_cd_restricted(
        mo_energy=jnp.asarray(res.mo_energy),
        mo_coeff=jnp.asarray(res.mo_coeff),
        nocc=1,
        df_factors=df,
        hcore_matrix=jnp.asarray(res.hcore_matrix),
        nuclear_repulsion=float(res.nuclear_repulsion),
        nw=_NW,
        max_iter=25,
        tol=1e-5,
        mixing=0.5,
    )
    assert out.converged
    e_c = float(out.correlation_energy)
    e_tot = float(out.total_energy)
    e_hf = float(res.total_energy)
    # correlation energy is negative and bounded for H2
    assert -0.2 < e_c < 0.0
    # total energy sits below the HF energy (GM energy includes E_c)
    assert e_tot < e_hf + 1e-6
    # chemical potential sits inside the gap
    e_qp = np.asarray(out.mo_energy)
    assert e_qp[0] < float(out.chemical_potential) < e_qp[1]
