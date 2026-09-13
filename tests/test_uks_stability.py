"""Shared unrestricted stability, including the XC spin response."""
import numpy as np
import pytest
import json
from gradscf import scf


def test_uks_hf_limit_shares_stability_and_restart():
    h=np.array([[-1.,.1],[.1,.4]])
    common=dict(overlap=np.eye(2),hcore=h,eri=np.zeros((2,2,2,2)),
                nalpha=1,nbeta=1,nuclear_repulsion=0.)
    hf=scf.stabilize_uhf_from_integrals(**common)
    ks=scf.stabilize_uks_from_integrals(**common,ao=np.zeros((0,2)),
        ao_deriv1=np.zeros((4,0,2)),grid_weights=np.zeros(0),config=scf.UKSConfig(xc_spec='hf'))
    assert ks.stable and ks.restarts==0
    np.testing.assert_allclose(ks.stability.eigenvalues,hf.stability.eigenvalues,atol=1e-12,rtol=0)


@pytest.mark.parametrize('xc',['svwn','pbe','pbe0'])
def test_stretched_h2_uks_spin_instability_and_restart(xc):
    pytest.importorskip('jax_xc')
    pytest.importorskip('pyscf')
    from comparisons.ten_system_scf_matrix import prepare, XC_REFERENCE
    from pyscf import dft
    from pyscf.soscf import newton_ah
    p=prepare(dict(atom='H 0 0 0; H 0 0 2.5',charge=0,spin=0),'3-21g',1)
    cfg=scf.UKSConfig(xc_spec=xc,max_cycle=200,conv_tol=1e-11,conv_tol_density=1e-9)
    args=dict(overlap=p['s'],hcore=p['h'],eri=p['eri'],nalpha=1,nbeta=1,
        nuclear_repulsion=p['enuc'],ao=p['ao'],ao_deriv1=p['deriv'],grid_weights=p['weights'],
        init_density_alpha=p['da'],init_density_beta=p['db'],config=cfg)
    saddle=scf.run_uks_from_integrals(**args)
    assert saddle.converged
    checked=scf.uks_stability(saddle,eri=p['eri'],ao=p['ao'],ao_deriv1=p['deriv'],
                             grid_weights=p['weights'],config=cfg)
    assert checked.stable is False
    ref=dft.UKS(p['mol'])
    ref.xc=XC_REFERENCE[xc]
    ref.grids.coords=p['coords'];ref.grids.weights=p['weights'];ref.small_rho_cutoff=0.
    ref.mo_coeff=np.stack([saddle.mo_coeff_alpha,saddle.mo_coeff_beta])
    ref.mo_occ=np.stack([saddle.mo_occ_alpha,saddle.mo_occ_beta])
    # Independent analytic PySCF XC-response Hessian, dense only as test oracle.
    _,hop,diagonal=newton_ah.gen_g_hop_uhf(ref,ref.mo_coeff,ref.mo_occ)
    dense=np.stack([2*hop(v) for v in np.eye(len(diagonal))],axis=1)
    np.testing.assert_allclose(checked.eigenvalues,np.linalg.eigvalsh(dense)[:3],atol=2e-6,rtol=0)
    outcome=scf.stabilize_uks_from_integrals(**args)
    assert outcome.stable and outcome.result.converged and outcome.restarts>=1
    assert outcome.result.total_energy<saddle.total_energy-1e-3
    ref.conv_tol=1e-11;ref.conv_tol_grad=1e-7;ref.max_cycle=200
    # Use the independent dense oracle to seed the reference restart. At an
    # exactly spin-symmetric stationary point a gradient-seeded iterative
    # stability search can miss the spin-breaking sector.
    from pyscf.scf.stability import _rotate_mo
    _,vectors=np.linalg.eigh(dense)
    split=(p['s'].shape[0]-1)
    mo=tuple(_rotate_mo(ref.mo_coeff[s],ref.mo_occ[s],vectors[s*split:(s+1)*split,0])
             for s in range(2))
    ref.kernel(dm0=ref.make_rdm1(mo,ref.mo_occ))
    assert ref.converged
    np.testing.assert_allclose(outcome.result.total_energy,ref.e_tot,atol=1e-7,rtol=0)
    print('UKS_STABILITY_RESULT '+json.dumps(dict(xc=xc,initial_energy=saddle.total_energy,
        final_energy=outcome.result.total_energy,reference_energy=ref.e_tot,
        energy_error=abs(outcome.result.total_energy-ref.e_tot),restarts=outcome.restarts,
        minimum_curvature=outcome.minimum_curvature,
        hessian_error=float(np.max(np.abs(np.asarray(checked.eigenvalues)-np.linalg.eigvalsh(dense)[:3]))))))
