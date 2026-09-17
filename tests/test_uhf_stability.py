"""Internal real-orbital UHF stability and directed saddle escape."""
import numpy as np
import pytest
import json
from pathlib import Path
from scipy.linalg import eigh
from gradscf import scf


def test_oh_saddle_is_detected_and_followed_to_stable_solution():
    pytest.importorskip('pyscf')
    from pyscf import gto, scf as reference_scf
    mol=gto.M(atom='O 0 0 0; H 0 0 .97',basis='3-21g',spin=1,cart=True,verbose=0)
    ref=reference_scf.UHF(mol)
    s,h=ref.get_ovlp(),ref.get_hcore()
    _,c=eigh(h,s)
    da,db=(c[:,:n]@c[:,:n].T for n in mol.nelec)
    # Fix the stationary state itself: tiny backend differences in a core
    # guess can otherwise change which SCF basin is reached before this test.
    seed=np.asarray(json.loads((Path(__file__).parent/'fixtures'/'oh_uhf_saddle_321g.json').read_text())['density'])
    args=dict(overlap=s,hcore=h,eri=mol.intor('int2e'),nalpha=5,nbeta=4,
              nuclear_repulsion=mol.energy_nuc(),init_density_alpha=seed[0],init_density_beta=seed[1],
              config=scf.UHFConfig(max_cycle=200,conv_tol=1e-11,conv_tol_density=1e-9))
    saddle=scf.run_uhf_from_integrals(**args)
    assert saddle.converged
    assert saddle.total_energy > -74.81
    checked=scf.stabilize_uhf_from_integrals(**args,max_restarts=0)
    assert not checked.stable
    assert checked.minimum_curvature < -.3
    fixed=scf.stabilize_uhf_from_integrals(**args)
    assert fixed.stable and fixed.result.converged
    assert fixed.restarts > 0
    ref.conv_tol=1e-12
    ref.conv_tol_grad=1e-8
    ref.kernel(dm0=np.array([da,db]))
    np.testing.assert_allclose(fixed.result.total_energy,ref.e_tot,atol=1e-8,rtol=0)
    assert fixed.result.total_energy < saddle.total_energy-.16
    assert fixed.stability.eigensolver_converged
    assert np.max(fixed.stability.residual_norms) < 1.01e-7
    np.testing.assert_allclose(fixed.stability.mo_coeff,
        [fixed.result.mo_coeff_alpha, fixed.result.mo_coeff_beta],atol=0,rtol=0)
    # An unresolved eigenproblem must not be silently labelled stable.
    unresolved=scf.uhf_stability(saddle,eri=args['eri'],max_cycle=1)
    assert unresolved.stable is None
    assert not unresolved.eigensolver_converged
    for dm,n in [(fixed.result.density_matrix_alpha,5),(fixed.result.density_matrix_beta,4)]:
        np.testing.assert_allclose(np.trace(dm@s),n,atol=1e-10,rtol=0)
    assert all(b <= a+1e-10 for a,b in zip(fixed.energy_history,fixed.energy_history[1:]))


def test_stability_empty_rotation_space_and_nonconvergence():
    args=dict(overlap=np.eye(1),hcore=-np.eye(1),eri=np.zeros((1,1,1,1)),
              nalpha=1,nbeta=0,nuclear_repulsion=0.)
    checked=scf.stabilize_uhf_from_integrals(**args)
    assert checked.stable and checked.restarts==0
    assert checked.minimum_curvature==float('inf')
    failed=scf.stabilize_uhf_from_integrals(**args,config=scf.UHFConfig(max_cycle=1))
    assert not failed.stable
    assert failed.restarts==0
    assert np.isnan(failed.minimum_curvature)


def test_uhf_curvature_matches_dense_energy_hessian():
    import jax
    import jax.numpy as jnp
    from gradscf.scf.orbital_optimization import _problem_functions
    h=jnp.array([[-1.,.1],[.1,.4]])
    eri=jnp.zeros((2,2,2,2))
    result=scf.run_uhf_from_integrals(overlap=jnp.eye(2),hcore=h,eri=eri,
                                     nalpha=1,nbeta=1,nuclear_repulsion=0.)
    checked=scf.uhf_stability(result,eri=eri)
    assert checked.stable
    dimension,_,_,_,energy=_problem_functions('uks','hf','col',((1.,0.),(1.,0.)))
    coeff=jnp.stack([result.mo_coeff_alpha,result.mo_coeff_beta])
    args=dict(hcore=h,eri=eri,nuclear_repulsion=jnp.array(0.),ao=jnp.zeros((0,2)),
              ao_deriv1=jnp.zeros((4,0,2)),grid_weights=jnp.zeros(0))
    dense=jax.hessian(energy)(jnp.zeros(dimension),coeff,args)
    np.testing.assert_allclose(checked.eigenvalues,np.linalg.eigvalsh(dense),atol=1e-10,rtol=0)
