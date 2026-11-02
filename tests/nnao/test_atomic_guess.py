import numpy as np
import jax.numpy as jnp
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.training.initial_guess import atomic_guess_data,atomic_density_guess

def test_neutral_atomic_guess_has_exact_charge_and_no_interatomic_density():
    layout=prepare_direct_basis([('Br',(0,0,0)),('Br',(0,0,2.3)),('O',(0,0,4.9)),
        ('P',(0,0,6.4)),('H',(1,0,6.4)),('H',(-.5,1,6.4)),('H',(-.5,-1,6.4))],
        basis_family='nnao_szp_l',core_primitives='6-31g')
    mask,occ=atomic_guess_data(layout);n=layout.topology.nao
    assert np.sum(occ)==96 and np.count_nonzero(np.isclose(occ,5/3))==6
    rng=np.random.default_rng(41);a=rng.normal(size=(n,n))*.01
    s=jnp.asarray(np.eye(n)+a@a.T)
    d=np.asarray(atomic_density_guess(s,jnp.asarray(mask),jnp.asarray(occ)))
    np.testing.assert_allclose(np.trace(d@np.asarray(s)),96,atol=1e-10)
    np.testing.assert_allclose(d[~mask],0,atol=1e-13)
    assert np.linalg.eigvalsh(d).min()>-1e-12

def test_partial_atomic_population_is_only_an_initial_guess():
    layout=prepare_direct_basis('H 0 0 0; H 0 0 .74',basis_family='nnao_szp_l')
    mask,occ=atomic_guess_data(layout)
    assert sorted(occ[occ>0].tolist())==[1.,1.]
    s=jnp.eye(layout.topology.nao)
    d=atomic_density_guess(s,jnp.asarray(mask),jnp.asarray(occ))
    np.testing.assert_allclose(jnp.trace(d@s),2.)
