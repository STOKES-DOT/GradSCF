"""DifferentiableSCF accepts current XC contracts, not historical fallbacks."""
from types import SimpleNamespace
import jax.numpy as jnp
import pytest
from gradscf.scf import differentiable as scf


def test_scf_binding_requires_scf_specific_entry():
    for functional in (SimpleNamespace(bind=lambda p:object()),
                       SimpleNamespace(bind_to_molecule=lambda p,m:object())):
        with pytest.raises(TypeError,match='bind_to_molecule_for_scf'):
            scf._resolved_xc_object({},functional,object())


@pytest.mark.parametrize('length',[4,6])
def test_restricted_potential_requires_complete_components(length):
    rho=jnp.ones(2);grad=jnp.zeros((2,3))
    values=(rho,grad,'LDA',0.) if length==4 else (rho,grad,rho,rho,'MGGA',0.)
    functional=SimpleNamespace(scf_potential_components_and_alpha=lambda p,m:values)
    with pytest.raises(ValueError,match='7'):
        scf._scf_xc_components({},functional,SimpleNamespace(ao=jnp.ones((2,1))),functional_dtype=jnp.float64)


def test_bound_grid_requires_components_not_local_potential():
    with pytest.raises(TypeError,match='grid_potential_components'):
        scf._grid_xc_potential_components_from_resolved(
            SimpleNamespace(local_potential=lambda rho:rho),molecule=SimpleNamespace())


@pytest.mark.parametrize('length',[3,4])
def test_current_scf_bound_grid_contract_is_retained(length):
    rho=jnp.ones(2);grad=jnp.zeros((2,3));tau=2*rho;lapl=3*rho
    bound=SimpleNamespace(grid_potential_components=lambda m:(rho,grad,tau,lapl)[:length],
                          response_feature_kind='MGGA',exact_exchange_fraction=.25)
    functional=SimpleNamespace(bind_to_molecule_for_scf=lambda p,m:bound)
    values=scf._scf_xc_components({},functional,SimpleNamespace(ao=jnp.ones((2,1))),
                                  functional_dtype=jnp.float64)
    assert len(values)==7
    assert jnp.all(values[2]==tau)
    assert jnp.all(values[3]==(lapl if length==4 else 0))
    assert values[5]==.25
