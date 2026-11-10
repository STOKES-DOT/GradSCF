"""Guides reuse shared scalar-functional potentials, with live parameters."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import dft
from gradscf.dft import iks
from test_hybrid_energy import one_ao_state


@pytest.mark.parametrize('layout',['rks','uks'])
def test_fermi_amaldi_guide_uses_total_density_and_electron_count(layout):
    density=jnp.array([[2.]]) if layout=='rks' else jnp.ones((2,1,1))
    target=iks.Target(density,jnp.eye(1),2 if layout=='rks' else (1,1),layout=layout)
    base=iks.guide_fock(target,jnp.array([[-1.]]),lambda d:.8*d)
    np.testing.assert_allclose(base,-.2,atol=1e-14)
    np.testing.assert_allclose(iks.guide_fock(target,jnp.array([[-1.]]),lambda d:.8*d,xc=None),-1.)


def test_external_functional_guide_retains_parameter_and_density_derivatives():
    state=one_ao_state();target=iks.Target(jnp.array([[2.]]),jnp.eye(1),2)
    functional=dft.custom_xc(input_fn=lambda fields:fields.density_matrix,
        energy_fn=lambda p,d:p*jnp.sum(d*d))
    base=lambda p:iks.guide_fock(target,jnp.array([[-1.]]),lambda d:.8*d,
        xc=functional,state=state,params=p)[0,0]
    np.testing.assert_allclose(base(.3),1.2,atol=1e-13)
    np.testing.assert_allclose(jax.grad(base)(.3),2.,atol=1e-13)


def test_named_hybrid_is_not_silently_used_as_local_xc_guide():
    target=iks.Target(jnp.array([[2.]]),jnp.eye(1),2)
    with pytest.raises(NotImplementedError,match='hybrid|nonlocal'):
        iks.guide_fock(target,jnp.array([[-1.]]),lambda d:.8*d,xc='pbe0',state=one_ao_state())


@pytest.mark.parametrize('xc',[dft.custom_xc('pbe0'),'r2scan'])
def test_named_nonlocal_operators_are_rejected_for_recipe_objects_too(xc):
    target=iks.Target(jnp.array([[2.]]),jnp.eye(1),2)
    with pytest.raises(NotImplementedError,match='hybrid|nonlocal|meta-GGA'):
        iks.guide_fock(target,jnp.array([[-1.]]),lambda d:.8*d,xc=xc,state=one_ao_state())
