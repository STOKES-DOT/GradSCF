"""Budgeted MO grid fields are reusable; OV transition tensors are not cached."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf.tddft.response_options import plan_response_memory
from gradscf.tddft.response import _restricted_grid_xc_response
from test_td_execution_reuse import reference


def test_grid_orbital_cache_is_charged_inside_action_budget():
    plan=plan_response_memory(25,100,5,jnp.float64,128.,grid_block_size=8192,orbital_grid_bytes=1000000)
    assert plan.orbital_grid_cache_bytes==1000000
    limited=plan_response_memory(25,100,5,jnp.float64,128.,grid_block_size=8192,orbital_grid_bytes=100000000)
    assert limited.orbital_grid_cache_bytes==0



def test_cached_orbital_grid_action_and_derivatives_match_block_recomputation():
    molecule=reference()
    co=molecule.mo_coeff[0,:,:2];cv=molecule.mo_coeff[0,:,2:]
    hessian=jnp.asarray(np.cos(np.arange(112)).reshape(4,4,7))*.01
    hessian=.5*(hessian+hessian.swapaxes(0,1))
    x=jnp.asarray(np.sin(np.arange(18)).reshape(3,2,3))
    def result(scale,cache):
        action,_=_restricted_grid_xc_response(molecule,co*scale,cv,hessian,
            feature_kind='GGA',dtype=x.dtype,block_size=3,compute_diagonal=False,
            cache_orbitals=cache)
        return action(x)
    before=jax.jit(lambda a:result(a,False))
    after=jax.jit(lambda a:result(a,True))
    point=jnp.asarray(1.)
    np.testing.assert_allclose(before(point),after(point),atol=3e-12,rtol=3e-12)
    ga=jax.jit(jax.grad(lambda a:jnp.sum(after(a)**2)))(point)
    gb=jax.jit(jax.grad(lambda a:jnp.sum(before(a)**2)))(point)
    np.testing.assert_allclose(ga,gb,atol=2e-11,rtol=2e-11)


def test_streamed_prepared_orbitals_do_not_reenter_ao_provider(monkeypatch):
    from test_streamed_response_grid import _reference,_orbitals
    from gradscf.integrals.grids.blocks import AOGrid
    molecule=_reference(jnp.asarray(.04),streamed=True)
    co,cv=_orbitals()
    hessian=jnp.broadcast_to(jnp.eye(4)[:,:,None],(4,4,7))*.01
    action,_=_restricted_grid_xc_response(molecule,co,cv,hessian,
        feature_kind='GGA',dtype=jnp.float64,block_size=3,compute_diagonal=False,
        cache_orbitals=True)
    def unexpected(*args,**kwargs):
        raise AssertionError('prepared action must not reevaluate streamed AO')
    monkeypatch.setattr(AOGrid,'block',unexpected)
    result=jax.jit(action)(jnp.ones((2,1,2)))
    assert bool(jnp.all(jnp.isfinite(result)))


def test_streamed_cached_orbitals_keep_geometry_jvp():
    from test_streamed_response_grid import _reference,_orbitals
    co,cv=_orbitals();x=jnp.asarray([[[.7,-.2]],[[.3,.8]]])
    def action(parameter,cache):
        molecule=_reference(parameter,streamed=True)
        hessian=jnp.broadcast_to(jnp.eye(4)[:,:,None],(4,4,7))*.01
        fn,_=_restricted_grid_xc_response(molecule,co,cv,hessian*molecule.grid.weights,
            feature_kind='GGA',dtype=jnp.float64,block_size=3,compute_diagonal=False,
            cache_orbitals=cache)
        return fn(x)
    before=lambda p:action(p,False);after=lambda p:action(p,True)
    point=jnp.asarray(.04);direction=jnp.asarray(.13)
    for a,b in zip(jax.jvp(before,(point,),(direction,)),jax.jvp(after,(point,),(direction,))):
        np.testing.assert_allclose(a,b,atol=2e-11,rtol=2e-11)
