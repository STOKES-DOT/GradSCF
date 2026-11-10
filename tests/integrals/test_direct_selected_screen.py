"""TDD: single-product screening must not retain exchange-only quartets for J.

The sparse density makes discarded contractions exactly zero in this fixture.
Call counts require the opt-in diagnostic library, never clean timing.
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.integrals._native.diagnostics import get_counts, reset_counts


@pytest.mark.skipif(os.getenv('GRADSCF_EXPECT_DIAGNOSTICS') != '1', reason='requires --diagnostics native build')
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('omega', [0., .3])
def test_selected_product_avoids_irrelevant_integral_calls(workers, omega, record_property):
    top, pars = integrals.prepare_basis('H 0 0 0; H 0 0 1; H 0 1 0; H 1 0 0',
        [[0, [1., 1.]]], unit='Bohr')
    plan = integrals.make_plan(top)
    basis = NativeDirectBasis(plan, pars)
    tensor = plan.evaluate('eri', pars, omega=omega)
    d = jnp.zeros((4, 4)).at[0, 1].set(.7)  # deliberately nonsymmetric
    expected = (jnp.einsum('pqrs,rs->pq', tensor, d), jnp.einsum('prqs,rs->pq', tensor, d))
    actions = {(j,k): jax.jit(lambda x,j=j,k=k: basis.get_jk(x,
        screen_density=d, screening_threshold=1e-12, workers=workers,
        omega=omega, with_j=j, with_k=k)) for j,k in ((True,True),(True,False),(False,True))}
    for f in actions.values():jax.block_until_ready(f(d))
    counts={}
    for (j,k), f in actions.items():
        reset_counts();result=jax.block_until_ready(f(d));info=get_counts()
        assert info['enabled'], 'A skipped/production counter cannot satisfy the TDD count gate'
        counts[j,k]=info['families']['eri4c']['evaluations']
        record_property(f'calls_J{int(j)}K{int(k)}', counts[j,k])
        record_property('native_sha256', info['library_sha256'])
        for index,enabled in enumerate((j,k)):
            np.testing.assert_allclose(result[index], expected[index] if enabled else 0.,atol=1e-12,rtol=1e-12)
    print('selected intor calls',workers,omega,counts,flush=True)
    assert counts[True,False] < counts[True,True], f'J-only retained K-only quartet work: {counts}'
    assert counts[False,True] < counts[True,True], f'K-only retained J-only quartet work: {counts}'


@pytest.mark.skipif(os.getenv('GRADSCF_TEST_GPU') != '1', reason='opt-in allocated CUDA device')
@pytest.mark.parametrize('selection', [(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('omega', [0., .3])
def test_selected_screen_cuda_value_jvp_vjp(selection, omega):
    from test_direct_density_screen import mask_for
    cpu,gpu=jax.devices('cpu')[0],jax.devices('gpu')[0]
    with jax.default_device(cpu):
        top,pars=integrals.prepare_basis('H 0 0 0; H 0 0 1; H 0 1 0; H 1 0 0',[[0,[1.,1.]]],unit='Bohr')
        plan=integrals.make_plan(top);basis=NativeDirectBasis(plan,pars)
        eri=np.asarray(plan.evaluate('eri',pars,omega=omega))
        screen=jnp.zeros((4,4)).at[0,1].set(.7)
        mask=mask_for(eri,screen,np.arange(5),cutoff=.05,with_j=selection[0],with_k=selection[1])
        masked=jnp.asarray(eri*mask)
        density=jnp.asarray(np.random.default_rng(416).normal(size=(2,4,4)))
        direction=jnp.sin(density)
        def dense(d):
            return (jnp.einsum('pqrs,...rs->...pq',masked,d) if selection[0] else jnp.zeros_like(d),
                    jnp.einsum('prqs,...rs->...pq',masked,d) if selection[1] else jnp.zeros_like(d))
        value,tangent=jax.jvp(dense,(density,),(direction,))
        expected_grad=jax.grad(lambda d:sum(jnp.sum(x*x) for x in dense(d)))(density)
    with jax.default_device(gpu):
        bound,dm,ds,ref=jax.device_put((basis,density,direction,screen),gpu)
        action=lambda d:bound.get_jk(d,screen_density=ref,screening_threshold=.05,workers=2,
                                     omega=omega,with_j=selection[0],with_k=selection[1])
        actual=jax.jit(lambda d,v:jax.jvp(action,(d,),(v,)))(dm,ds)
        gradient=jax.jit(jax.grad(lambda d:sum(jnp.sum(x*x) for x in action(d))))(dm)
    for a,b in zip(jax.tree.leaves((actual,gradient)),jax.tree.leaves(((value,tangent),expected_grad))):
        assert a.devices()=={gpu}
        np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=1e-9,rtol=1e-9)
