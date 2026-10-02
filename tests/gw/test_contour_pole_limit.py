"""Analytic one-screening-pole checks at the Green-function contour boundary."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.gw.freq import scaled_legendre_grid
from gradscf.gw.self_energy import sigma_cd


@pytest.mark.parametrize('explicit_occupation',[True,False])
@pytest.mark.parametrize('occupied',[True,False])
def test_own_pole_value_and_frequency_derivative(occupied,explicit_occupation):
    freq,weight=scaled_legendre_grid(80)
    gap=1.; transition=.2; vertex=.5
    coupling=4*transition**2*gap
    excitation=np.sqrt(gap**2+coupling)
    spectral_weight=vertex**2*coupling/(2*excitation)
    energy=-.3 if occupied else .3
    ctx=dict(mo_energy=jnp.array([energy]),ef=0.,eta=0.,freqs=freq,wts=weight,
             wmn_p=(-vertex**2*coupling/(freq**2+excitation**2))[:,None],
             wmn_static=jnp.array([-vertex**2*coupling/excitation**2]),
             b_pm=jnp.array([[vertex]]),b_mp=jnp.array([[vertex]]),
             channels=((jnp.array([-.5,.5]),jnp.array([[[transition]]]),2.),))
    if explicit_occupation:
        ctx['ef']=energy  # Empty/full spin channels need no artificial gap.
        ctx['occupation_sign']=jnp.array([-1. if occupied else 1.])
    fun=lambda w:jnp.real(sigma_cd(w,ctx))
    sign=1 if occupied else -1
    value,slope=jax.jit(jax.value_and_grad(fun))(energy)
    np.testing.assert_allclose(value,sign*spectral_weight/excitation,atol=1e-12)
    np.testing.assert_allclose(slope,-spectral_weight/excitation**2,atol=1e-11)
    for delta in (-.03,.03):
        np.testing.assert_allclose(fun(energy+delta),spectral_weight/(delta+sign*excitation),atol=2e-10)


@pytest.mark.parametrize('occupied',[True,False])
def test_own_pole_parameter_response_matches_spectral_model(occupied):
    freq,weight=scaled_legendre_grid(80)
    transition=.2
    coupling=4*transition**2
    excitation=jnp.sqrt(1+coupling)
    energy=-.3 if occupied else .3
    def value(t):
        vertex=.5*(1+.1*t)
        pole=energy+.03*t
        ctx=dict(mo_energy=jnp.array([pole]),ef=0.,eta=0.,freqs=freq,wts=weight,
            wmn_p=(-vertex**2*coupling/(freq**2+excitation**2))[:,None],
            wmn_static=jnp.array([-vertex**2*coupling/excitation**2]),
            b_pm=jnp.array([[vertex]]),b_mp=jnp.array([[vertex]]),
            channels=((jnp.array([-.5,.5]),jnp.array([[[transition]]]),2.),))
        return jnp.real(sigma_cd(pole,ctx))
    expected=(1 if occupied else -1)*.1*.5**2*coupling/excitation**2
    np.testing.assert_allclose(jax.jit(jax.grad(value))(0.),expected,atol=1e-12)
    np.testing.assert_allclose(jax.jvp(value,(0.,),(1.,))[1],expected,atol=1e-12)
