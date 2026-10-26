"""Density response is invariant to equally occupied degenerate orbital gauges."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers.eigen.spectral import occupied_projector


@pytest.mark.parametrize('splitting',[0.,1e-8,2e-6])
def test_spectral_density_degenerate_response_and_higher_derivative(splitting):
    rng=np.random.default_rng(19)
    q=jnp.asarray(np.linalg.qr(rng.normal(size=(4,4)))[0])
    eigenvalues=jnp.array([-2.,-2.+splitting,1.,1.+splitting])
    matrix=(q*eigenvalues)@q.T
    spectral_density=lambda a:2*occupied_projector(a,2)
    direction=jnp.asarray(rng.normal(size=(4,4)));direction=(direction+direction.T)/2
    probe=jnp.asarray(rng.normal(size=(4,4)));probe=(probe+probe.T)/2
    density,tangent=jax.jit(lambda a,b:jax.jvp(spectral_density,
                                             (a,),(b,)))(matrix,direction)
    np.testing.assert_allclose(density,2*q[:,:2]@q[:,:2].T,atol=1e-13)
    cross=q[:,:2].T@direction@q[:,2:]
    cross=2*cross/(eigenvalues[:2,None]-eigenvalues[None,2:])
    expected=q[:,:2]@cross@q[:,2:].T
    np.testing.assert_allclose(tangent,expected+expected.T,atol=2e-12,rtol=2e-12)
    objective=lambda a:jnp.sum(spectral_density(a)*probe)
    gradient=jax.grad(objective)(matrix)
    np.testing.assert_allclose(jnp.sum(gradient*direction),jnp.sum(tangent*probe),atol=2e-12)
    hvp=jax.jvp(jax.grad(objective),(matrix,),(direction,))[1]
    step=1e-4
    finite_difference=(jax.grad(objective)(matrix+step*direction)-jax.grad(objective)(matrix-step*direction))/(2*step)
    np.testing.assert_allclose(hvp,finite_difference,atol=2e-7,rtol=2e-6)


@pytest.mark.parametrize('nocc',[0,3])
def test_empty_and_full_projector_response(nocc):
    matrix=jnp.diag(jnp.array([-2.,-2.,1.]))
    derivative=jax.jvp(lambda a:occupied_projector(a,nocc),(matrix,),(jnp.ones_like(matrix),))[1]
    np.testing.assert_allclose(derivative,0.,atol=1e-14)
