"""Shared physical assembly and explicit exact-exchange accounting."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('alpha',[0.,.25,1.])
def test_restricted_unrestricted_closed_shell_and_density_derivative(alpha):
    from gradscf.scf.energy import XCContribution, restricted_energy, restricted_fock, unrestricted_energy, unrestricted_fock
    h=jnp.array([[-1.,.1],[.1,.3]])
    factors=jnp.array([[[.4,.12],[.12,.3]],[[.1,.02],[.02,.2]]])
    eri=jnp.einsum('apq,ars->pqrs',factors,factors)
    d=jnp.array([[1.5,.2],[.2,.5]])
    def jk(d):
        return jnp.einsum('pqrs,rs->pq',eri,d),jnp.einsum('prqs,rs->pq',eri,d)
    def terms(d):
        return XCContribution(.07*jnp.sum(d*d),.14*d,alpha)
    j,k=jk(d);xc=terms(d)
    energy=restricted_energy(d,h,j,k,xc,nuclear_repulsion=.7)
    fock=restricted_fock(h,j,k,xc)
    expected=jax.grad(lambda density:restricted_energy(density,h,*jk(density),terms(density),nuclear_repulsion=.7))(d)
    np.testing.assert_allclose(fock,expected,atol=1e-12,rtol=0)
    spin=jnp.stack([d/2,d/2]);ks=jnp.stack([k/2,k/2])
    spin_xc=XCContribution(xc.energy,jnp.stack([xc.potential]*2),alpha)
    np.testing.assert_allclose(unrestricted_energy(spin,h,j,ks,spin_xc,nuclear_repulsion=.7),energy,atol=1e-12,rtol=0)
    np.testing.assert_allclose(unrestricted_fock(h,j,ks,spin_xc),jnp.stack([fock]*2),atol=1e-12,rtol=0)


def test_embedded_exchange_energy_is_counted_once_and_extra_fock_is_preserved():
    from gradscf.scf.energy import XCContribution, unrestricted_energy, unrestricted_fock
    d=jnp.array([[[.7,.1],[.1,.3]],[[.4,-.1],[-.1,.2]]])
    h=jnp.diag(jnp.array([-1.,.3]));j=jnp.eye(2)*.4;k=d*.2
    alpha=.35;exc=.17;potential=jnp.ones((2,2,2))*.03;extra=jnp.ones_like(potential)*.02
    exchange=-.5*alpha*jnp.einsum('spq,spq->',d,k)
    separate=XCContribution(exc,potential,alpha,extra)
    embedded=XCContribution(exc+exchange,potential,alpha,extra,energy_includes_exact_exchange=True)
    np.testing.assert_allclose(unrestricted_energy(d,h,j,k,separate),unrestricted_energy(d,h,j,k,embedded),atol=1e-14,rtol=0)
    np.testing.assert_allclose(unrestricted_fock(h,j,k,embedded),h+j-alpha*k+potential+extra,atol=1e-14,rtol=0)
