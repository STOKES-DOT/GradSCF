"""Shared real-orbital HF/KS energy and Fock algebra.

XC adapters own functional evaluation. The potential here excludes the explicit
alpha*K term; extra_fock contains any additional potential supplied by an
adapter. Some legacy bound functionals include exact exchange in their returned
energy while returning separate semilocal potentials. They must set the explicit
energy flag below. This prevents counting their exchange energy twice without
changing the independently specified Fock contributions.
"""
from typing import NamedTuple, Any
import jax.numpy as jnp
from jax.lax import Precision


class XCContribution(NamedTuple):
    energy: Any
    potential: Any
    exact_exchange_fraction: Any
    extra_fock: Any = 0.
    energy_includes_exact_exchange: bool = False


def restricted_fock(hcore, coulomb, exchange, xc: XCContribution):
    return hcore+coulomb-.5*xc.exact_exchange_fraction*exchange+xc.potential+xc.extra_fock


def restricted_energy(density, hcore, coulomb, exchange, xc: XCContribution, *, nuclear_repulsion=0.):
    one=jnp.einsum('ij,ij->',density,hcore,precision=Precision.HIGHEST)
    j=.5*jnp.einsum('ij,ij->',density,coulomb,precision=Precision.HIGHEST)
    k=-.25*xc.exact_exchange_fraction*jnp.einsum('ij,ij->',density,exchange,precision=Precision.HIGHEST)
    return one+j+jnp.where(xc.energy_includes_exact_exchange,0.,k)+xc.energy+nuclear_repulsion


def unrestricted_fock(hcore, coulomb, exchange_spin, xc: XCContribution):
    return hcore+coulomb-xc.exact_exchange_fraction*exchange_spin+xc.potential+xc.extra_fock


def unrestricted_energy(density_spin, hcore, coulomb, exchange_spin, xc: XCContribution, *, nuclear_repulsion=0.):
    density=jnp.sum(density_spin,axis=0)
    one=jnp.einsum('ij,ij->',density,hcore,precision=Precision.HIGHEST)
    j=.5*jnp.einsum('ij,ij->',density,coulomb,precision=Precision.HIGHEST)
    k=-.5*xc.exact_exchange_fraction*jnp.einsum('sij,sij->',density_spin,exchange_spin,precision=Precision.HIGHEST)
    return one+j+jnp.where(xc.energy_includes_exact_exchange,0.,k)+xc.energy+nuclear_repulsion


__all__=['XCContribution','restricted_energy','restricted_fock','unrestricted_energy','unrestricted_fock']
