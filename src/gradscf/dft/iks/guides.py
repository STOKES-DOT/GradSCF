"""Fixed local guide construction using the shared density/potential contract."""
import jax.numpy as jnp

from gradscf.dft.derivatives import xc_energy_and_potential_from_density
from .assessment import energy_function
from .types import _check


def guide_fock(target, hcore, coulomb, *, xc='fermi_amaldi', state=None, params=None):
    """Return a fixed base Hamiltonian for local WY or ZMP inversion.

    xc=None means hcore only, matching the upstream no-guide convention.
    Fermi-Amaldi gives hcore+(1-1/N)*J[n_target]. A traditional semilocal
    recipe or external Functional gives hcore+J[n_target]+vxc[n_target], using
    the SAME scalar-AD potential as ordinary GradSCF evaluation. state supplies
    AO/grid features for that case; params remain live for model derivatives.

    Named hybrids are rejected instead of dropping exact exchange. External
    scalar models must themselves define a local multiplicative functional
    if a standard KS inversion is claimed; an arbitrary density-matrix energy
    does not establish that property. No neural architecture is imposed.
    """
    if target.layout not in ('rks','uks'):
        raise NotImplementedError('Standard local guides currently require RKS/UKS')
    h=jnp.asarray(hcore)
    n=target.overlap.shape[0]
    if h.shape!=(n,n) or jnp.iscomplexobj(h):
        raise ValueError('hcore must be a real spatial AO matrix')
    _check(jnp.all(jnp.isfinite(h)) & jnp.allclose(h,h.T), 'hcore must be finite and symmetric')
    base=h if target.layout=='rks' else jnp.stack((h,h))
    if xc is None:
        return base
    density=target.density if target.layout=='rks' else target.density.sum(axis=0)
    hartree=jnp.asarray(coulomb(density))
    if hartree.shape!=(n,n) or jnp.iscomplexobj(hartree):
        raise ValueError('Coulomb action must return a real spatial AO matrix')
    _check(jnp.all(jnp.isfinite(hartree)) & jnp.allclose(hartree,hartree.T),
           'Coulomb action must be finite and symmetric')
    if isinstance(xc,str) and xc=='fermi_amaldi':
        electrons=float(target.nelec) if target.layout=='rks' else sum(target.nelec)
        if electrons<=0:
            raise ValueError('Fermi-Amaldi guide requires a positive electron count')
        return base+(1-1/electrons)*hartree
    from gradscf.dft.functional import custom_xc
    from gradscf.dft.xc import TraditionalXCFunctional, hybrid_coeff, xc_type
    if isinstance(xc,str):
        xc=custom_xc(xc)
    if isinstance(xc,TraditionalXCFunctional):
        if float(hybrid_coeff(xc.spec))!=0:
            raise NotImplementedError('A hybrid/nonlocal guide needs a separately defined inverse-GKS problem')
        if xc_type(xc.spec).lower() not in ('lda','gga'):
            raise NotImplementedError('A meta-GGA guide requires an explicit generalized-KS operator contract')
    if state is None:
        raise ValueError('An XC guide requires the declared AO/grid state')
    overlap=jnp.asarray(state.overlap_matrix)
    if overlap.shape!=target.overlap.shape:
        raise ValueError('State and target AO dimensions differ')
    _check(jnp.allclose(overlap,target.overlap,atol=1e-12,rtol=1e-12),'State and target overlap differ')
    energy=energy_function(state,xc,layout=target.layout,component='xc',nuclear_repulsion=0.)
    value=xc_energy_and_potential_from_density(params,molecule=None,density=target.density,
        xc_energy_fn=lambda p,unused,d:energy(p,d))
    return base+hartree+value.vxc_matrix
