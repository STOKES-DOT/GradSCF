"""Lazy fixed-target inputs; orbital provenance is separate from the density."""
from types import SimpleNamespace

import jax.numpy as jnp

from gradscf.dft.functional import DensityInputs


AVAILABLE_INPUTS=frozenset(('n','sigma','grad','s','u','lap','q','z','tau','alpha',
                           'rho_occ','tau_occ','virtual_0','virtual_1'))
ORBITAL_WINDOWS=(.2,1.)  # Hartree, fixed before the search starts.


def prepare_case_orbitals(case):
    """Validate frozen reference inputs before any model proposal is scored."""
    if not hasattr(case,'orbital_reference'):
        from gradscf.dft.iks.orbital_features import prepare_orbital_reference
        case.orbital_reference=prepare_orbital_reference(case.target,
            case.problem.base_fock+case.vxc,method='PBE@fixed-target-density')
    return case.orbital_reference


def features_for_candidate(case,names):
    """Return scalar grid leaves; no target energy or inverse potential is read."""
    names=set(names)
    if not names<=AVAILABLE_INPUTS:
        raise ValueError('Unknown local-potential inputs: '+', '.join(sorted(names-AVAILABLE_INPUTS)))
    if not hasattr(case,'_open_inputs'):
        a=case.arrays
        physical=SimpleNamespace(rho=jnp.asarray(a['rho']),grad_rho=jnp.asarray(a['grad_rho']),
                                 laplacian_rho=jnp.asarray(a['laplacian_rho']),weights=jnp.asarray(a['grid_weights']))
        case._open_inputs=case.inputs(physical,case.fields)
    cache=case._open_inputs
    if names & {'tau','alpha'} and 'tau' not in cache:
        cartesian=case.transform@case.target.density@case.transform.T
        physical=DensityInputs(jnp.stack((.5*cartesian,.5*cartesian)),case.context)
        cache['tau']=physical.tau
        uniform=.3*(3*jnp.pi**2)**(2/3)*cache['n']**(5/3)
        cache['alpha']=(cache['tau']-cache['sigma']/(8*cache['n']))/uniform
    if names & {'rho_occ','tau_occ','virtual_0','virtual_1'} and 'rho_occ' not in cache:
        from gradscf.dft.iks.orbital_features import orbital_grid_features
        cache.update(orbital_grid_features(prepare_case_orbitals(case),case.grid,
            transform=case.transform,windows=ORBITAL_WINDOWS,max_memory=8000.))
    return {name:cache[name] for name in names}
