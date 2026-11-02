"""Neutral main-group atomic populations for a bounded SCF initial-guess rescue."""
import numpy as np
import jax
import jax.numpy as jnp
from gradscf.solvers.eigen.spectral import inverse_sqrt

def atomic_guess_data(layout):
    """Return AO atom mask and spherical neutral-atom occupations.

    This is a population-based seed, not an isolated-atom SCF calculation.
    Fixed core shells are doubly occupied; valence s/p populations use their
    first radial column, distributed equally across angular components.
    """
    if layout.charge!=0 or layout.spin!=0 or any(e is not None for e in layout.ecps):
        raise ValueError('Atomic population rescue requires neutral all-electron singlet inputs.')
    owners=[];occupation=np.zeros(layout.topology.nao);shells=[];offset=0
    for atom,l,role,nc in zip(layout.shell_atoms,layout.topology.angular_momenta,layout.roles,layout.topology.contraction_counts):
        width=(2*l+1)*nc;owners.extend([atom]*width);shells.append((atom,l,role,offset,width));offset+=width
    for atom,z in enumerate(layout.topology.nuclear_charges):
        core=0;valence=[]
        for owner,l,role,start,width in shells:
            if owner!=atom:continue
            if role=='core':occupation[start:start+width]=2.;core+=2*width
            elif role.startswith('valence_'):valence.append((l,start))
        remaining=int(z)-core
        if remaining<0:raise ValueError('Fixed core has more electrons than the neutral atom.')
        for l,start in sorted(valence):
            if l not in (0,1):raise ValueError('Atomic population rescue currently supports main-group s/p valence.')
            electrons=min(remaining,2*(2*l+1));occupation[start:start+2*l+1]=electrons/(2*l+1);remaining-=electrons
        if remaining:raise ValueError('Valence layout cannot represent the neutral atomic population.')
    owners=np.asarray(owners)
    return owners[:,None]==owners[None,:],occupation

def atomic_density_guess(overlap,atom_mask,occupation):
    """Atomic-block Lowdin seed; implicit SCF response is independent of its seed."""
    s=jax.lax.stop_gradient(overlap)
    x=inverse_sqrt(jnp.where(atom_mask,s,0.),1e-10)
    density=(x*occupation)@x.T
    return jax.lax.stop_gradient(jnp.where(atom_mask,density,0.))
