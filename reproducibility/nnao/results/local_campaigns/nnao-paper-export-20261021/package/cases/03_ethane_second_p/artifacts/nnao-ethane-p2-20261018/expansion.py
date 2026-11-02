"""Same-pool expansion of carbon p radial space for a bounded diagnostic."""
from dataclasses import replace
import jax.numpy as jnp
import numpy as np
from gradscf.model.nnao.initialization import radial_reference_coefficients


def expand_carbon_p(lower,coefficients):
    """Retain every original AO and add one carbon p radial column.

    Exponents, core blocks and H shells remain unchanged. Additional columns
    use the existing Gaussian-metric initialization, with diffuse primitives
    considered first. No MACE head or production basis family is changed.
    """
    bound=lower.bind(coefficients)
    columns=[];slots=[];cursor=[0]*len(lower.symbols);added=0
    for owner,ell,role,slot,alpha,c in zip(lower.shell_atoms,lower.topology.angular_momenta,
            lower.roles,lower.slots,bound.exponents,bound.coefficients):
        if lower.symbols[owner]=='C' and role=='valence_p':
            if ell!=1 or c.shape[1]!=1:
                raise ValueError('Carbon p expansion requires an original single radial column.')
            shell=dict(l=ell,exponents=np.asarray(alpha),coefficients=np.asarray(c[:,0]))
            c=jnp.asarray(radial_reference_coefficients(shell,2),dtype=bound.centers.dtype)
            added+=1
        columns.append(c)
        if slot<0:
            slots.append(-1)
        else:
            slots.append(cursor[owner]);cursor[owner]+=c.shape[1]
    if not added:
        raise ValueError('No carbon valence p shell was found.')
    return replace(lower,
        topology=replace(lower.topology,contraction_counts=tuple(c.shape[1] for c in columns)),
        parameters=replace(bound,coefficients=tuple(columns)),slots=tuple(slots),
        num_channels=max(cursor))
