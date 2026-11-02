"""Fixed-pool radial-space embedding for the nine-family diagnostic."""
from dataclasses import replace
import jax.numpy as jnp
import numpy as np

def lift(lower,coefficients,zeta):
    old_zeta=lower.num_channels//3
    if zeta not in (1,2,3) or old_zeta>zeta:
        raise ValueError('Only same-pool expansion to one, two or three valence channels is supported.')
    bound=lower.bind(coefficients);columns=[]
    for role,l,a,c in zip(lower.roles,lower.topology.angular_momenta,bound.exponents,bound.coefficients):
        if not role.startswith('valence_'):
            columns.append(c);continue
        a,c=np.asarray(a),np.asarray(c)
        if zeta>len(a):raise ValueError('More radial columns than primitives.')
        metric=(2*np.sqrt(a[:,None]*a[None,:])/(a[:,None]+a[None,:]))**(l+1.5)
        result=[np.array(x,copy=True) for x in c.T];orthogonal=[]
        for column in result:
            v=column.copy()
            for _ in range(2):
                for q in orthogonal:v-=q*(q@metric@v)
            norm=float(v@metric@v)
            if norm<=1e-10:raise ValueError('Existing radial columns are linearly dependent.')
            orthogonal.append(v/np.sqrt(norm))
        for index in np.argsort(a):
            if len(result)==zeta:break
            v=np.eye(len(a))[index]
            for _ in range(2):
                for q in orthogonal:v-=q*(q@metric@v)
            norm=float(v@metric@v)
            if norm>1e-10:
                orthogonal.append(v/np.sqrt(norm));result.append(v/np.linalg.norm(v))
        if len(result)!=zeta:raise ValueError('Cannot construct independent additional radial columns.')
        columns.append(jnp.asarray(np.column_stack(result),dtype=bound.centers.dtype))
    return replace(lower,
        topology=replace(lower.topology,contraction_counts=tuple(c.shape[1] for c in columns)),
        parameters=replace(bound,coefficients=tuple(columns)),
        slots=tuple((s//old_zeta)*zeta if s>=0 else s for s in lower.slots),num_channels=3*zeta)

def active_mask(layout):
    mask=np.zeros_like(np.asarray(layout.reference_outputs()),dtype=bool)
    for atom,slot,role,n,c in zip(layout.shell_atoms,layout.slots,layout.roles,
                                layout.topology.primitive_counts,layout.topology.contraction_counts):
        if slot>=0 and not (role.startswith('valence_') and n==c):
            mask[atom,slot:slot+c,:n]=True
    return mask

def normalize(coefficients,layout):
    result=np.array(coefficients,copy=True)
    for atom,slot,n,c in zip(layout.shell_atoms,layout.slots,
                            layout.topology.primitive_counts,layout.topology.contraction_counts):
        if slot>=0:
            for column in range(slot,slot+c):
                v=result[atom,column,:n];norm=np.linalg.norm(v)
                if not np.isfinite(norm) or norm<1e-12:raise ValueError('Invalid raw radial vector.')
                result[atom,column,:n]=v/norm
    return result
