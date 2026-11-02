"""Experimental SZ->DZ embedding using exactly the same mixed primitive pool."""
from dataclasses import fields, replace
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.model.nnao.basis import DirectBasis

class ExpandedBasis(DirectBasis):
    """Artifact-local multi-column adapter for the frozen SZ-only checkout.

    Uses the current GradSCF DirectBasis column convention while leaving the
    original production checkout and its fixed-pool cache identity untouched.
    """
    def reference_outputs(self):
        out=jnp.zeros((len(self.symbols),self.num_channels,self.max_primitives),
                      dtype=self.parameters.centers.dtype)
        for atom,slot,c in zip(self.shell_atoms,self.slots,self.parameters.coefficients):
            if slot>=0:out=out.at[atom,slot:slot+c.shape[1],:len(c)].set(c.T)
        return out

    def bind(self,outputs,*,coords_bohr=None):
        raw=jnp.asarray(outputs,dtype=self.parameters.centers.dtype)
        if raw.shape!=(len(self.symbols),self.num_channels,self.max_primitives):
            raise ValueError('Wrong expanded contraction output shape.')
        columns=[]
        for atom,slot,n,fixed in zip(self.shell_atoms,self.slots,
                self.topology.primitive_counts,self.parameters.coefficients):
            if slot<0:
                columns.append(fixed);continue
            c=raw[atom,slot:slot+fixed.shape[1],:n].T
            norm=jnp.linalg.norm(c,axis=0)
            if not isinstance(norm,jax.core.Tracer) and (not np.isfinite(norm).all() or np.any(norm<1e-12)):
                raise ValueError('Zero or nonfinite expanded contraction.')
            columns.append(c/jnp.where(norm>=1e-12,norm,jnp.nan))
        coords=self.parameters.nuclear_coords if coords_bohr is None else jnp.asarray(coords_bohr,dtype=raw.dtype)
        if coords.shape!=(len(self.symbols),3):raise ValueError('Wrong coordinate shape.')
        return replace(self.parameters,coefficients=tuple(columns),nuclear_coords=coords,
                       centers=coords[jnp.asarray(self.shell_atoms)])

def second_column(a,c,l):
    """Diffuse-first Gaussian-metric Gram-Schmidt; no ordinary Euclidean PCA."""
    a,c=np.asarray(a),np.asarray(c)
    metric=(2*np.sqrt(a[:,None]*a[None,:])/(a[:,None]+a[None,:]))**(l+1.5)
    first=c/np.sqrt(c@metric@c)
    for index in np.argsort(a):
        v=np.eye(len(a))[index]
        for _ in range(2):v=v-first*(first@metric@v)
        norm=float(v@metric@v)
        if norm>1e-10:return v/np.linalg.norm(v)
    raise ValueError('No independent second radial column.')

def expand_valence_dz(sz, coefficients):
    if any(n!=1 for n in sz.topology.contraction_counts) or sz.num_channels!=3:
        raise ValueError('Expected single radial contraction in the original SZ layout.')
    bound=sz.bind(coefficients)
    columns=[]
    for role,l,a,c in zip(sz.roles,sz.topology.angular_momenta,bound.exponents,bound.coefficients):
        if role.startswith('valence_'):
            matrix=np.column_stack((np.asarray(c[:,0]),second_column(a,c[:,0],l)))
            columns.append(jnp.asarray(matrix,dtype=c.dtype))
        else:
            columns.append(c)
    topology=replace(sz.topology,contraction_counts=tuple(c.shape[1] for c in columns))
    values={f.name:getattr(sz,f.name) for f in fields(sz)}
    values.update(topology=topology,parameters=replace(bound,coefficients=tuple(columns)),
                  slots=tuple(2*s if s>=0 else s for s in sz.slots),num_channels=6)
    return ExpandedBasis(**values)
