"""Sole Gaussian/Coulomb interface and live Cartesian AO parameter binding.

IntegralPlan normalizes raw shell coefficients. CartesianBasis already defines
AO linear combinations, which must not be renormalized a second time.
"""
from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp
import numpy as np
from .basis.types import BasisTopology, BasisParameters
from .basis.cartesian import cartesian_angular_tuples
from .basis.normalization import radial_primitive_norm
from .plan import make_plan


def static_zeta(zeta):
    if isinstance(zeta,jax.core.Tracer):
        raise NotImplementedError('libcint Rinv zeta is static; zeta differentiation is not implemented.')
    if zeta is None or np.isposinf(float(zeta)):
        return 0.,False
    value=float(zeta)
    if not np.isfinite(value) or value<0:
        raise ValueError('Rinv zeta must be nonnegative or None for point Coulomb.')
    return value,value==0.


@dataclass(frozen=True)
class CartesianBinding:
    plan: object
    parameters: BasisParameters
    selection: tuple[int,...]
    nuclear_weights: object
    integer_nuclear_weights: bool

    @property
    def complete(self):
        return tuple(sorted(self.selection))==tuple(range(self.plan.topology.nao))

    @property
    def ordered(self):
        return self.selection==tuple(range(self.plan.topology.nao))

    def _operands(self,separate_parameters):
        p=self.parameters;top=self.plan.topology
        weighted=tuple(c*radial_primitive_norm(l,a)[:,None]
                       for l,a,c in zip(top.angular_momenta,p.exponents,p.coefficients))
        fixed=replace(p,centers=jnp.zeros_like(p.centers),nuclear_coords=jnp.zeros_like(p.nuclear_coords))
        if separate_parameters:
            fixed=replace(fixed,exponents=tuple(jnp.ones_like(a) for a in p.exponents))
            packed=tuple(jnp.zeros_like(c) for c in weighted)
        else:packed=weighted
        atm,bas,env=self.plan._pack(fixed,packed)
        coords=jnp.concatenate((p.nuclear_coords,p.centers))
        return atm,bas,env,coords,jnp.concatenate(tuple(c.T.ravel() for c in weighted)),jnp.concatenate(p.exponents)

    def evaluate(self,operator,*,origin=None,zeta=None,aosym='s1',shell_slice=None):
        if operator == 'nuclear' and not self.integer_nuclear_weights:
            weights = self.nuclear_weights
            if not isinstance(weights, jax.core.Tracer) and not np.isfinite(np.asarray(weights)).all():
                raise ValueError('Nuclear weights must be finite.')
            if not len(weights):
                return jnp.zeros((len(self.selection),)*2, dtype=jnp.float64)
            potentials = jax.lax.map(lambda point: self.evaluate('rinv', origin=point),
                                     self.parameters.nuclear_coords)
            return -jnp.einsum('a,aij->ij', weights, potentials)
        if operator=='eri':
            from .backends.native.autodiff.geometry import evaluate_geometry
            if aosym=='s1':
                from .backends.native.autodiff.eri_basis import evaluate_eri_basis
                atm,bas,env,coords,coefficients,exponents=self._operands(True)
                value=evaluate_eri_basis(atm,bas,env,coefficients,coords,jnp.zeros(3),
                    self.plan.topology.nao,cart=True,exponents=exponents,shell_slice=shell_slice)
            else:
                atm,bas,env,coords,_,_=self._operands(False)
                value=evaluate_geometry('eri',atm,bas,env,coords,jnp.zeros(3),
                    self.plan.topology.nao,cart=True,shell_slice=shell_slice,
                    compact_layout=None if aosym=='s1' else 0 if aosym=='s4' else 1)
            if aosym!='s1':
                if self.ordered:return value
                i,j=np.tril_indices(len(self.selection));ids=np.asarray(self.selection)
                a,b=ids[i],ids[j];hi=np.maximum(a,b);lo=np.minimum(a,b)
                pairs=hi*(hi+1)//2+lo
                if aosym=='s4':return value[pairs[:,None],pairs[None,:]]
                i,j=np.tril_indices(len(pairs));a,b=pairs[i],pairs[j]
                hi=np.maximum(a,b);lo=np.minimum(a,b)
                return value[hi*(hi+1)//2+lo]
            if shell_slice is not None or self.ordered:return value
            ids=jnp.asarray(self.selection)
            return value[jnp.ix_(ids,ids,ids,ids)]
        if origin is None:
            if operator=='dipole':
                charges=self.nuclear_weights
                if not len(charges):
                    raise ValueError('Dipole without explicit origin requires nonzero nuclear charges.')
                origin=charges@self.parameters.nuclear_coords/jnp.sum(charges)
            else:origin=jnp.zeros(3)
        from .backends.native.autodiff.evaluation import evaluate_differentiable
        atm,bas,env,coords,coefficients,exponents=self._operands(True)
        if operator=='rinv':
            width,zero=static_zeta(zeta)
            if zero:return jnp.zeros((len(self.selection),)*2,dtype=jnp.float64)
            env=env.at[7].set(width)
        value=evaluate_differentiable(operator,atm,bas,env,coefficients,coords,
            jnp.asarray(origin,dtype=jnp.float64),self.plan.topology.nao,cart=True,exponents=exponents)
        if self.ordered:return value
        ids=jnp.asarray(self.selection)
        return value[...,ids[:,None],ids[None,:]]

    def direct(self):
        """Project shell-direct primitive J/K without a primitive four-tensor."""
        from .basis.contraction import primitive_basis
        from .backends.native.jk import NativeDirectBasis,ProjectedNativeDirectBasis
        top=self.plan.topology;p=self.parameters
        primitive,parameters=primitive_basis(top,p)
        angular=tuple((l+1)*(l+2)//2 for l in top.angular_momenta)
        rows=sum(n*m for n,m in zip(top.primitive_counts,angular))
        transform=jnp.zeros((rows,top.nao),dtype=jnp.float64)
        start=stop=0
        for m,c in zip(angular,p.coefficients):
            block=jnp.kron(c,jnp.eye(m))
            transform=transform.at[start:start+block.shape[0],stop:stop+block.shape[1]].set(block)
            start+=block.shape[0];stop+=block.shape[1]
        return ProjectedNativeDirectBasis(NativeDirectBasis(make_plan(primitive),parameters),
                                         transform[:,jnp.asarray(self.selection)])


def bind_cartesian(basis,*,atom_coords=None,atom_charges=None):
    """Bind current AO leaves; shell metadata may only describe shared objects.

    Equal numerical arrays need not be the same differentiable parameter. Keep
    the fast shared-shell representation only when each AO references those
    exact leaves. Real nuclear weights are separate from libcint's integer ATM
    metadata, because overlap/kinetic/ERI do not depend on nuclear charges.
    """
    angular=[];exponents=[];coefficients=[];centers=[];selection=[]
    def append(ao):
        offset = sum((l+1)*(l+2)//2 for l in angular)
        l = sum(ao.angular)
        angular.append(l)
        exponents.append(jnp.asarray(ao.exponents))
        coefficients.append(jnp.asarray(ao.coefficients)[:,None])
        centers.append(jnp.asarray(ao.center))
        return offset+cartesian_angular_tuples(l).index(ao.angular)
    if basis.shells:
        selection=[None]*basis.nao;offset=0
        for shell in basis.shells:
            l=sum(shell.angulars[0]);components=cartesian_angular_tuples(l)
            indices=np.asarray(shell.ao_indices)
            if indices.shape!=(len(shell.angulars),):raise ValueError('Shell AO index shape does not match its angular components.')
            for index,component in zip(indices,shell.angulars):
                if int(index)!=index or not 0<=int(index)<basis.nao or selection[int(index)] is not None:
                    raise ValueError('Shell AO indices must cover the public AO order exactly once.')
                selection[int(index)]=-1
            shared = all(basis.aos[int(index)].angular == component and
                         all(getattr(basis.aos[int(index)], field) is getattr(shell, field)
                             for field in ('center', 'exponents', 'coefficients'))
                         for index, component in zip(indices, shell.angulars))
            if shared:
                for index, component in zip(indices, shell.angulars):
                    selection[int(index)]=offset+components.index(component)
                angular.append(l);exponents.append(jnp.asarray(shell.exponents))
                coefficients.append(jnp.asarray(shell.coefficients)[:,None]);centers.append(jnp.asarray(shell.center))
            else:
                for index in indices:
                    selection[int(index)]=append(basis.aos[int(index)])
            offset=sum((l+1)*(l+2)//2 for l in angular)
        if any(index is None for index in selection):raise ValueError('Shell metadata omits a public AO.')
        selection=tuple(selection)
    else:
        for ao in basis.aos:
            selection.append(append(ao))
        selection=tuple(selection)
    coords=basis.atom_coords if atom_coords is None else atom_coords
    charges=basis.atom_charges if atom_charges is None else atom_charges
    if coords is None or charges is None:coords=jnp.zeros((0,3));charges=()
    coords=jnp.asarray(coords,dtype=jnp.float64)
    weights=jnp.asarray(charges,dtype=jnp.float64)
    if coords.ndim!=2 or coords.shape[1]!=3 or weights.shape!=(len(coords),):
        raise ValueError('Nuclear positions and weights require (n,3) and (n,).')
    integer=False
    if not any(isinstance(leaf,jax.core.Tracer) for leaf in jax.tree.leaves(charges)):
        values=np.asarray(charges)
        integer=bool(np.isfinite(values).all() and np.all(values>=0)
            and np.all(values==np.trunc(values)) and np.all(values<=np.iinfo(np.int32).max))
    # Only the point-nuclear operator uses ATM charges. Weighted potentials use
    # live Rinv origins/weights; zeros here are irrelevant operator metadata.
    charges=tuple(int(x) for x in values) if integer else (0,)*len(coords)
    topology=BasisTopology(tuple(angular),tuple(a.size for a in exponents),
        tuple(c.shape[1] for c in coefficients),tuple(charges),True)
    parameters=BasisParameters(tuple(exponents),tuple(coefficients),jnp.stack(centers),jnp.asarray(coords))
    return CartesianBinding(make_plan(topology),parameters,selection,weights,integer)


__all__=['make_plan','bind_cartesian']
