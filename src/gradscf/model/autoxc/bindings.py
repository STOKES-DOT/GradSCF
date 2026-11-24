"""Prepared analytic local potentials; formula AD and physical AD stay separate."""
from dataclasses import dataclass,replace
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from gradscf._memory import validate_memory_mb
from gradscf.ao2mo.reference import transform_fock
from gradscf.dft.iks.potential import (project_potential,project_potential_pullback,
                                     _plan_potential_projection)
from .density import pair_block_size, pair_quadrature


@lru_cache(maxsize=16)
def _field_programs(candidate,names):
    parameters=tuple(p.name for p in candidate.parameters)
    def values(vector,fields):
        result=candidate.evaluate(dict(zip(parameters,vector)),
                                  {name:fields[:,i] for i,name in enumerate(names)})
        return jnp.broadcast_to(result,(fields.shape[0],))
    def pullback(vector,fields,cotangent):
        return jax.grad(lambda v:jnp.vdot(values(v,fields),cotangent))(vector)
    return jax.jit(values),jax.jit(pullback),jax.jit(jax.jacfwd(values,argnums=0))


@lru_cache(maxsize=16)
def _pair_programs(candidate, names, block_size):
    parameters = tuple(p.name for p in candidate.parameters)

    def values(vector, fields, coords, weights):
        pair = pair_quadrature(coords, weights, block_size,
                               jnp.result_type(vector, fields, coords, weights))
        result = candidate.evaluate(dict(zip(parameters, vector)),
                                    dict(zip(names, fields.T)), pair=pair)
        return jnp.broadcast_to(result, (coords.shape[0],))

    def pullback(vector, fields, coords, weights, cotangent):
        return jax.grad(lambda v: jnp.vdot(values(v, fields, coords, weights), cotangent))(vector)

    return jax.jit(values), jax.jit(pullback), jax.jit(jax.jacfwd(values, argnums=0))


@jax.jit
def _matrix_score(problem,baseline,correction):
    return problem.score(baseline+correction)


def _matrix_loss(problem,baseline,correction):
    score=_matrix_score(problem,baseline,correction)
    return jnp.where(score.valid & jnp.isfinite(score.reward),-score.reward,jnp.nan)


_matrix_value_grad=jax.jit(jax.value_and_grad(_matrix_loss,argnums=2))
_diagnose=jax.jit(lambda reporter,result:reporter.diagnose(result))
_embed=jax.jit(jax.vmap(transform_fock,in_axes=(0,None)))


def _prepare_pair_definitions(candidate, inputs, grid, vector, block_size, max_memory):
    fixed = candidate.fixed_pair_definitions
    if not fixed:
        return candidate, inputs
    inputs = dict(inputs)
    definitions = dict(candidate.definitions)
    for name in fixed:
        definition = replace(candidate, formula=name, definitions=tuple(definitions.items()))
        names = tuple(sorted(definition.features))
        fields = (jnp.stack([jnp.broadcast_to(inputs[key], (grid.ngrids,)) for key in names], axis=1)
                  if names else jnp.zeros((grid.ngrids, 0), dtype=vector.dtype))
        dtype = jnp.result_type(fields, grid.coords, grid.weights, vector)
        block = pair_block_size(grid.ngrids, block_size, max_memory, dtype)
        value = _pair_programs(definition, names, block)[0]
        inputs[name] = value(vector, fields, grid.coords, grid.weights)
        del definitions[name]
    return replace(candidate, definitions=tuple(definitions.items())), inputs


@dataclass(frozen=True)
class BoundLocalPotential:
    problem: object
    reporter: object
    grid: object
    candidate: object
    inputs: object
    names: tuple
    baseline: object
    transform: object
    max_memory: float
    cached_affine: bool
    offset: object = None
    panels: object = None
    sample_details: object = None
    pair_block: int = 0

    def _vector(self,parameters):
        return jnp.asarray([parameters[p.name] for p in self.candidate.parameters],dtype=self.baseline.dtype)

    def _values(self,vector):
        if self.pair_block:
            program = _pair_programs(self.candidate, self.names, self.pair_block)[0]
            values = program(vector, self.inputs, self.grid.coords, self.grid.weights)
        else:
            program=_field_programs(self.candidate,self.names)[0]
            values=jnp.concatenate([program(vector,chunk) for chunk in self.inputs])[:self.grid.ngrids]
        return jnp.where(jnp.all(jnp.isfinite(values)),values,jnp.nan)

    def _correction(self,vector):
        if self.cached_affine:
            return self.offset+jnp.tensordot(vector,self.panels,axes=1)
        raw=project_potential(self.grid,self._values(vector),max_memory=self.max_memory)
        return transform_fock(raw,self.transform)

    def potential(self,parameters):
        return self.baseline+self._correction(self._vector(parameters))

    def __call__(self,parameters):
        return _matrix_score(self.problem,self.baseline,self._correction(self._vector(parameters)))

    def loss_and_grad(self,vector):
        value,gradient=_matrix_value_grad(self.problem,self.baseline,self._correction(vector))
        if self.cached_affine:
            return value,jnp.einsum('ij,pij->p',gradient,self.panels)
        raw_gradient=self.transform@gradient@self.transform.T
        cotangent=project_potential_pullback(self.grid,raw_gradient,max_memory=self.max_memory)
        if self.pair_block:
            pullback = _pair_programs(self.candidate, self.names, self.pair_block)[1]
            return value, pullback(vector, self.inputs, self.grid.coords, self.grid.weights, cotangent)
        size=self.inputs.shape[1]
        cotangent=jnp.pad(cotangent,(0,self.inputs.shape[0]*size-cotangent.size)).reshape((-1,size))
        pullback=_field_programs(self.candidate,self.names)[1]
        result=jnp.zeros_like(vector)
        for fields,weight in zip(self.inputs,cotangent):
            result=result+pullback(vector,fields,weight)
        return value,result

    def diagnose_result(self,result):
        return _diagnose(self.reporter,result) if self.reporter is not None else result


def bind_local_potential(problem,grid,candidate,inputs,*,baseline=None,reporter=None,
                         transform=None,max_memory=8000.,block_size=8192,cache_affine=True):
    """Bind a formula to fixed sampled scalar features and the shared WY map.

    No term/parameter limit or density prefactor is imposed. Nonlinear
    parameters use pointwise VJPs plus the shared physical projection transpose;
    physical programs are independent of formula identity. Affine panels are
    cached only if their estimated storage fits one quarter of max_memory.
    This budget covers the projection/cache, not opaque model or compiler RSS.
    External neural/callable fields can use WuYang.local directly.
    """
    if candidate.kind!='potential' or not candidate.features<=set(inputs):
        raise ValueError('Require a local-potential candidate and all declared inputs')
    if not isinstance(block_size,int) or block_size<1:
        raise ValueError('block_size must be positive')
    max_memory=validate_memory_mb(max_memory)
    grid=replace(grid,block_size=min(block_size,grid.ngrids))
    baseline=jnp.asarray(problem.baseline_potential if baseline is None else baseline)
    if baseline.ndim!=2:
        raise NotImplementedError('Sampled analytic binding currently requires real RKS matrices')
    transform=jnp.eye(grid.nao,dtype=baseline.dtype) if transform is None else jnp.asarray(transform)
    if transform.shape!=(grid.nao,baseline.shape[0]):
        raise ValueError('AO embedding must map grid AOs to the problem space')
    if candidate.pair_operators:
        vector = jnp.asarray(tuple(candidate.initial.values()), dtype=baseline.dtype)
        candidate, inputs = _prepare_pair_definitions(candidate, inputs, grid, vector, block_size, max_memory)
    names=tuple(sorted(candidate.features));size=min(block_size,grid.ngrids)
    columns=[jnp.broadcast_to(jnp.asarray(inputs[name]),(grid.ngrids,)) for name in names]
    if any(jnp.iscomplexobj(c) for c in columns):raise ValueError('Local scalar features must be real')
    fields=jnp.stack(columns,axis=1) if columns else jnp.zeros((grid.ngrids,0),dtype=baseline.dtype)
    if not bool(jnp.all(jnp.isfinite(fields))):raise ValueError('Local scalar features must be finite')
    p=len(candidate.parameters)
    zeros=jnp.zeros(p,dtype=baseline.dtype)
    pair_block = 0
    if candidate.pair_operators:
        pair_block = pair_block_size(grid.ngrids, block_size, max_memory,
                                     jnp.result_type(fields, grid.coords, grid.weights, zeros))
        value, _, jacobian = _pair_programs(candidate, names, pair_block)
        chunks = fields
        value_args = (fields, grid.coords, grid.weights)
        local_value = lambda: value(zeros, *value_args)
        local_directions = lambda: jacobian(zeros, *value_args)
        field_dtype = jax.eval_shape(value, zeros, *value_args).dtype
    else:
        padding=(-grid.ngrids)%size
        fields=(jnp.pad(fields,((0,padding),(0,0)),mode='edge') if columns
                else jnp.zeros((grid.ngrids+padding,0),dtype=baseline.dtype))
        chunks=fields.reshape((-1,size,len(names)))
        value, _, jacobian = _field_programs(candidate,names)
        local_value = lambda: jnp.concatenate([value(zeros,c) for c in chunks])[:grid.ngrids]
        local_directions = lambda: jnp.concatenate([jacobian(zeros,c) for c in chunks])[:grid.ngrids]
        field_dtype=jax.eval_shape(value,zeros,chunks[0]).dtype
    _,physical_dtype=_plan_potential_projection(
        grid,jnp.zeros((grid.ngrids,1),dtype=field_dtype),max_memory)
    estimated=(grid.ngrids*(p+1)+3*p*baseline.size)*physical_dtype.itemsize
    cached=bool(cache_affine and candidate.affine and estimated<=max_memory*1e6*.25)
    if cached and p:
        try:
            _plan_potential_projection(grid,jnp.zeros((grid.ngrids,p),dtype=field_dtype),max_memory)
        except ValueError:
            cached=False
    offset=panels=None
    if cached:
        offset=transform_fock(project_potential(grid,local_value(),max_memory=max_memory),transform)
        if p:
            directions=local_directions()
            panels=_embed(project_potential(grid,directions,max_memory=max_memory),transform)
        else:panels=jnp.zeros((0,)+baseline.shape,dtype=baseline.dtype)
        jax.block_until_ready((offset,panels))
    return BoundLocalPotential(problem,reporter,grid,candidate,chunks,names,baseline,
                               transform,max_memory,cached,offset,panels,pair_block=pair_block)
