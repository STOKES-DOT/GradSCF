"""Fixed-field local potentials with shared projection and compiled scoring."""
from dataclasses import dataclass, replace
from functools import lru_cache
import time

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.dft.iks import project_potential


FIELD_NAMES = ('n', 'sigma', 'grad', 's', 'u')
GGA_FIELDS = frozenset(FIELD_NAMES)
BLOCK_SIZE = 8192
MAX_PARAMETERS = 4
PROJECTION_MEMORY_MB = 8000.


def fixed_fields(rho, gradient, *, floor, block_size=BLOCK_SIZE):
    """Cache only fixed-target scalar descriptors; padding is a regular vacuum."""
    rho, gradient = jnp.asarray(rho), jnp.asarray(gradient)
    if rho.ndim != 1 or gradient.shape != (rho.size, 3) or not rho.size or block_size < 1:
        raise ValueError('Require density (ngrids,) and gradient (ngrids,3)')
    n = jnp.sqrt(rho*rho+floor*floor)
    sigma = jnp.sum(gradient*gradient, axis=1)
    s2 = sigma/(4*(3*jnp.pi**2)**(2/3)*n**(8/3))
    fields = jnp.stack((n, sigma, jnp.sqrt(sigma), jnp.sqrt(s2), s2/(1+s2)), axis=1)
    padding = (-rho.size) % block_size
    if padding:
        fields = jnp.pad(fields, ((0,padding),(0,0)))
        fields = fields.at[rho.size:,0].set(1.)
    return tuple(fields.reshape((-1,block_size,len(FIELD_NAMES))))


@lru_cache(maxsize=8)
def field_program(candidate):
    """One small local-value/parameter-AD program, reused across all targets."""
    names = tuple(p.name for p in candidate.parameters)
    def point(vector, fields):
        params = {name: vector[index] for index,name in enumerate(names)}
        return candidate.evaluate(params, dict(zip(FIELD_NAMES,fields)))
    derivative = jax.vmap(jax.value_and_grad(point, argnums=0), in_axes=(None,0))
    @jax.jit
    def channels(fields):
        value, gradient = derivative(jnp.zeros(MAX_PARAMETERS,dtype=fields.dtype),fields)
        return jnp.concatenate((value[:,None],gradient),axis=1)
    return channels


@jax.jit
def _embed(matrices, transform):
    return jax.vmap(lambda matrix: transform.T@matrix@transform)(matrices)


def _potential_score(problem, baseline, offset, panels, vector):
    return problem.score(baseline+offset+jnp.tensordot(vector,panels,axes=1))


_score_program = jax.jit(_potential_score)


def _negative_reward(problem, baseline, offset, panels, vector):
    score = _potential_score(problem,baseline,offset,panels,vector)
    return jnp.where(score.valid & jnp.isfinite(score.reward),-score.reward,jnp.nan)


_loss_program = jax.jit(jax.value_and_grad(_negative_reward,argnums=4))


@jax.jit
def _diagnose_program(reporter, result):
    return reporter.diagnose(result)


@dataclass(frozen=True)
class BoundPotential:
    case: object
    candidate: object
    offset: object
    panels: object

    @property
    def names(self): return tuple(p.name for p in self.candidate.parameters)

    @property
    def directions(self): return {name:self.panels[index] for index,name in enumerate(self.names)}

    def _vector(self, params):
        values = jnp.asarray([params[name] for name in self.names],dtype=self.offset.dtype)
        return jnp.pad(values,(0,MAX_PARAMETERS-values.size))

    def potential(self, params):
        return self.case.vxc+self.offset+jnp.tensordot(self._vector(params),self.panels,axes=1)

    def __call__(self, params):
        return _score_program(self.case.problem,self.case.vxc,self.offset,self.panels,self._vector(params))

    def loss_and_grad(self, vector):
        padded = jnp.pad(vector,(0,MAX_PARAMETERS-vector.size))
        value, gradient = _loss_program(self.case.problem,self.case.vxc,self.offset,self.panels,padded)
        return value, gradient[:vector.size]

    def diagnose_result(self, result):
        return _diagnose_program(self.case.reporter,result)


def bind_potential(case, candidate):
    """Assemble affine potential channels, with no density differentiation."""
    if (candidate.kind!='potential' or not candidate.affine or
            not candidate.features<=GGA_FIELDS or len(candidate.parameters)>case.max_parameters or
            set(candidate.initial)&GGA_FIELDS):
        raise ValueError('Direct binding requires an affine density/gradient potential')
    started = time.perf_counter()
    if not hasattr(case,'_direct_field_chunks'):
        case._direct_field_chunks = fixed_fields(case.arrays['rho'],case.arrays['grad_rho'],floor=case.floor)
    program = field_program(candidate)
    channels = jnp.concatenate([program(chunk) for chunk in case._direct_field_chunks],axis=0)[:case.grid.ngrids]
    channels.block_until_ready()
    if not bool(jnp.all(jnp.isfinite(channels))):
        raise ValueError('Candidate potential is nonfinite on the reference grid')
    fields_seconds = time.perf_counter()-started
    grid = replace(case.grid,block_size=min(BLOCK_SIZE,case.grid.ngrids))
    projection_started = time.perf_counter()
    if bool(jnp.any(channels[:,0]!=0)):
        offset = _embed(project_potential(grid,channels[:,:1],max_memory=PROJECTION_MEMORY_MB),case.transform)[0]
    else:
        offset = jnp.zeros_like(case.vxc)
    if candidate.parameters:
        panels = _embed(project_potential(grid,channels[:,1:],max_memory=PROJECTION_MEMORY_MB),case.transform)
    else:
        panels = jnp.zeros((MAX_PARAMETERS,)+case.vxc.shape,dtype=case.vxc.dtype)
    jax.block_until_ready((offset,panels))
    case.timings.update(last_potential_fields_seconds=fields_seconds,
        last_potential_projection_seconds=time.perf_counter()-projection_started,
        last_candidate_assembly_seconds=time.perf_counter()-started)
    return {case.system_id:BoundPotential(case,candidate,offset,panels)}
