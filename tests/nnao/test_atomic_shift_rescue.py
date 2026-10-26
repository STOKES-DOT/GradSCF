"""The atomic fallback must solve and validate the unshifted SCF problem."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from gradscf.model.nnao.training.objective import _run_scf_with_rescue
from gradscf.scf.rks import RKSConfig


@pytest.mark.parametrize('atomic_converged,polish_converged',[(True,True),(False,True),(False,False)])
def test_atomic_shift_rescue_is_bounded_and_requires_raw_polish(atomic_converged,polish_converged):
    class Result(NamedTuple):
        converged: object
        cycles: object
        density_matrix: object
        mo_coeff: object
        mo_energy: object

    events=[]
    def solver(*,config,init_density=None,init_mo_coeff=None,init_mo_energy=None):
        seed=jnp.asarray(0.) if init_density is None else init_density[0,0]
        shifted=config.level_shift>0
        stage=jnp.where(seed==7,4 if shifted else 3,
                        jnp.where(seed==8,5,1 if shifted else 2 if init_density is not None else 0))
        error=jnp.asarray(0.)
        if init_mo_coeff is not None:
            error=jnp.where(stage==5,jnp.abs(init_mo_coeff[0,0]-9)+jnp.abs(init_mo_energy[0]-10)
                +abs(config.level_shift-cfg.level_shift)+abs(config.damping-cfg.damping),0.)
        jax.debug.callback(lambda k,e:events.append((int(k),float(e))),stage,error,ordered=True)
        converged=((stage==3)&atomic_converged)|((stage==5)&polish_converged)|(stage==4)
        value=jnp.where(stage==4,8.,1.)
        return Result(converged,jnp.asarray(2),jnp.full((1,1),value),jnp.full((1,1),9.),jnp.full(1,10.))

    cfg=RKSConfig(xc_spec='hf')
    result,used,cycles=jax.jit(lambda:_run_scf_with_rescue(solver,{},cfg,
        rescue_level_shift=.2,fallback_guess=lambda:jnp.full((1,1),7.)))()
    jax.block_until_ready((result,used,cycles))
    expected=[0,1,2,1,2,3]+([] if atomic_converged else [4,5])
    assert events==[(k,0.) for k in expected]
    assert bool(result.converged)==(atomic_converged or polish_converged)
    assert bool(used) and int(cycles)==2*len(expected)
