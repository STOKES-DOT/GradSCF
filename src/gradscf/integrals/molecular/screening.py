"""Schwarz estimates from bounded libcint shell windows."""
import jax.numpy as jnp
import numpy as np
from ..libcint import bind_cartesian
from .eri import eri_element


def schwarz_bounds(basis):
    result=jnp.zeros((basis.nao,basis.nao),dtype=jnp.float64)
    for i in range(basis.nao):
        for j in range(i+1):
            value=jnp.sqrt(jnp.maximum(eri_element(basis,i,j,i,j),0.))
            result=result.at[i,j].set(value).at[j,i].set(value)
    return result


def shell_pair_schwarz_bounds(basis):
    binding=bind_cartesian(basis);count=len(binding.plan.topology.angular_momenta)
    result=np.zeros((count,count))
    for i in range(count):
        for j in range(i+1):
            block=binding.evaluate('eri',shell_slice=(i,i+1,j,j+1,i,i+1,j,j+1))
            result[i,j]=result[j,i]=np.sqrt(max(float(jnp.max(jnp.abs(block))),0.))
    return result

__all__ = ['schwarz_bounds', 'shell_pair_schwarz_bounds']
