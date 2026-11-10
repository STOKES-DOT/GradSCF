"""A uniform derivative workload must not leave one large first-shell tail."""
from dataclasses import replace
import os
import re
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis


@pytest.mark.skipif(os.environ.get('GRADSCF_EXPECT_CPU_DIAGNOSTICS') != '1',
                    reason='requires disposable CPU-instrumented diagnostic source')
def test_uniform_geometry_vjp_uses_balanced_canonical_tasks(capfd, record_property):
    atom = ';'.join(f'H {.1*np.sin(i)} {.1*np.cos(i)} {.8*i}' for i in range(8))
    topology, parameters = integrals.prepare_basis(atom, [[0, [.7, 1.]]], cart=True)
    plan = integrals.make_plan(topology)
    rng = np.random.default_rng(1101)
    density = jnp.asarray(rng.normal(size=(topology.nao, topology.nao)))
    weights = jnp.asarray(rng.normal(size=(2, topology.nao, topology.nao)))
    def loss(r, workers):
        basis = NativeDirectBasis(plan, replace(parameters, centers=r, nuclear_coords=r))
        result = basis.get_jk(density, workers=workers)
        return sum(jnp.sum(a*b) for a, b in zip(result, weights))
    coordinates = parameters.nuclear_coords
    serial = jax.jit(jax.grad(lambda r: loss(r, 1)))
    parallel = jax.jit(jax.grad(lambda r: loss(r, 4)))
    expected = jax.block_until_ready(serial(coordinates))
    jax.block_until_ready(parallel(coordinates))
    capfd.readouterr()
    actual = jax.block_until_ready(parallel(coordinates))
    trace = capfd.readouterr().err
    loads = [int(x) for x in re.findall(r'GRADSCF_CPU_GEOMETRY.*quartets=(\d+)', trace)]
    record_property('worker_quartets', str(loads))
    record_property('diagnostic_worker_trace', trace)
    assert len(loads) == 4, trace
    pairs = 8*9//2
    assert sum(loads) == pairs*(pairs+1)//2
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
    assert min(loads) > 0 and max(loads) <= 2*min(loads), loads
