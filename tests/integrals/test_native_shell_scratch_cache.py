"""Scratch-size queries cache shapes, never physical integral values."""
from dataclasses import replace
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get('GRADSCF_EXPECT_DIAGNOSTICS') != '1',
                    reason='requires the separately instrumented native build')
@pytest.mark.parametrize('kind', ['jvp', 'vjp', 'hessian_jvp'])
def test_repeated_four_center_shapes_reuse_only_scratch_queries(kind):
    from gradscf import integrals
    from gradscf.integrals._native.diagnostics import get_counts, reset_counts
    from gradscf.integrals.backends.native.autodiff.eri_basis import evaluate_eri_basis

    top, parameters = integrals.prepare_basis('He 0 0 0',
        {'He': [[0, [.8, 1.]]]}, unit='Bohr', cart=True)
    plan = integrals.make_plan(top)
    fixed = replace(parameters, centers=jnp.zeros_like(parameters.centers),
                    nuclear_coords=jnp.zeros_like(parameters.nuclear_coords))
    atoms, basis, environment, coefficients = plan.coefficient_data(fixed)
    coordinates = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    def value(c):
        return evaluate_eri_basis(atoms, basis, environment, c, coordinates,
            jnp.zeros(3), top.nao, cart=True)
    direction = jnp.full_like(coefficients, .11)
    if kind == 'jvp':
        function = lambda c: jax.jvp(value, (c,), (direction,))[1]
        power = 3
    elif kind == 'vjp':
        function = jax.grad(lambda c: jnp.sum(value(c)))
        power = 3
    else:
        first = lambda c: jax.jvp(value, (c,), (direction,))[1]
        function = lambda c: jax.jvp(first, (c,), (direction,))[1]
        power = 2
    compiled = jax.jit(function)
    first = jax.block_until_ready(compiled(coefficients))
    reset_counts()
    second = jax.block_until_ready(compiled(coefficients*1.03))
    counts = get_counts()['families']['eri4c']
    assert counts['evaluations'] == (12 if kind == 'hessian_jvp' else 4)
    assert counts['cache_queries'] == 1
    np.testing.assert_allclose(second, first*1.03**power, atol=2e-12, rtol=2e-12)


@pytest.mark.skipif(os.environ.get('GRADSCF_EXPECT_DIAGNOSTICS') != '1',
                    reason='requires the separately instrumented native build')
def test_hessian_scratch_cache_keeps_kernel_identity_across_quartets(monkeypatch):
    from gradscf import integrals
    from gradscf.integrals._native.diagnostics import get_counts, reset_counts

    monkeypatch.setenv('GRADSCF_NATIVE_THREADS', '1')
    top, p = integrals.prepare_basis('H 0 0 0; H 0 0 1.4',
        {'H': [[0, [.8, 1.]]]}, unit='Bohr', cart=True)
    plan = integrals.make_plan(top)
    value = lambda centers: plan.evaluate('eri', replace(p, centers=centers))
    u = jnp.sin(jnp.arange(p.centers.size).reshape(p.centers.shape))*.1
    v = jnp.cos(jnp.arange(p.centers.size).reshape(p.centers.shape))*.1
    first = lambda r: jax.jvp(value, (r,), (u,))[1]
    compiled = jax.jit(lambda r: jax.jvp(first, (r,), (v,))[1])
    jax.block_until_ready(compiled(p.centers))
    reset_counts()
    actual = jax.block_until_ready(compiled(p.centers))
    counts = get_counts()['families']['eri4c']
    assert counts['evaluations'] > 3
    # ipip1, ipvip1 and ip1ip2 require three separate scratch sizes, despite
    # identical physical shell shapes. Switching kernels must retain all three.
    assert counts['cache_queries'] == 3
    assert np.isfinite(actual).all()
