"""A derivative-only first request must initialize its own libcint handlers."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('layout', ['s4', 's8', 'window', 'slab'])
@pytest.mark.parametrize('product', ['jvp', 'vjp', 'hvp'])
def test_bounded_geometry_derivative_first_in_fresh_process(layout, product):
    source = Path(__file__).parents[2]/'src'
    code = r'''
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from gradscf import integrals
from gradscf.integrals.backends.native.autodiff.geometry import evaluate_eri_slab

top, p = integrals.prepare_basis('H 0 0 0; H 0 0 1.4',
    {'H': [[0, [.8, 1.]]]}, unit='Bohr', cart=True)
plan = integrals.make_plan(top)
coordinates = jnp.concatenate((p.nuclear_coords, p.centers))
direction = jnp.array([[0., 0., 0.], [0., 0., .03],
                       [0., 0., 0.], [0., 0., .07]])
if LAYOUT == 'slab':
    fixed = replace(p, centers=jnp.zeros_like(p.centers),
                    nuclear_coords=jnp.zeros_like(p.nuclear_coords))
    atoms, basis, env = plan.pack(fixed)
    value = lambda r: evaluate_eri_slab(atoms, basis, env, r,
        jnp.array([0, 1, 1, 2, 0, 2, 0, 2], dtype=jnp.int32),
        (1, 1, 2, 2), cart=True)
else:
    def value(r):
        moving = replace(p, nuclear_coords=r[:2], centers=r[2:])
        return plan.evaluate('eri', moving,
            aosym=LAYOUT if LAYOUT in ('s4', 's8') else 's1',
            shell_slice=(0, 1, 1, 2, 0, 2, 0, 2) if LAYOUT == 'window' else None)
loss = lambda r: jnp.sum(value(r))
gradient = jax.grad(loss)
if PRODUCT == 'jvp':
    function = lambda r: jax.jvp(value, (r,), (direction,))[1]
elif PRODUCT == 'vjp':
    function = gradient
else:
    function = lambda r: jax.jvp(gradient, (r,), (direction,))[1]
# No earlier value call, explicit register call, or diagnostic helper is used.
actual = jax.block_until_ready(jax.jit(function)(coordinates))
assert np.isfinite(actual).all() and np.linalg.norm(np.asarray(actual)) > 1e-7
h = 1e-4
if PRODUCT == 'jvp':
    expected = (value(coordinates+h*direction)-value(coordinates-h*direction))/(2*h)
elif PRODUCT == 'hvp':
    expected = (gradient(coordinates+h*direction)-gradient(coordinates-h*direction))/(2*h)
else:
    expected = (loss(coordinates+h*direction)-loss(coordinates-h*direction))/(2*h)
    actual = jnp.vdot(actual, direction)
np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=3e-6)
'''
    code = 'LAYOUT='+repr(layout)+'\nPRODUCT='+repr(product)+'\n'+code
    environment = dict(os.environ, PYTHONPATH=str(source),
                       JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1')
    process = subprocess.run([sys.executable, '-c', code], env=environment,
                             capture_output=True, text=True, timeout=6000)
    assert process.returncode == 0, process.stdout+process.stderr
