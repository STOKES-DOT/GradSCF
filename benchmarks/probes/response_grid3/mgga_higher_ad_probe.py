"""Opt-in TPSS higher-AD diagnostic, outside normal test discovery.

The combined candidate-v1 group was terminated after 601.07 s with peak RSS
40,656,637,952 bytes and four buffered dots, without XML. Individual test
outcomes cannot be assigned from that log; the TPSS result is inconclusive.
Frozen v1 production SHA: 9a1df0bb7cadbb83efedaf8e022cba88bf091f78debb43977d8ead35169baee3.
Original test SHA: 8ced42700ff51df0baa70214be477a9566493d23cc15238dbca2a4ecfaa2f544.

To reproduce the attempted prepared-MGGA probe, use that frozen v1 source.
The final source deliberately retains the original MGGA block-local HVP.
Run explicitly inside Slurm with GRADSCF_RUN_SLOW_MGGA_AD=1, JAX_ENABLE_X64=1:
python -m pytest -vv -s benchmarks/probes/response_grid3/mgga_higher_ad_probe.py
"""
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.molecules import QuadratureGrid
from gradscf.tddft import _unrestricted_semilocal_response as spin

pytestmark = pytest.mark.skipif(
    os.environ.get('GRADSCF_RUN_SLOW_MGGA_AD') != '1' or not os.environ.get('SLURM_JOB_ID'),
    reason='Optional high-order MGGA diagnostic requires explicit Slurm opt-in')


def _dynamic_local_action(theta, spec, kind, block_size):
    # Algebraic AO/grid and density directions, not a molecular geometry claim.
    base = jnp.linspace(.15, .85, 20).reshape(5, 4)
    variation = jnp.linspace(-.04, .07, 20).reshape(5, 4)
    ao = base + theta * variation
    # Independent orbital derivatives keep MGGA away from the iso-orbital boundary.
    pattern = jnp.linspace(-.08, .12, 20).reshape(5, 4)
    derivative = jnp.stack((ao, .11*ao + pattern + .03*theta,
                            -.13*ao + .7*pattern[::-1] + .02*theta,
                            .17*ao - .4*pattern[:, ::-1] - .01*theta))
    density = jnp.stack((jnp.diag(jnp.array([1., .8, .12, .07])),
                         jnp.diag(jnp.array([.7, .3, .09, .04]))))
    direction = jnp.stack((.07*jnp.eye(4) + .01*jnp.ones((4, 4)),
                           -.03*jnp.eye(4) + .005*jnp.ones((4, 4))))
    reference = SimpleNamespace(ao=ao, ao_deriv1=derivative,
        rdm1=density + theta*direction,
        grid=QuadratureGrid(jnp.linspace(.1, .3, 5) * jnp.exp(.2*theta)))
    coeff = jnp.eye(4)
    functional = spin.UnrestrictedSemilocalResponseFunctional(spec)
    action = spin.build_unrestricted_semilocal_response_action(
        reference, coeff[:, :2], coeff[:, 2:], coeff[:, :1], coeff[:, 1:],
        functional.spin_grid_response_hvp, feature_kind=kind,
        dtype=jnp.float64, block_size=block_size)
    return action(jnp.linspace(-.2, .3, 8).reshape(2, 2, 2),
                  jnp.linspace(.15, -.1, 6).reshape(2, 1, 3))


@pytest.mark.parametrize('spec,kind', [('tpss', 'MGGA')])
def test_prepared_hessian_keeps_dynamic_reference_jvp_vjp_and_fd(spec, kind):
    prepared = jax.jit(lambda theta: _dynamic_local_action(theta, spec, kind, 3))
    # None selects the actual unprepared full-grid HVP, not another blocked path.
    unprepared = jax.jit(lambda theta: _dynamic_local_action(theta, spec, kind, None))
    point, direction = jnp.asarray(.04), jnp.asarray(.13)
    first = jax.block_until_ready(prepared(point))
    for actual, expected in zip(first, unprepared(point)):
        np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-11)
    moved = point + .035
    changed = jax.block_until_ready(prepared(moved))
    assert max(float(jnp.max(jnp.abs(a-b))) for a, b in zip(changed, first)) > 1e-9
    for actual, expected, fresh in zip(changed, unprepared(moved),
                                       _dynamic_local_action(moved, spec, kind, 3)):
        np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-11)
        np.testing.assert_allclose(actual, fresh, atol=3e-11, rtol=3e-11)

    tangent = jax.jvp(prepared, (point,), (direction,))[1]
    expected_tangent = jax.jvp(unprepared, (point,), (direction,))[1]
    assert max(float(jnp.max(jnp.abs(value))) for value in tangent) > 1e-10
    for actual, expected in zip(tangent, expected_tangent):
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-9)
    cotangent = tuple(jnp.linspace(.1, .4, value.size).reshape(value.shape) for value in first)
    gradient = jax.vjp(prepared, point)[1](cotangent)[0]
    expected_gradient = jax.vjp(unprepared, point)[1](cotangent)[0]
    assert np.isfinite(float(gradient)) and abs(float(gradient)) > 1e-10
    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-9, rtol=2e-9)

    def loss(theta):
        return sum(jnp.vdot(weight, value) for weight, value in zip(cotangent, unprepared(theta)))

    # Dimensionless theta perturbs density, AO/derivative and weights together.
    for step in (3e-4, 1e-4):
        finite = (loss(point + step) - loss(point - step)) / (2*step)
        np.testing.assert_allclose(gradient, finite, atol=3e-7, rtol=3e-5)
