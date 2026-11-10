"""Known local spin Hessians are prepared once per numerical reference.

This traces actual Hessian evaluations, not elapsed time. It intentionally
fails before the preparation optimization; numerical execution is remote only.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.molecules import QuadratureGrid
from gradscf.tddft import _unrestricted_semilocal_response as spin


def test_known_local_joint_hessian_is_prepared_before_repeated_actions(monkeypatch):
    ao = jnp.linspace(.1, .9, 20).reshape(5, 4)
    reference = SimpleNamespace(
        ao=ao, ao_deriv1=jnp.stack((ao, .11*ao, -.13*ao, .17*ao)),
        grid=QuadratureGrid(jnp.linspace(.1, .3, 5)),
        rdm1=jnp.stack((jnp.diag(jnp.array([1., 1., 0., 0.])),
                       jnp.diag(jnp.array([1., 0., 0., 0.])))),
    )
    coeff = jnp.eye(4)
    orbital_args = (reference, coeff[:, :2], coeff[:, 2:], coeff[:, :1], coeff[:, 1:])
    functional = spin.UnrestrictedSemilocalResponseFunctional('pbe')
    alpha = jnp.linspace(-.2, .3, 8).reshape(2, 2, 2)
    beta = jnp.linspace(.15, -.1, 6).reshape(2, 1, 3)
    expected_action = spin.build_unrestricted_semilocal_response_action(
        *orbital_args, functional.spin_grid_response_hvp,
        feature_kind='GGA', dtype=jnp.float64, block_size=None)
    expected = jax.block_until_ready(expected_action(alpha, beta))
    original = spin.grid_hessian
    calls = []

    def observed(spec, kind):
        evaluate = original(spec, kind)

        def apply(fields, *args, **kwargs):
            calls.append(tuple(fields.shape))
            return evaluate(fields, *args, **kwargs)

        return apply

    monkeypatch.setattr(spin, 'grid_hessian', observed)
    action = spin.build_unrestricted_semilocal_response_action(
        *orbital_args, functional.spin_grid_response_hvp,
        feature_kind='GGA', dtype=jnp.float64, block_size=3)
    preparations = len(calls)
    assert preparations > 0, 'Known local joint Hessian is still deferred into the action'
    for scale in (1., .7):
        actual = jax.block_until_ready(action(scale*alpha, scale*beta))
        for result, reference_result in zip(actual, expected):
            np.testing.assert_allclose(result, scale*reference_result, atol=2e-11, rtol=2e-11)
    assert len(calls) == preparations, 'Repeated action re-evaluated the fixed-reference XC Hessian'


def test_known_hf_bypasses_grid_hessian_preparation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('HF must not prepare any grid Hessian')

    monkeypatch.setattr(spin, '_prepare_spin_grid_hessian', forbidden)
    reference = SimpleNamespace(grid=QuadratureGrid(jnp.ones(5)))
    coeff = jnp.eye(2)
    functional = spin.UnrestrictedSemilocalResponseFunctional('hf')
    action = spin.build_unrestricted_semilocal_response_action(
        reference, coeff[:, :1], coeff[:, 1:], coeff[:, :1], coeff[:, 1:],
        functional.spin_grid_response_hvp, feature_kind='LDA',
        dtype=jnp.float64, block_size=3)
    inputs = (jnp.ones((2, 1, 1)), jnp.ones((2, 1, 1)) * .3)
    for actual, source in zip(action(*inputs), inputs):
        np.testing.assert_array_equal(actual, jnp.zeros_like(source))


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


@pytest.mark.parametrize('spec,kind', [('svwn', 'LDA'), ('pbe', 'GGA')])
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


def test_known_mgga_keeps_block_local_hvp(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('MGGA must retain its original block-local HVP')

    monkeypatch.setattr(spin, '_prepare_spin_grid_hessian', forbidden)
    reference = SimpleNamespace(grid=QuadratureGrid(jnp.ones(5)))
    coeff = jnp.eye(2)
    functional = spin.UnrestrictedSemilocalResponseFunctional('tpss')
    action = spin.build_unrestricted_semilocal_response_action(
        reference, coeff[:, :1], coeff[:, 1:], coeff[:, :1], coeff[:, 1:],
        functional.spin_grid_response_hvp, feature_kind='MGGA',
        dtype=jnp.float64, block_size=3)
    assert callable(action)  # Block values/derivatives are covered by existing MGGA tests.
