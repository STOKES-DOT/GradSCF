"""Screened spherical d-shell basis/geometry response, CPU float64 and Bohr.

Two He atoms give even electron count. An individual zero raw coefficient,
general s contractions, actual five-component d shells, and nonsymmetric
complex densities exercise the shell transforms. The fixed partial mask is
an approximation, not a physical gradient-error bound. Dense mixed AD is
never used: mixed references are differences of supported first derivatives.
"""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.plan import IntegralPlan
from test_direct_density_screen import mask_for
from test_direct_parameter_ad import changed, point_and_direction


CUTOFF = 1e-3
STEPS = (2e-4, 1e-4)


@pytest.fixture(scope="module")
def spherical_problem():
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        topology, parameters = integrals.prepare_basis(
            "He 0 0 0; He .2 -.1 1.7",
            {"He": [[0, [1.3, .7, .2], [.35, 0., .6]],
                    [2, [.9, .6], [.25, .4]]]}, unit="Bohr", cart=False)
        dimensions = tuple(nc*(2*l+1) for l, nc in zip(topology.angular_momenta, topology.contraction_counts))
        assert dimensions == (2, 5, 2, 5) and topology.nao == 14
        assert sum(topology.nuclear_charges) % 2 == 0
        assert float(parameters.coefficients[0][1, 0]) == 0.
        offsets = np.concatenate(([0], np.cumsum(dimensions)))
        plan = integrals.make_plan(topology)
        rng = np.random.default_rng(20261112)
        shape = (2, topology.nao, topology.nao)
        density = jnp.asarray(.2*(rng.normal(size=shape)+1j*rng.normal(size=shape)))
        weights = tuple(jnp.asarray(rng.normal(size=shape)+1j*rng.normal(size=shape)) for _ in range(2))
        screen = np.zeros((2, topology.nao, topology.nao), dtype=np.complex128)
        # Retain interatomic s/d terms while omitting the isolated second d
        # quartet. The first reference block has no transpose counterpart.
        screen[0, offsets[0]:offsets[1], offsets[2]:offsets[3]] = .7j/4
        screen[1, offsets[1]:offsets[2], offsets[1]:offsets[2]] = .03*np.exp(.37j)/25
        screen = jnp.asarray(screen)
        tensor = np.asarray(plan.evaluate("eri", parameters))
        mask = mask_for(tensor, screen, offsets, cutoff=CUTOFF)
        assert mask.any() and not mask.all(), "A genuine partial quartet mask is required"
        d_first = slice(offsets[1], offsets[2])
        s_second = slice(offsets[2], offsets[3])
        assert mask[d_first, d_first, s_second, s_second].any(), "Retain interatomic d-shell response"
        geometry_direction = jnp.array([[.07, -.03, .05], [.07, -.03, .05],
                                        [-.02, .04, -.06], [-.02, .04, -.06]])
    return SimpleNamespace(cpu=cpu, topology=topology, parameters=parameters, plan=plan,
                           density=density, screen=screen, weights=weights, mask=mask,
                           offsets=offsets, geometry_direction=geometry_direction)


def _functions(problem, field):
    def at(centers, value):
        return replace(changed(problem.parameters, field, value), centers=centers)

    frozen_mask = jnp.asarray(problem.mask)

    def reference(centers, value):
        eri = problem.plan.evaluate("eri", at(centers, value))*frozen_mask
        return (jnp.einsum("pqrs,brs->bpq", eri, problem.density),
                jnp.einsum("prqs,brs->bpq", eri, problem.density))

    def direct(centers, value):
        return problem.plan.get_jk(at(centers, value), problem.density,
                                  screen_density=problem.screen,
                                  screening_threshold=CUTOFF, workers=2)

    return at, reference, direct


def _unit(point, index):
    return jnp.zeros_like(point).reshape(-1).at[index].set(1.).reshape(point.shape)


def _check_masks(problem, at, centers, point, direction):
    parameter_directions = (direction, *(_unit(point, i) for i in range(point.size)))
    for step in STEPS:
        for sign in (-1, 1):
            states = [(centers, point+sign*step*v) for v in parameter_directions]
            states.append((centers+sign*step*problem.geometry_direction, point))
            for r, x in states:
                tensor = np.asarray(problem.plan.evaluate("eri", at(r, x)))
                mask = mask_for(tensor, problem.screen, problem.offsets, cutoff=CUTOFF)
                np.testing.assert_array_equal(mask, problem.mask,
                    err_msg=f"Parameter/coordinate displacement changed the mask at step {sign*step}")


def _loss(values, weights):
    return sum(jnp.vdot(weight, value).real for value, weight in zip(values, weights))


def _compare(actual, expected, *, atol=3e-10, rtol=3e-9):
    for observed, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(np.asarray(observed)).all()
        assert next(iter(observed.devices())).platform == "cpu"
        np.testing.assert_allclose(observed, reference, atol=atol, rtol=rtol)


def _forbid_eri(monkeypatch):
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == "eri":
            raise AssertionError("Spherical direct products must not materialize full or packed ERIs")
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, "evaluate", checked)


@pytest.mark.parametrize("field", ["coefficients", "log_exponents"])
def test_screened_spherical_first_basis_products(spherical_problem, monkeypatch, field):
    p = spherical_problem
    with jax.default_device(p.cpu):
        point, direction = point_and_direction(p.parameters, field)
        centers = p.parameters.centers
        at, reference, direct = _functions(p, field)
        _check_masks(p, at, centers, point, direction)
        exact = jax.jit(lambda x: reference(centers, x))
        expected_first = jax.jit(lambda x, v: jax.jvp(exact, (x,), (v,)))(point, direction)
        expected_pull = jax.jit(jax.grad(lambda x: _loss(exact(x), p.weights)))(point)
        assert float(jnp.linalg.norm(expected_pull)) > 1e-6
        if field == "coefficients":
            assert abs(float(expected_pull[1, 0])) > 1e-6, "Zero raw coefficient must retain response"
        finite_differences = []
        for step in STEPS:
            push = jax.tree.map(lambda a, b: (a-b)/(2*step),
                               exact(point+step*direction), exact(point-step*direction))
            pull = [(_loss(exact(point+step*_unit(point, i)), p.weights)
                     - _loss(exact(point-step*_unit(point, i)), p.weights))/(2*step)
                    for i in range(point.size)]
            finite_differences.append((push, jnp.stack(pull).reshape(point.shape)))
        _forbid_eri(monkeypatch)

        action = lambda x: direct(centers, x)
        observed_first = jax.jit(lambda x, v: jax.jvp(action, (x,), (v,)))(point, direction)
        observed_pull = jax.jit(jax.grad(lambda x: _loss(action(x), p.weights)))(point)
        _compare((observed_first, observed_pull), (expected_first, expected_pull))
        for expected in finite_differences:
            _compare((observed_first[1], observed_pull), expected, atol=3e-8, rtol=3e-6)
        np.testing.assert_allclose(_loss(observed_first[1], p.weights), jnp.vdot(observed_pull, direction),
                                   atol=3e-10, rtol=3e-9)


@pytest.mark.parametrize("field", ["coefficients", "log_exponents"])
def test_screened_spherical_geometry_basis_pulls(spherical_problem, monkeypatch, field):
    p = spherical_problem
    with jax.default_device(p.cpu):
        point, direction = point_and_direction(p.parameters, field)
        centers, u = p.parameters.centers, p.geometry_direction
        at, reference, direct = _functions(p, field)
        _check_masks(p, at, centers, point, direction)
        exact_geometry = jax.jit(jax.grad(lambda r, x: _loss(reference(r, x), p.weights), argnums=0))
        references = []
        for step in STEPS:
            coordinate_pull = (exact_geometry(centers, point+step*direction)
                               - exact_geometry(centers, point-step*direction))/(2*step)
            parameter_pull = []
            for i in range(point.size):
                v = _unit(point, i)
                delta = (exact_geometry(centers, point+step*v)
                         - exact_geometry(centers, point-step*v))/(2*step)
                parameter_pull.append(jnp.vdot(u, delta))
            references.append((coordinate_pull, jnp.stack(parameter_pull).reshape(point.shape)))
        _compare(references[0], references[1], atol=2e-7, rtol=4e-6)
        assert all(float(jnp.linalg.norm(x)) > 1e-6 for x in references[-1])
        _forbid_eri(monkeypatch)

        loss = lambda r, x: _loss(direct(r, x), p.weights)
        geometry_gradient = jax.grad(loss, argnums=0)
        basis_gradient = jax.grad(loss, argnums=1)
        coordinate_pull = jax.jit(jax.grad(lambda r: jnp.vdot(
            direction, basis_gradient(r, point))))(centers)
        parameter_pull = jax.jit(jax.grad(lambda x: jnp.vdot(
            u, geometry_gradient(centers, x))))(point)
        # The same free-coordinate pull reached by differentiating the geometry
        # gradient must agree with transposing the basis-gradient route above.
        forward_coordinate = jax.jit(lambda x, v: jax.jvp(
            lambda y: geometry_gradient(centers, y), (x,), (v,))[1])(point, direction)
        _compare(forward_coordinate, coordinate_pull, atol=2e-9, rtol=2e-8)
        for expected in references:
            _compare((coordinate_pull, parameter_pull), expected, atol=2e-7, rtol=4e-6)
        np.testing.assert_allclose(jnp.vdot(u, coordinate_pull), jnp.vdot(direction, parameter_pull),
                                   atol=2e-9, rtol=2e-8)
