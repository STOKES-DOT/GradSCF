"""Fully reoptimized CASSCF energy and reduced-density response."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.mc.orbitals import solve_casscf
from .test_casscf import CASES, molecular_case


def _directional_difference(function, value, direction, step):
    return (function(value + step * direction) - function(value - step * direction)) / (2. * step)


def geometry_case(name):
    _, inputs, opts, _, _ = molecular_case(name)
    atom, basis, _, _, _, _, _, _ = CASES[name]
    topology, parameters = integrals.prepare_basis(atom, basis=basis, cart=False)
    plan = integrals.make_plan(topology)
    charges = jnp.asarray(topology.nuclear_charges)
    upper = jnp.triu_indices(charges.size, k=1)
    center_offsets = parameters.centers[:, None, :] - parameters.nuclear_coords[None, :, :]
    shell_atoms = jnp.argmin(jnp.sum(center_offsets**2, axis=-1), axis=1)

    def solve(coords, *, with_response=True):
        moving = replace(parameters, centers=coords[shell_atoms], nuclear_coords=coords)
        overlap = plan.evaluate("overlap", moving)
        h = plan.evaluate("kinetic", moving) + plan.evaluate("nuclear", moving)
        eri = plan.evaluate("eri", moving)
        inverse_distances = jnp.linalg.norm(coords[upper[0]] - coords[upper[1]], axis=1)
        enuc = jnp.sum(charges[upper[0]] * charges[upper[1]] / inverse_distances)
        return solve_casscf(overlap, h, inputs[2], enuc, **(opts | {"eri": eri}),
                            with_response=with_response)

    return name, parameters.nuclear_coords, solve, opts


@pytest.mark.parametrize("name", ["H2-R", "H3-D"])
def test_fully_reoptimized_geometry_energy(name):
    _, coords, solve, _ = geometry_case(name)
    energy = lambda r: solve(r).total_energy
    primal_energy = jax.jit(lambda r: solve(r, with_response=False).total_energy)
    _, gradient = jax.jit(jax.value_and_grad(energy))(coords)
    if name == "H2-R":
        reference = jnp.array([[0., 0., -0.067936230549270],
                               [0., 0., 0.067936230549270]])
        direction = jnp.zeros_like(coords).at[1, 2].set(1.)
        tolerance = 3e-6
        np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-7, rtol=0)
    else:
        reference = jnp.array([[-0.015587542014807, 0., -0.028379625416768],
                               [0.094323792061573, 0., -0.042777051454901],
                               [-0.078736250046767, 0., 0.071156676871669]])
        direction = jnp.sin(jnp.arange(coords.size, dtype=coords.dtype) + 1.).reshape(coords.shape)
        direction /= jnp.linalg.norm(direction)
        tolerance = 1e-5
    np.testing.assert_allclose(gradient, reference, rtol=0, atol=tolerance)
    computed = jax.jit(lambda r, d: jax.jvp(energy, (r,), (d,))[1])(coords, direction)
    np.testing.assert_allclose(computed, jnp.vdot(gradient, direction), rtol=0, atol=2e-8)
    for step in (2e-3, 1e-3):
        finite = _directional_difference(primal_energy, coords, direction, step)
        np.testing.assert_allclose(computed, finite, rtol=0, atol=tolerance)


def test_reoptimized_ao_densities():
    _, coords, solve, options = geometry_case("H2-R")
    operators, active = options["operators"], jnp.asarray(options["active"])

    def observable(coords, *, with_response=True):
        solution = solve(coords, with_response=with_response)
        coeff = solution.mo_coeff[:, active]
        d1, d2 = operators.rdm12(solution.ci_coeff)
        density1 = coeff @ d1 @ coeff.T
        density2 = jnp.einsum("pi,qj,rk,sl,ijkl->pqrs", coeff, coeff, coeff, coeff,
                              d2, optimize="optimal")
        return jnp.concatenate((density1.reshape(-1), density2.reshape(-1),
                                jnp.trace(d1)[None]))

    direction = jnp.zeros_like(coords).at[1, 2].set(1.)
    f = jax.jit(observable)
    output, pullback = jax.vjp(f, coords)
    weights = jnp.cos(jnp.arange(output.size, dtype=output.dtype) + 1.)
    weights /= jnp.linalg.norm(weights)
    lhs = jnp.vdot(pullback(weights)[0], direction)
    tangent = jax.jit(lambda r, d: jax.jvp(f, (r,), (d,))[1])(coords, direction)
    rhs = jnp.vdot(weights, tangent)
    np.testing.assert_allclose(lhs, rhs, rtol=0, atol=2e-8)
    np.testing.assert_allclose(tangent[-1], 0., rtol=0, atol=1e-8)
    primal = jax.jit(lambda r: observable(r, with_response=False))
    for step in (2e-4, 1e-4):
        fd = _directional_difference(primal, coords, direction, step)
        np.testing.assert_allclose(rhs, jnp.vdot(fd, weights), rtol=0, atol=8e-6)


def test_response_with_explicit_integrals():
    _, (s, h, c, enuc), opts, _, _ = molecular_case("H2-R")
    g = opts.pop("eri")
    arrays = (s, h, g, enuc)
    def energy(s, h, g, enuc):
        return solve_casscf(s, h, c, enuc, **opts, eri=g).total_energy
    jit_value, grad = jax.jit(jax.value_and_grad(energy, argnums=(0, 1, 2, 3)))(*arrays)
    np.testing.assert_allclose(jit_value, energy(*arrays), atol=1e-10, rtol=0)
    for index, value in enumerate(arrays):
        v = jnp.sin(jnp.arange(value.size, dtype=value.dtype) + 1.).reshape(value.shape)
        if index < 2:
            v = 0.5 * (v + v.T)
        elif index == 2:
            v = 0.5 * (v + v.transpose(1, 0, 2, 3))
            v = 0.5 * (v + v.transpose(0, 1, 3, 2))
            v = 0.5 * (v + v.transpose(2, 3, 0, 1))
        v /= jnp.linalg.norm(v)
        def replace_argument(x):
            return energy(*(arrays[:index] + (x,) + arrays[index+1:]))
        projected = jnp.vdot(grad[index], v)
        jvp = jax.jvp(replace_argument, (value,), (v,))[1]
        np.testing.assert_allclose(jvp, projected, atol=1e-8, rtol=0)
        for step in (2e-4, 1e-4):
            fd = _directional_difference(jax.jit(replace_argument), value, v, step)
            np.testing.assert_allclose(projected, fd, atol=3e-6, rtol=0)
