"""CPU ERI oracles versus actual GPU direct callbacks, float64 and Bohr.

All reference values/products are built on CPU before the ERI ban. Every live
direct operand is then explicitly transferred to one GPU; CPU output placement
cannot satisfy these tests. Only absence of a visible GPU permits a skip.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.integrals.plan import IntegralPlan
from gradscf.scf.rsh import _direct_rsh_jk
from test_direct_density_screen import CUTOFF, mask_for, system as screen_fixture
from test_direct_parameter_ad import changed, point_and_direction, problem as basis_fixture


@pytest.fixture(scope="module")
def devices():
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError as error:
        message = str(error)
        if (("Unknown backend" in message and "gpu" in message)
                or "No visible GPU devices" in message):
            pytest.skip("No GPU device is visible to JAX")
        raise  # A configured GPU initialization failure is a failed test setup.
    if not gpu_devices:
        pytest.skip("No GPU device is visible to JAX")
    return jax.devices("cpu")[0], gpu_devices[0]


def _assert_device(tree, device):
    for value in jax.tree.leaves(tree):
        assert value.devices() == {device}, (value.shape, value.devices(), device)
        assert next(iter(value.devices())).platform == device.platform


def _compare(actual, expected, gpu, *, atol=3e-9, rtol=3e-9):
    _assert_device(actual, gpu)
    for observed, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(np.asarray(observed)).all()
        np.testing.assert_allclose(np.asarray(observed), np.asarray(reference), atol=atol, rtol=rtol)


def _forbid_eri(monkeypatch):
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == "eri":
            raise AssertionError("GPU direct products must not materialize full or packed AO ERIs")
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, "evaluate", checked)


def _loss(values, weights):
    return sum(jnp.vdot(weight, value).real for value, weight in zip(values, weights))


def _geometry_bundle(action, coordinates, direction, weights):
    value_and_jvp = jax.jit(lambda r, u: jax.jvp(action, (r,), (u,)))(coordinates, direction)
    gradient = jax.jit(jax.grad(lambda r: _loss(action(r), weights)))
    vjp = gradient(coordinates)
    hvp = jax.jit(lambda r, u: jax.jvp(gradient, (r,), (u,))[1])(coordinates, direction)
    return value_and_jvp, vjp, hvp


@pytest.mark.parametrize("rsh", [False, True], ids=["ordinary", "rsh"])
def test_gpu_screened_geometry_products_use_cpu_oracle(devices, monkeypatch, rsh):
    cpu, gpu = devices
    with jax.default_device(cpu):
        # Calling the fixture body here prevents pytest setup from using the
        # process-wide CUDA default for its full-ERI reference calculation.
        system = screen_fixture.__wrapped__()
        plan, parameters = system.plan, system.parameters
        nnuc = len(system.top.nuclear_charges)
        coordinates = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
        direction = .07*jnp.sin(jnp.arange(coordinates.size).reshape(coordinates.shape)+.2)
        density, screen = system.density, system.screen
        weights = (jnp.cos(density), .3*jnp.sin(density))
        full_mask = mask_for(system.eri[0.], screen, system.offsets)
        assert full_mask.any() and not full_mask.all()
        full_mask = jnp.asarray(full_mask)
        if rsh:
            lr_mask = mask_for(system.eri[.3], screen, system.offsets, with_j=False, with_k=True)
            assert lr_mask.any() and not lr_mask.all()
            lr_mask = jnp.asarray(lr_mask)

        def reference(r):
            moving = replace(parameters, nuclear_coords=r[:nnuc], centers=r[nnuc:])
            eri = plan.evaluate("eri", moving)*full_mask
            coulomb = jnp.einsum("pqrs,...rs->...pq", eri, density)
            exchange = jnp.einsum("prqs,...rs->...pq", eri, density)
            if rsh:
                eri_lr = plan.evaluate("eri", moving, omega=.3)*lr_mask
                exchange = .19*exchange + (.65-.19)*jnp.einsum("prqs,...rs->...pq", eri_lr, density)
            return coulomb, exchange

        expected = _geometry_bundle(reference, coordinates, direction, weights)
        _assert_device((parameters, density, screen, coordinates, direction, weights, expected), cpu)
        expected = jax.tree.map(np.asarray, expected)

    _forbid_eri(monkeypatch)
    with jax.default_device(gpu):
        parameters, density, screen, coordinates, direction, weights = jax.device_put(
            (parameters, density, screen, coordinates, direction, weights), gpu)
        _assert_device((parameters, density, screen, coordinates, direction, weights), gpu)

        def action(r):
            moving = replace(parameters, nuclear_coords=r[:nnuc], centers=r[nnuc:])
            provider = NativeDirectBasis(plan, moving)
            controls = dict(screen_density=screen, screening_threshold=CUTOFF, workers=2)
            if rsh:
                return _direct_rsh_jk(provider, density, (.3, .65, .19), **controls)
            return provider.get_jk(density, **controls)

        actual = _geometry_bundle(action, coordinates, direction, weights)
    _compare(actual, expected, gpu)


@pytest.mark.parametrize("field", ["coefficients", "log_exponents"])
def test_gpu_first_basis_products_use_cpu_oracle(devices, monkeypatch, field):
    cpu, gpu = devices
    with jax.default_device(cpu):
        plan, parameters, density = basis_fixture.__wrapped__()
        point, direction = point_and_direction(parameters, field)
        screen = jnp.ones_like(density[0])
        weights = (jnp.sin(density), .3*jnp.cos(density))

        def reference(value):
            eri = plan.evaluate("eri", changed(parameters, field, value))
            return (jnp.einsum("pqrs,brs->bpq", eri, density),
                    jnp.einsum("prqs,brs->bpq", eri, density))

        expected_first = jax.jit(lambda p, v: jax.jvp(reference, (p,), (v,)))(point, direction)
        expected_vjp = jax.jit(jax.grad(lambda p: _loss(reference(p), weights)))(point)
        if field == "coefficients":
            assert abs(float(expected_vjp[1, 0])) > 1e-6
        expected = expected_first, expected_vjp
        _assert_device((parameters, density, screen, point, direction, weights, expected), cpu)
        expected = jax.tree.map(np.asarray, expected)

    _forbid_eri(monkeypatch)
    with jax.default_device(gpu):
        parameters, density, screen, point, direction, weights = jax.device_put(
            (parameters, density, screen, point, direction, weights), gpu)
        _assert_device((parameters, density, screen, point, direction, weights), gpu)

        def action(value):
            return plan.get_jk(changed(parameters, field, value), density,
                               screen_density=screen, screening_threshold=0., workers=2)

        actual_first = jax.jit(lambda p, v: jax.jvp(action, (p,), (v,)))(point, direction)
        actual_vjp = jax.jit(jax.grad(lambda p: _loss(action(p), weights)))(point)
    _compare((actual_first, actual_vjp), expected, gpu)


@pytest.mark.parametrize("field", ["coefficients", "log_exponents"])
def test_gpu_geometry_basis_mixed_products_use_cpu_first_gradient_fd(devices, monkeypatch, field):
    cpu, gpu = devices
    with jax.default_device(cpu):
        plan, parameters, density = basis_fixture.__wrapped__()
        point, direction = point_and_direction(parameters, field)
        centers, geometry_direction = point_and_direction(parameters, "centers")
        screen = jnp.ones_like(density[0])
        weights = (jnp.sin(density), .3*jnp.cos(density))

        def reference(r, value):
            moving = replace(changed(parameters, field, value), centers=r)
            eri = plan.evaluate("eri", moving)
            values = (jnp.einsum("pqrs,brs->bpq", eri, density),
                      jnp.einsum("prqs,brs->bpq", eri, density))
            return _loss(values, weights)

        # Only the first coordinate derivative is differentiated analytically
        # by the dense oracle; parameter differentiation uses finite differences.
        exact_geometry = jax.jit(jax.grad(reference, argnums=0))
        expected = []
        for step in (2e-4, 1e-4):
            coordinate_pull = (exact_geometry(centers, point+step*direction)
                               - exact_geometry(centers, point-step*direction))/(2*step)
            parameter_pull = []
            for index in range(point.size):
                unit = jnp.zeros_like(point).reshape(-1).at[index].set(1.).reshape(point.shape)
                delta = (exact_geometry(centers, point+step*unit)
                         - exact_geometry(centers, point-step*unit))/(2*step)
                parameter_pull.append(jnp.vdot(geometry_direction, delta))
            expected.append((coordinate_pull, jnp.stack(parameter_pull).reshape(point.shape)))
        assert all(float(jnp.linalg.norm(value)) > 1e-6 for value in expected[-1])
        _assert_device((parameters, density, screen, point, direction, centers,
                        geometry_direction, weights, expected), cpu)
        expected = jax.tree.map(np.asarray, expected)
        for left, right in zip(expected[0], expected[1]):
            np.testing.assert_allclose(left, right, atol=2e-7, rtol=4e-6)

    _forbid_eri(monkeypatch)
    with jax.default_device(gpu):
        parameters, density, screen, point, direction, centers, geometry_direction, weights = jax.device_put(
            (parameters, density, screen, point, direction, centers, geometry_direction, weights), gpu)
        _assert_device((parameters, density, screen, point, direction, centers, geometry_direction, weights), gpu)

        def loss(r, value):
            moving = replace(changed(parameters, field, value), centers=r)
            values = plan.get_jk(moving, density, screen_density=screen,
                                 screening_threshold=0., workers=2)
            return _loss(values, weights)

        geometry_gradient = jax.grad(loss, argnums=0)
        coordinate_pull = jax.jit(lambda x, v: jax.jvp(
            lambda y: geometry_gradient(centers, y), (x,), (v,))[1])(point, direction)
        parameter_pull = jax.jit(jax.grad(lambda x: jnp.vdot(
            geometry_direction, geometry_gradient(centers, x))))(point)
        left = jnp.vdot(geometry_direction, coordinate_pull)
        right = jnp.vdot(direction, parameter_pull)
    for reference in expected:
        _compare((coordinate_pull, parameter_pull), reference, gpu, atol=2e-7, rtol=4e-6)
    _assert_device((left, right), gpu)
    np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=2e-9, rtol=2e-8)
