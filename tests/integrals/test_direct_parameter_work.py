"""Basis adjoint work selection before native lowering, ordinary Coulomb.

Trace the existing float64/Bohr fixture. A coefficient-only gradient must not
schedule exponent kernels; exponent gradients must retain the normalization
coefficient term as well as the native bare-Gaussian exponent term.
"""
import jax
import jax.numpy as jnp
import pytest

from gradscf.integrals.backends.native.jk import NativeDirectBasis
from test_direct_parameter_ad import changed, point_and_direction, problem


def _primitive_names(graph):
    if hasattr(graph, "jaxpr"):
        yield from _primitive_names(graph.jaxpr)
    elif hasattr(graph, "eqns"):
        for equation in graph.eqns:
            yield equation.primitive.name
            yield from _primitive_names(equation.params)
    elif isinstance(graph, dict):
        for value in graph.values():
            yield from _primitive_names(value)
    elif isinstance(graph, (tuple, list)):
        for value in graph:
            yield from _primitive_names(value)


@pytest.mark.parametrize("field, has_exponent_pull", [
    ("coefficients", False), ("log_exponents", True)])
def test_parameter_gradient_schedules_only_needed_native_adjoints(problem, field, has_exponent_pull):
    plan, parameters, density = problem
    point, _ = point_and_direction(parameters, field)
    weights = (jnp.sin(density), .3*jnp.cos(density))

    def loss(value):
        provider = NativeDirectBasis(plan, changed(parameters, field, value))
        result = provider.get_jk(density, workers=2, screening_threshold=0.)
        return sum(jnp.sum(matrix*weight) for matrix, weight in zip(result, weights))

    names = set(_primitive_names(jax.make_jaxpr(jax.grad(loss))(point)))
    assert "gradscf_direct_jk_coefficient_vjp" in names
    assert ("gradscf_direct_jk_exponent_vjp" in names) == has_exponent_pull
