"""Moving Becke partitions keep values/response without an unrolled pair graph."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.data.molecule import parse_molecule_spec
from gradscf.integrals.grids import (
    _build_molecular_grid_from_spec_jax, _build_molecular_grid_from_spec_numpy)

FOUR_ATOMS = "O 0 0 0; H 0 -1.4 1.1; H 0 1.4 1.1; He 2.3 .2 -.5"


def test_partition_program_does_not_inline_every_pair_per_atom():
    spec = parse_molecule_spec(FOUR_ATOMS, unit="Bohr")
    program = jax.make_jaxpr(lambda r: _build_molecular_grid_from_spec_jax(
        replace(spec, coords_bohr=r), level=0))(jnp.asarray(spec.coords_bohr))
    visited = set()
    def count(graph):
        graph = getattr(graph, "jaxpr", graph)
        if id(graph) in visited:
            return 0
        visited.add(id(graph))
        total = 0
        for equation in graph.eqns:
            total += equation.primitive.name.startswith("scatter")
            for parameter in equation.params.values():
                for body in parameter if isinstance(parameter, (tuple, list)) else (parameter,):
                    if hasattr(body, "jaxpr") or hasattr(body, "eqns"):
                        total += count(body)
        return total
    assert count(program) <= 4*len(spec.symbols), "Pair updates must be staged, not copied O(natom^3) times"


@pytest.mark.parametrize("atom", ["He 0 0 0", FOUR_ATOMS], ids=["no-pairs", "mixed-atoms"])
def test_partition_values_jvp_vjp_hvp_match_numpy_and_two_fd_steps(atom):
    spec = parse_molecule_spec(atom, unit="Bohr")
    r = jnp.asarray(spec.coords_bohr)
    direction = jnp.sin(jnp.arange(r.size, dtype=r.dtype)+1).reshape(r.shape)
    direction = direction/jnp.linalg.norm(direction)
    grid = lambda x: _build_molecular_grid_from_spec_jax(replace(spec, coords_bohr=x), level=0)
    value = jax.jit(grid)(r)
    reference = _build_molecular_grid_from_spec_numpy(spec, level=0)
    for actual, expected in zip(value, reference):
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    _, tangent = jax.jvp(grid, (r,), (direction,))
    cotangent = tuple(jnp.cos(jnp.arange(x.size, dtype=r.dtype)).reshape(x.shape)/x.size for x in value)
    _, pull = jax.vjp(grid, r)
    left = sum(jnp.vdot(a, b) for a, b in zip(cotangent, tangent))
    np.testing.assert_allclose(left, jnp.vdot(pull(cotangent)[0], direction), atol=2e-10, rtol=2e-10)
    def moment(x):
        coords, weights = grid(x)
        return jnp.sum(weights*jnp.exp(-jnp.sum(coords**2, axis=1)))
    gradient = jax.jit(jax.grad(moment))
    g = gradient(r)
    hvp = jax.jit(lambda x: jax.jvp(gradient, (x,), (direction,))[1])(r)
    for step in (2e-4, 1e-4):
        plus, minus = jax.jit(moment)(r+step*direction), jax.jit(moment)(r-step*direction)
        np.testing.assert_allclose(jnp.vdot(g, direction), (plus-minus)/(2*step), atol=3e-7, rtol=3e-7)
        np.testing.assert_allclose(hvp, (gradient(r+step*direction)-gradient(r-step*direction))/(2*step),
                                   atol=3e-6, rtol=3e-6)
