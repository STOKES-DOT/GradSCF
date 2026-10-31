"""Shared metadata bodies are scanned once but charged at every call site."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp

from gradscf._memory import jaxpr_temporary_bytes


class Equations:
    def __init__(self, values):
        self.values, self.visits = values, 0

    def __iter__(self):
        self.visits += 1
        return iter(self.values)


def graph(children=(), *, leaf=False):
    variable = SimpleNamespace(aval=jax.ShapeDtypeStruct((4,) if leaf else (), jnp.float64))
    equation = SimpleNamespace(primitive=SimpleNamespace(name="alloc" if leaf else "call"),
        outvars=(variable,), params={"branches": children})
    return SimpleNamespace(eqns=Equations((equation,)))


def test_shared_diamond_keeps_multiplicity_without_exponential_visits():
    nodes = [graph(leaf=True)]
    for _ in range(12):
        nodes.append(graph((nodes[-1], nodes[-1])))
    assert jaxpr_temporary_bytes(nodes[-1]) == (32+8)*2**12-8
    assert [node.eqns.visits for node in nodes] == [1]*len(nodes)


def test_closed_wrappers_share_the_same_body():
    leaf = graph(leaf=True)
    root = graph((SimpleNamespace(jaxpr=leaf), SimpleNamespace(jaxpr=leaf)))
    assert jaxpr_temporary_bytes(root) == 8+2*32
    assert leaf.eqns.visits == 1


def test_scratch_charges_and_cache_are_local_to_each_estimate():
    leaf = graph(leaf=True)
    root = graph((leaf, leaf))
    assert jaxpr_temporary_bytes(root, primitive_bytes={"alloc": 64}) == 8+2*(32+64)
    assert jaxpr_temporary_bytes(root, primitive_bytes={"alloc": 128}) == 8+2*(32+128)
    assert leaf.eqns.visits == 2


def test_real_nested_jit_graph_retains_the_original_byte_sum():
    inner = jax.jit(lambda x: jnp.sin(x)+x)
    program = jax.make_jaxpr(lambda x: inner(x)+inner(x+1.))(
        jax.ShapeDtypeStruct((3,), jnp.float64))
    # Four outer vectors and two inner vectors at each of two call sites.
    assert jaxpr_temporary_bytes(program) == (4+2*2)*3*8
