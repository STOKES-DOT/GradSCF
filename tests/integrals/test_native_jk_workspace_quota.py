"""Private accumulation quota is execution metadata, not a physical input."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis, native_jk_workspace_bytes


def basis():
    top, params = integrals.prepare_basis('H 0 0 0; H .1 .2 1.2; H -.2 .1 2.3',
        [[0, [1.2, .8], [.4, .3]], [1, [.7, 1.]]], cart=True)
    return NativeDirectBasis(integrals.make_plan(top), params)


@pytest.mark.parametrize('quota', [0, 8192, 2*1024**2])
def test_quota_preserves_values_and_density_products(quota):
    original = basis()
    selected = NativeDirectBasis(original.plan, original.parameters, private_output_bytes=quota)
    eri = original.plan.evaluate('eri', original.parameters)
    rng = np.random.default_rng(1104)
    d, v = jnp.asarray(rng.normal(size=(2, 2, original.nao, original.nao)))
    full = lambda x: (jnp.einsum('pqrs,...rs->...pq', eri, x),
                      jnp.einsum('prqs,...rs->...pq', eri, x))
    action = lambda x: selected.get_jk(x, workers=4)
    for a, b in zip(jax.tree.leaves(jax.jvp(action, (d,), (v,))),
                    jax.tree.leaves(jax.jvp(full, (d,), (v,)))):
        np.testing.assert_allclose(a, b, atol=2e-10, rtol=2e-10)
    weights = tuple(jnp.sin(x) for x in full(d))
    scalar = lambda x, fn: sum(jnp.sum(w*y) for w, y in zip(weights, fn(x)))
    actual = jax.jit(jax.grad(lambda x: scalar(x, action)))(d)
    expected = jax.grad(lambda x: scalar(x, full))(d)
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)


def test_budgeted_provider_keeps_physics_and_matches_workspace_estimate():
    from gradscf.integrals.backends.native.workspace import bind_jk_workspace
    from gradscf._memory import workspace_bytes
    original = basis()
    selected = bind_jk_workspace(original, workers=4, max_memory=8., resident_bytes=1024)
    assert selected.parameters is original.parameters and selected.plan is original.plan
    assert selected.private_output_bytes is not None
    assert 3*native_jk_workspace_bytes(selected, 4, 0.) <= workspace_bytes(8., resident_bytes=1024)
    bounded = bind_jk_workspace(replace(selected, private_output_bytes=8192),
        workers=4, max_memory=8., resident_bytes=1024)
    assert bounded.private_output_bytes <= 8192


def test_legacy_provider_tree_and_quota_metadata():
    original = basis()
    old = NativeDirectBasis.tree_unflatten((original.plan.topology, original.plan.backend),
                                          (original.parameters,))
    assert old.private_output_bytes is None
    selected = replace(old, private_output_bytes=8192)
    restored = jax.tree.map(lambda x: x, selected)
    assert restored.private_output_bytes == 8192
    bare = native_jk_workspace_bytes(selected, 4, 0., private_output_bytes=0)
    assert native_jk_workspace_bytes(selected, 4, 0.)-bare == 8192


@pytest.mark.parametrize('unrestricted', [False, True])
def test_scf_input_assembly_binds_existing_max_memory(unrestricted):
    from gradscf.scf.inputs import build_rks_integral_inputs, build_uks_integral_inputs
    from gradscf.scf import RKSConfig, UKSConfig
    builder = build_uks_integral_inputs if unrestricted else build_rks_integral_inputs
    config = (UKSConfig if unrestricted else RKSConfig)(xc_spec='pbe0', jk_backend='direct',
        direct_scf_workers=4, max_memory=16., max_cycle=4)
    data = builder(atom='H 0 0 0; H 0 0 .8', basis='sto-3g',
        charge=1 if unrestricted else 0, spin=1 if unrestricted else 0,
        config=config, grids_level=0, cart=True)
    assert data.direct_basis.private_output_bytes is not None
    assert 3*native_jk_workspace_bytes(data.direct_basis, 4, 0.) < 16e6
