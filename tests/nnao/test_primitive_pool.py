"""Active primitive pools keep fixed core contractions and their electrons."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.model.nnao import prepare_direct_basis
from gradscf.model.nnao.initialization import pool_channel_keys, with_pool_parameters
from gradscf.model.nnao.integrals.pool import active_pool_basis, pool_projection


def test_only_valence_and_polarization_are_uncontracted():
    layout = prepare_direct_basis('C 0 0 0; H 0 0 1.09', cart=False,
                                  basis_family='szp663_direct', core_primitives=6)
    top, parameters = active_pool_basis(layout)
    assert layout.topology.nao == 14 and top.nao == 52
    assert sum(top.nuclear_charges) == 7  # Core electrons were not removed.
    for slot, old, new, count in zip(layout.slots, layout.parameters.coefficients,
                                    parameters.coefficients, top.contraction_counts):
        if slot < 0:
            assert count == 1
            np.testing.assert_array_equal(new, old)
        else:
            np.testing.assert_array_equal(new, np.eye(old.shape[0]))


def test_pool_scale_and_spacing_are_independent_and_role_specific():
    layout = prepare_direct_basis('C 0 0 0', cart=False, basis_family='szp663_direct')
    keys = pool_channel_keys([layout])
    assert ('C', 2, 'polarization') in keys
    values = jnp.tile(jnp.array([.04, .08]), (len(keys), 1))
    moved = with_pool_parameters(layout, values, keys)
    for slot, a, b in zip(layout.slots, layout.parameters.exponents, moved.parameters.exponents):
        if slot < 0:
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(jnp.log(b).mean()-jnp.log(a).mean(), .04, atol=1e-13)
            np.testing.assert_allclose(jnp.ptp(jnp.log(b))/jnp.ptp(jnp.log(a)), np.exp(.08))
    derivative = jax.jit(jax.grad(lambda x: sum(jnp.sum(a) for a in
        with_pool_parameters(layout, x, keys).parameters.exponents)))(values)
    assert np.isfinite(derivative).all()


def test_repeated_primitive_exponents_are_rejected():
    layout = prepare_direct_basis('H 0 0 0', cart=False, basis_family='szp663_direct')
    aa = list(layout.parameters.exponents)
    aa[0] = aa[0].at[1].set(aa[0][0])
    bad = replace(layout, parameters=replace(layout.parameters, exponents=tuple(aa)))
    with pytest.raises(ValueError, match='duplicate|spacing'):
        pool_channel_keys([bad])


@pytest.mark.parametrize('deformed', [False, True])
def test_mixed_pool_projection_matches_native_contraction_and_gradient(deformed):
    from gradscf import integrals
    from gradscf.integrals.molecular.density_fitting import project_factors, unpack_factors
    atom = 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587'
    layout = prepare_direct_basis(atom, cart=False, basis_family='szp663_direct')
    if deformed:
        keys = pool_channel_keys([layout])
        layout = with_pool_parameters(layout, jnp.full((len(keys), 2), .08), keys)
    top, parameters = active_pool_basis(layout)
    assert top.nao == 64 and layout.topology.nao == 18
    at, ap = integrals.prepare_basis(atom, 'def2-universal-jkfit', cart=False)
    aux = integrals.make_auxiliary_plan(top, at)
    metric = aux.metric_factor(parameters, ap)
    plan = integrals.make_plan(top)
    s = plan.evaluate('overlap', parameters)
    h = plan.evaluate('kinetic', parameters)+plan.evaluate('nuclear', parameters)
    b = aux.factors(parameters, ap, metric_factor=metric)
    contracted_plan = integrals.make_plan(layout.topology)
    contracted_aux = integrals.make_auxiliary_plan(layout.topology, at)
    def project(c):
        t = pool_projection(layout, c)
        return t.T@s@t, t.T@h@t, project_factors(b, t)
    def native(c):
        p = layout.bind(c)
        return (contracted_plan.evaluate('overlap', p),
            contracted_plan.evaluate('kinetic', p)+contracted_plan.evaluate('nuclear', p),
            unpack_factors(contracted_aux.factors(p, ap, metric_factor=metric), layout.topology.nao))
    c = layout.reference_outputs()+.02*jnp.sin(jnp.arange(layout.reference_outputs().size).reshape(layout.reference_outputs().shape))
    a, vjp_a = jax.vjp(project, c)
    z, vjp_z = jax.vjp(native, c)
    for x, y in zip(a, z):
        np.testing.assert_allclose(x, y, rtol=2e-10, atol=2e-10)
    cot = tuple(jnp.cos(jnp.arange(x.size)).reshape(x.shape) for x in a)
    np.testing.assert_allclose(vjp_a(cot)[0], vjp_z(cot)[0], rtol=2e-9, atol=2e-9)
