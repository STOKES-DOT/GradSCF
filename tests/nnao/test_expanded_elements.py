"""Expanded main-group vocabulary, explicit reference cores, and no ECPs."""
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.initialization import FAMILY_GRID, pool_channel_keys

ELEMENTS = 'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Ga Ge As Se Br Kr'.split()


@pytest.mark.parametrize('family', tuple(FAMILY_GRID))
def test_all_dataset_elements_have_independent_all_electron_layouts(family):
    from gradscf import integrals
    from gradscf.data.molecule import atomic_number
    for symbol in ELEMENTS:
        layout = prepare_direct_basis(f'{symbol} 0 0 0', basis_family=family,
                                       core_primitives='reference')
        assert layout.nelectron == atomic_number(symbol)
        assert all(ecp is None for ecp in layout.ecps)
        assert len(pool_channel_keys([layout])) in (2, 3)
        output = layout.reference_outputs()
        bound = layout.bind(output+.01)
        for role, before, after in zip(layout.roles, layout.parameters.coefficients, bound.coefficients):
            if role == 'core':
                np.testing.assert_array_equal(before, after)
        s = integrals.make_plan(layout.topology).evaluate('overlap', layout.bind(output))
        assert np.isfinite(s).all()
        assert np.linalg.eigvalsh(s).min() > 1e-8, (symbol, family)


def test_reference_alias_uses_canonical_core_source_and_legacy_counts_stay_explicit():
    for symbol in ('C', 'Na', 'Cl', 'Ca'):
        old = prepare_direct_basis(f'{symbol} 0 0 0', core_primitives='6-31g')
        new = prepare_direct_basis(f'{symbol} 0 0 0', core_primitives='reference')
        assert old.topology == new.topology
        for a, b in zip(old.parameters.coefficients, new.parameters.coefficients):
            np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match='Unsupported|template'):
        prepare_direct_basis('Pb 0 0 0', core_primitives=3)
    with pytest.raises(ValueError, match='six-primitive'):
        prepare_direct_basis('I 0 0 0', core_primitives=6)


def test_heavy_graph_and_explicit_vocabulary():
    from gradscf.data.molecule import atomic_number
    from gradscf.model.nnao.config import model_config_for_elements, exponent_keys_for_elements
    elements = tuple(atomic_number(s) for s in ELEMENTS)
    config = model_config_for_elements(ELEMENTS, basis_family='nnao_tzp_m')
    assert config['elements'] == elements
    graph = build_graph([1, 35, 36], [[0, 0, 0], [0, 0, 1.8], [0, 0, 4]], element_order=elements)
    assert graph['node_attrs'].shape == (3, 26)
    np.testing.assert_array_equal(graph['node_type'], [0, elements.index(35), elements.index(36)])
    keys = exponent_keys_for_elements(ELEMENTS, basis_family='nnao_tzp_m')
    assert ('Kr', 2) in keys and ('Br', 1) in keys and ('Na', 2) not in keys
    with pytest.raises(ValueError, match='unique'):
        model_config_for_elements(['H', 'H'])


def test_reference_heavy_core_matches_unmodified_parent():
    import json
    from importlib.resources import files
    from gradscf.data.molecule import atomic_number
    data = json.loads(files('gradscf.model.nnao').joinpath('data/bse_631g_v1.json').read_text())
    for symbol in ('Ga', 'Br'):
        layout = prepare_direct_basis(f'{symbol} 0 0 0', basis_family='nnao_szp_s', core_primitives='reference')
        assert Counter(l for l, role in zip(layout.topology.angular_momenta, layout.roles) if role == 'core')[2] == 1
        parent = []
        for shell in data['elements'][str(atomic_number(symbol))]['electron_shells']:
            for index, c in enumerate(shell['coefficients']):
                l = shell['angular_momentum'][index] if len(shell['angular_momentum']) > 1 else shell['angular_momentum'][0]
                parent.append([l, *np.column_stack((np.asarray(shell['exponents'], dtype=float), np.asarray(c, dtype=float))).tolist()])
        for l, role, a, c in zip(layout.topology.angular_momenta, layout.roles,
                                  layout.parameters.exponents, layout.parameters.coefficients):
            if role == 'core':
                rows = np.column_stack((a, c))
                assert any(b[0] == l and np.array_equal(np.asarray(b[1:]), rows) for b in parent)


@pytest.mark.parametrize('symbol', ['Cl', 'Br', 'Ge', 'As', 'Se'])
def test_heavy_overlap_coefficient_vjp_has_fixed_core(symbol):
    from gradscf import integrals
    layout = prepare_direct_basis(f'{symbol} 0 0 0; H 0 0 2',
        basis_family='nnao_dzp_m', core_primitives='reference')
    plan = integrals.make_plan(layout.topology)
    c = layout.reference_outputs()
    objective = lambda x: jnp.sum(plan.evaluate('overlap', layout.bind(x))**2)
    gradient = jax.grad(objective)(c)
    direction = np.cos(np.arange(c.size)).reshape(c.shape)
    direction /= np.linalg.norm(direction)
    step = 1e-4
    fd = (objective(c+step*direction)-objective(c-step*direction))/(2*step)
    np.testing.assert_allclose(jnp.sum(gradient*direction), fd, atol=1e-6, rtol=2e-5)


def test_expanded_workflow_requires_explicit_vocabulary_and_records_core_policy():
    from gradscf.model.nnao.workflows.pool_calibration import _validate_definition, load_references
    from gradscf.model.nnao.config import model_config_for_definition
    data = dict(load_references(ids=['HF']), basis_family='nnao_szp_s', core_primitives='reference')
    data['structures'][0] = dict(data['structures'][0], symbols=['H', 'Cl'])
    with pytest.raises(ValueError, match='elements'):
        _validate_definition(data)
    data['elements'] = ELEMENTS
    _validate_definition(data)
    assert len(model_config_for_definition(data)['elements']) == 26
    assert model_config_for_definition(load_references())['elements'] == (1, 6, 7, 8, 9)


def test_grid_defaults_to_reference_core_policy_for_heavy_elements():
    layout = prepare_direct_basis('Br 0 0 0', basis_family='nnao_szp_m')
    assert layout.nelectron == 35 and layout.core_policy == '6-31g'
    assert Counter(n for n, role in zip(layout.topology.primitive_counts, layout.roles) if role == 'core') == {6:5,3:1}


def test_real_mace_expanded_heads_isolated_atoms_and_jit_gradient():
    pytest.importorskip('cuequivariance')
    pytest.importorskip('mace_jax')
    from gradscf.data.molecule import atomic_number
    from gradscf.model.nnao.config import model_config_for_elements
    from gradscf.model.nnao.training.trainer import make_mace_predictor
    config = model_config_for_elements(ELEMENTS, basis_family='nnao_tzp_m')
    initial, predict, backward = make_mace_predictor(0, config)
    z = [atomic_number(s) for s in ELEMENTS]
    graph = build_graph(z, np.zeros((len(z), 3)), batch=np.arange(len(z)), element_order=config['elements'])
    prediction = predict(initial, graph)
    assert prediction.shape == (26, 9, 4)
    for i, symbol in enumerate(ELEMENTS):
        layout = prepare_direct_basis(f'{symbol} 0 0 0', basis_family='nnao_tzp_m')
        np.testing.assert_allclose(prediction[i], layout.reference_outputs()[0], atol=1e-13)
    derivative = backward(initial, graph, 2*prediction)
    assert np.isfinite(derivative).all() and np.linalg.norm(derivative) > 0


def test_real_mace_expanded_implicit_scf_and_resume(tmp_path):
    import json
    pytest.importorskip('cuequivariance')
    pytest.importorskip('mace_jax')
    from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
    from gradscf.model.nnao.initialization import with_pool_parameters, core_reference_identity
    rows = [dict(structure_id='HCl', symbols=['H', 'Cl'], coords_angstrom=[[0,0,0],[0,0,1.27]],
                 charge=0, spin=0, weight=1),
            dict(structure_id='He', symbols=['He'], coords_angstrom=[[0,0,0]], charge=0, spin=0, weight=1)]
    family = 'nnao_szp_s'
    data = dict(method='RHF', units='Angstrom', basis_family=family, core_primitives='reference',
                elements=ELEMENTS, structures=rows)
    layouts = [prepare_direct_basis(list(zip(r['symbols'], r['coords_angstrom'])), basis_family=family) for r in rows]
    keys = pool_channel_keys(layouts)
    values = np.full((len(keys), 2), .01)
    channels = {}
    for layout in layouts:
        moved = with_pool_parameters(layout, values, keys)
        for atom, l, role, slot, a, b in zip(layout.shell_atoms, layout.topology.angular_momenta,
                layout.roles, layout.slots, layout.parameters.exponents, moved.parameters.exponents):
            if slot >= 0:
                key = (layout.symbols[atom], l, role)
                channels[key] = dict(element=key[0], l=l, role=role,
                    reference_exponents_bohr_minus2=np.asarray(a).tolist(),
                    candidate_exponents_bohr_minus2=np.asarray(b).tolist())
    candidate = dict(parameters=dict(keys=keys, values=values.tolist()), channels=list(channels.values()),
                     core_reference=core_reference_identity(layouts[0]))
    args = dict(data=data, candidate=candidate, cache_dir=tmp_path/'cache', output_dir=tmp_path/'train')
    first = run_fixed_pool(**args, epochs=1)
    second = run_fixed_pool(**args, epochs=2, resume=True)
    assert first['accepted_epoch'] == 1 and second['accepted_epoch'] == 2
    assert second['status'] == 'complete'
    assert json.loads((tmp_path/'train/gate.json').read_text())['passed']
