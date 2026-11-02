"""Nine primitive-pool / valence-zeta combinations preserve angular symmetry."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.integrals.pool import active_pool_basis, pool_projection


@pytest.mark.parametrize('tier,pool_size', [('s', 32), ('m', 37), ('l', 52)])
@pytest.mark.parametrize('zeta', [1, 2, 3])
def test_grid_dimensions_and_independent_radial_columns(tier, pool_size, zeta):
    family = f'nnao_{("szp", "dzp", "tzp")[zeta-1]}_{tier}'
    layout = prepare_direct_basis('C 0 0 0; H 0 0 1.1', basis_family=family)
    assert layout.topology.nao == 9 + 5*zeta
    assert active_pool_basis(layout)[0].nao == pool_size
    raw = layout.reference_outputs()
    parameters = layout.bind(raw)
    for role, l, alpha, c, n in zip(layout.roles, layout.topology.angular_momenta,
            parameters.exponents, parameters.coefficients, layout.topology.contraction_counts):
        assert n == (zeta if role.startswith('valence_') else 1)
        if role != 'core':
            a = np.asarray(alpha)
            metric = (2*np.sqrt(a[:, None]*a[None, :])/(a[:, None]+a[None, :]))**(l+1.5)
            assert np.linalg.eigvalsh(c.T@metric@c).min() > 1e-5
    t = pool_projection(layout, raw)
    assert t.shape == (pool_size, layout.topology.nao)
    derivative = jax.grad(lambda x: jnp.sum(pool_projection(layout, x)**2))(raw)
    assert np.isfinite(derivative).all()


def test_large_single_zeta_preserves_existing_layout():
    old = prepare_direct_basis('O 0 0 0; H 0 0 1', basis_family='szp663_direct')
    new = prepare_direct_basis('O 0 0 0; H 0 0 1', basis_family='nnao_szp_l', core_primitives=6)
    assert old.topology == new.topology
    np.testing.assert_array_equal(old.reference_outputs(), new.reference_outputs())
    for a, b in zip(old.parameters.exponents, new.parameters.exponents):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('family', ['nnao_dzp_s', 'nnao_tzp_m'])
def test_multizeta_projection_matches_native_overlap_and_vjp(family):
    from gradscf import integrals
    layout = prepare_direct_basis('O 0 0 0; H 0 -.757 .587; H 0 .757 .587', basis_family=family)
    top, p = active_pool_basis(layout)
    s = integrals.make_plan(top).evaluate('overlap', p)
    native = integrals.make_plan(layout.topology)
    raw = layout.reference_outputs()
    def project(c):
        t = pool_projection(layout, c)
        return t.T@s@t
    def direct(c):
        return native.evaluate('overlap', layout.bind(c))
    a, va = jax.vjp(project, raw)
    b, vb = jax.vjp(direct, raw)
    np.testing.assert_allclose(a, b, atol=1e-11, rtol=1e-11)
    assert np.linalg.eigvalsh(a).min() > 1e-5
    cot = jnp.cos(jnp.arange(a.size)).reshape(a.shape)
    np.testing.assert_allclose(va(cot)[0], vb(cot)[0], atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('family', ['nnao_dzp_s', 'nnao_tzp_m'])
def test_multizeta_implicit_energy_direction(family):
    from gradscf.model.nnao.initialization import pool_channel_keys
    from gradscf.model.nnao.training.pool import PoolRHF
    row = dict(structure_id='H2', symbols=['H', 'H'], charge=0, spin=0,
               coords_angstrom=[[0., 0., 0.], [0., 0., .74]], weight=1.)
    atom = list(zip(row['symbols'], row['coords_angstrom']))
    layout = prepare_direct_basis(atom, basis_family=family)
    keys = pool_channel_keys([layout])
    model = PoolRHF(row, keys, basis_family=family)
    free = model.prepare_contraction(jnp.zeros((len(keys), 2)))
    c = layout.reference_outputs()
    energy, gradient, info = free.evaluate(c)
    direction = np.cos(np.arange(c.size)).reshape(c.shape)
    direction /= np.linalg.norm(direction)
    step = 1e-4
    plus = free.evaluate(c+step*direction, gradient=False)[0]
    minus = free.evaluate(c-step*direction, gradient=False)[0]
    np.testing.assert_allclose(np.sum(gradient*direction), (plus-minus)/(2*step),
                               atol=2e-6, rtol=2e-5)
    assert info['converged'] and info['stable'] and np.isfinite(energy)


def test_grid_workflow_accepts_explicit_basis_metadata():
    from gradscf.model.nnao.workflows.pool_calibration import _validate_definition
    _validate_definition(dict(method='RHF', units='Angstrom',
                              basis_family='nnao_tzp_s', core_primitives=6))
    with pytest.raises(ValueError):
        _validate_definition(dict(method='RHF', units='Angstrom',
                                  basis_family='nnao_unknown', core_primitives=6))


@pytest.mark.parametrize('family', ['nnao_szp_s', 'nnao_dzp_m', 'nnao_tzp_l'])
def test_real_mace_grid_head_matches_reference(family):
    pytest.importorskip('mace_jax')
    from gradscf.model.nnao.config import MODEL_CONFIG
    from gradscf.model.nnao.graph import build_graph
    from gradscf.model.nnao.training.trainer import make_mace_predictor
    atom = [('O', [0., 0., 0.]), ('H', [0., -.757, .587]), ('H', [0., .757, .587])]
    layout = prepare_direct_basis(atom, basis_family=family)
    initial, predict, _ = make_mace_predictor(0, dict(MODEL_CONFIG, basis_family=family))
    graph = build_graph([8, 1, 1], [a[1] for a in atom], element_order=MODEL_CONFIG['elements'])
    np.testing.assert_allclose(predict(initial, graph), layout.reference_outputs(), atol=1e-13)


def test_grid_candidate_selection_and_reuse(tmp_path):
    import json
    from gradscf.model.nnao.workflows.pool_calibration import load_references, run_pilot
    from gradscf.model.nnao.workflows.basis_grid import select_candidate
    from gradscf.model.nnao.initialization import bind_pool_candidate
    data = dict(load_references(ids=['H2']), basis_family='nnao_dzp_s', core_primitives='6-31g')
    run_pilot(data, tmp_path/'pilot', calibration_steps=0, contraction_steps=1, starts=1)
    candidate = select_candidate(tmp_path/'pilot', data)
    layout = prepare_direct_basis('H 0 0 0; H 0 0 .74', basis_family='nnao_dzp_s')
    bound = bind_pool_candidate(layout, candidate)
    assert bound.topology.nao == 10
    reused = run_pilot(data, tmp_path/'reused', calibration_steps=0,
                       contraction_steps=1, starts=1, fixed_candidate=candidate)
    assert reused['calibration_status'] == 'fixed_candidate_reused'
    assert len(reused['comparisons']) == 1
    report = tmp_path/'pilot/summary.json'
    result = json.loads(report.read_text())
    result['comparisons'] = []
    report.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='No evaluated'):
        select_candidate(tmp_path/'pilot', data)


@pytest.mark.parametrize('family', ['nnao_dzp_s', 'nnao_tzp_m'])
def test_real_mace_multizeta_train_resume(tmp_path, family):
    import json
    pytest.importorskip('mace_jax')
    from gradscf.model.nnao.workflows.pool_calibration import load_references
    from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
    from gradscf.model.nnao.initialization import pool_channel_keys, with_pool_parameters
    data = dict(load_references(ids=['H2']), basis_family=family)
    layout = prepare_direct_basis('H 0 0 0; H 0 0 .74', basis_family=family, core_primitives=data['core_primitives'])
    keys = pool_channel_keys([layout])
    values = np.full((len(keys), 2), .01)
    moved = with_pool_parameters(layout, values, keys)
    channels = {}
    for atom, l, role, slot, a, b in zip(layout.shell_atoms, layout.topology.angular_momenta,
            layout.roles, layout.slots, layout.parameters.exponents, moved.parameters.exponents):
        if slot >= 0:
            channels[('H', l, role)] = dict(element='H', l=l, role=role,
                reference_exponents_bohr_minus2=np.asarray(a).tolist(),
                candidate_exponents_bohr_minus2=np.asarray(b).tolist())
    candidate = dict(parameters=dict(keys=keys, values=values.tolist()), channels=list(channels.values()))
    args = dict(data=data, candidate=candidate, cache_dir=tmp_path/'cache', output_dir=tmp_path/'train')
    first = run_fixed_pool(**args, epochs=1)
    resumed = run_fixed_pool(**args, epochs=2, resume=True)
    assert first['accepted_epoch'] == 1 and resumed['accepted_epoch'] == 2
    assert resumed['status'] == 'complete'
    assert json.loads((tmp_path/'train/gate.json').read_text())['passed']
