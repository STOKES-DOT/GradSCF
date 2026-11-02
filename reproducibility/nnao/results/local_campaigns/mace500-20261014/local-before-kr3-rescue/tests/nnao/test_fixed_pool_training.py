"""Stage C fixes candidate exponents and differentiates only MACE contractions."""
import copy
import json

import numpy as np
import pytest

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import pool_channel_keys, with_pool_parameters
from gradscf.model.nnao.workflows.pool_calibration import load_references


def candidate_for(layout):
    keys = pool_channel_keys([layout])
    values = np.full((len(keys), 2), .03)
    moved = with_pool_parameters(layout, values, keys)
    channels = {}
    for atom, l, role, slot, before, after in zip(layout.shell_atoms,
            layout.topology.angular_momenta, layout.roles, layout.slots,
            layout.parameters.exponents, moved.parameters.exponents):
        if slot >= 0:
            key = (layout.symbols[atom], l, role)
            channels[key] = dict(element=key[0], l=l, role=role,
                reference_exponents_bohr_minus2=np.asarray(before).tolist(),
                candidate_exponents_bohr_minus2=np.asarray(after).tolist())
    return dict(channels=list(channels.values()),
                parameters=dict(keys=keys, values=values.tolist()))


def test_candidate_binding_checks_order_and_keeps_core():
    from gradscf.model.nnao.initialization import bind_pool_candidate
    layout = prepare_direct_basis('O 0 0 0', cart=False,
                                  basis_family='szp663_direct', core_primitives=6)
    data = candidate_for(layout)
    bound = bind_pool_candidate(layout, data)
    for slot, a, b in zip(layout.slots, layout.parameters.exponents, bound.parameters.exponents):
        if slot < 0:
            np.testing.assert_array_equal(a, b)
        else:
            assert not np.array_equal(a, b)
    bad = copy.deepcopy(data)
    bad['channels'][0]['reference_exponents_bohr_minus2'].reverse()
    with pytest.raises(ValueError, match='reference|order'):
        bind_pool_candidate(layout, bad)
    bad = copy.deepcopy(data)
    bad['channels'][0]['candidate_exponents_bohr_minus2'][0] *= 1.2
    with pytest.raises(ValueError, match='candidate|parameter'):
        bind_pool_candidate(layout, bad)
    bad = copy.deepcopy(data)
    bad['channels'].append(bad['channels'][0])
    with pytest.raises(ValueError, match='duplicate|unique'):
        bind_pool_candidate(layout, bad)


def test_fixed_pool_cache_reloads_without_native_calls_and_guards_identity(tmp_path, monkeypatch):
    from gradscf import integrals
    from gradscf.model.nnao.initialization import bind_pool_candidate
    from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
    row = load_references(ids=['H2'])['structures'][0]
    layout = prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
                                  cart=False, basis_family='szp663_direct', core_primitives=6)
    layout = bind_pool_candidate(layout, candidate_for(layout))
    path = tmp_path/'cache.npz'
    experiment, manifest = prepare_fixed_pool(row, layout, path, storage='host')
    c = layout.reference_outputs()
    e, g, _ = experiment.evaluate(c)
    before = path.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail('Fixed-pool hot/reload path called a native integral plan.')
    monkeypatch.setattr(integrals, 'make_plan', forbidden)
    monkeypatch.setattr(integrals, 'make_auxiliary_plan', forbidden)
    restored, other = prepare_fixed_pool(row, layout, path, storage='mmap')
    e2, g2, _ = restored.evaluate(c)
    assert manifest == other and path.read_bytes() == before
    np.testing.assert_allclose(e2, e, atol=1e-12, rtol=0)
    np.testing.assert_allclose(g2, g, atol=1e-10, rtol=1e-9)
    assert restored.raw[2].ndim == 2  # Packed DF; no four-index ERI.
    keys = pool_channel_keys([layout])
    moved = with_pool_parameters(layout, np.full((len(keys), 2), .01), keys)
    with pytest.raises(ValueError, match='identity|signature'):
        prepare_fixed_pool(row, moved, path)
    data = json.loads(path.with_suffix('.json').read_text())
    data['cache_sha256'] = 'bad'
    path.with_suffix('.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='hash|checksum'):
        prepare_fixed_pool(row, layout, path)


def test_real_mace_fixed_pool_training_and_resume(tmp_path):
    pytest.importorskip('cuequivariance')
    pytest.importorskip('mace_jax')
    from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
    data = load_references(ids=['H2'])
    layout = prepare_direct_basis('H 0 0 0; H 0 0 .74', cart=False,
                                  basis_family='szp663_direct', core_primitives=6)
    candidate = candidate_for(layout)
    args = dict(data=data, candidate=candidate, cache_dir=tmp_path/'cache', seed=0)
    first = run_fixed_pool(**args, output_dir=tmp_path/'split', epochs=1)
    assert first['status'] == 'complete' and first['accepted_epoch'] == 1
    resumed = run_fixed_pool(**args, output_dir=tmp_path/'split', epochs=2, resume=True)
    whole = run_fixed_pool(**args, output_dir=tmp_path/'whole', epochs=2)
    assert resumed['status'] == whole['status'] == 'complete'
    for name in ('parameters', 'gradient', 'optimizer'):
        with np.load(tmp_path/'split/last.npz') as a, np.load(tmp_path/'whole/last.npz') as b:
            np.testing.assert_allclose(a[name], b[name], rtol=0, atol=0)
    gate = json.loads((tmp_path/'split/gate.json').read_text())
    assert gate['passed'] and gate['native_derivatives_in_training'] is False
    assert first['identity']['trainable'] == 'mace_parameters_only'
    assert first['identity']['frozen_parameter_indices'] == [5, 262, 2631, 2888]
    assert first['identity']['trainable_parameter_count'] == 7167


def test_fixed_pool_workflow_rejects_nonuniform_weights_before_preparing(tmp_path):
    from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
    data = load_references(ids=['H2', 'CH4'])
    data['structures'][0]['weight'] = 2
    with pytest.raises(ValueError, match='equal'):
        run_fixed_pool(data, {}, tmp_path/'cache', tmp_path/'run', epochs=1)


def test_direction_gate_reuses_gradient_evaluator_and_records_each_result(tmp_path):
    from gradscf.model.nnao.workflows.fixed_pool import check_parameter_directions
    calls = []
    def evaluate(x):
        calls.append(x.copy())
        return float(x@x), 2*x, {}
    def wrong(x):
        return float(x@x), np.zeros_like(x), {}
    x = np.array([.3, -.4])
    records = check_parameter_directions(x, [('a', evaluate), ('bad', wrong)],
                                         seed=0, output_dir=tmp_path)
    assert len(calls) == 3
    assert records[0]['passed'] and not records[1]['passed']
    report = json.loads((tmp_path/'gate-progress.json').read_text())
    assert report['completed'] == 2 and report['total'] == 2
    assert report['checks'] == records


def test_direction_gate_does_not_perturb_immutable_model_constants(tmp_path):
    from gradscf.model.nnao.workflows.fixed_pool import check_parameter_directions
    calls = []
    def evaluate(x):
        calls.append(x.copy())
        # Mirrors an upstream Param(is_mutable=False) read with stop_gradient.
        return float(x[0]*np.sum(x[1:]**2)), np.r_[0., 2*x[0]*x[1:]], {}
    x = np.array([1.7, .4, .6])
    records = check_parameter_directions(x, [('model', evaluate)], seed=0,
        output_dir=tmp_path, trainable_mask=np.array([False, True, True]))
    assert records[0]['passed']
    assert all(point[0] == x[0] for point in calls)
    with pytest.raises(ValueError, match='mask|trainable'):
        check_parameter_directions(x, [('model', evaluate)], seed=0,
            output_dir=tmp_path, trainable_mask=np.zeros(3, dtype=bool))


def test_fixed_adam_preserves_frozen_values_even_with_old_momentum():
    from gradscf.model.nnao.training.trainer import Adam
    x = np.array([1.7, .4, .6])
    old = Adam(1e-4)
    _, state = old.update(x, np.ones(3), old.init(x))
    new = Adam(1e-4, trainable_mask=np.array([False, True, True]))
    # The exact old optimizer payload remains decodable; no state reset.
    restored = new.decode(old.encode(state), new.init(x))
    candidate, _ = new.update(x, np.ones(3), restored)
    assert candidate[0] == x[0]
    assert candidate[1] != x[1]


