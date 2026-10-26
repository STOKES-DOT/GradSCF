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


def test_same_primitive_pool_cache_is_shared_across_zeta_levels(tmp_path,monkeypatch):
    from gradscf import integrals
    from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
    row=load_references(ids=['H2'])['structures'][0]
    atom=list(zip(row['symbols'],row['coords_angstrom']))
    layouts=[prepare_direct_basis(atom,basis_family=f'nnao_{z}_s',core_primitives='6-31g')
             for z in ['szp','dzp','tzp']]
    path=tmp_path/'shared.npz'
    _,first=prepare_fixed_pool(row,layouts[0],path,storage='mmap')
    saved=path.read_bytes()
    def forbidden(*args,**kwargs):
        pytest.fail('Changing only contraction multiplicity recomputed primitive integrals.')
    monkeypatch.setattr(integrals,'make_plan',forbidden)
    monkeypatch.setattr(integrals,'make_auxiliary_plan',forbidden)
    for layout in layouts[1:]:
        _,manifest=prepare_fixed_pool(row,layout,path,storage='mmap')
        assert manifest['signature']==first['signature']
        assert manifest['contracted_nao']==layout.topology.nao
        assert path.read_bytes()==saved


def test_shared_cache_has_a_single_writer_for_parallel_zeta_jobs(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    from gradscf.model.nnao.integrals import fixed_pool
    row=load_references(ids=['H2'])['structures'][0]
    layout=prepare_direct_basis('H 0 0 0; H 0 0 .74',basis_family='nnao_szp_s',core_primitives='6-31g')
    writes=[];barrier=threading.Barrier(2);original=fixed_pool.write_fixed_cache
    def write(*args):
        writes.append(1);time.sleep(.1)
        return original(*args)
    monkeypatch.setattr(fixed_pool,'write_fixed_cache',write)
    def prepare(_):
        barrier.wait()
        return fixed_pool.prepare_fixed_pool(row,layout,tmp_path/'shared.npz',storage='mmap')[1]
    with ThreadPoolExecutor(max_workers=2) as pool:
        a,b=list(pool.map(prepare,range(2)))
    assert writes==[1] and a==b


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


@pytest.fixture
def mocked_fixed_pool(monkeypatch):
    """Exercise workflow gates/checkpoints without native integrals or perturbations."""
    from gradscf.model.nnao import graph
    from gradscf.model.nnao.integrals import fixed_pool as integrals
    from gradscf.model.nnao.training import trainer
    from gradscf.model.nnao.workflows import fixed_pool as workflow
    data = load_references(ids=['H2'])
    layout = prepare_direct_basis('H 0 0 0; H 0 0 .74', cart=False,
                                  basis_family='szp663_direct', core_primitives=6)
    reference = np.asarray(layout.reference_outputs())
    reference = reference / np.maximum(np.linalg.norm(reference, axis=-1, keepdims=True), 1e-30)
    state = dict(calls=0, directions=0, forwards=0, backwards=0, energy=-1., gradient=np.zeros(2),
        info=dict(converged=True, stable=True, orbital_residual=0.,
                  fixed_point_residual=0., min_overlap_eigenvalue=1.,
                  reconstruction_error=0.))
    class Experiment:
        def evaluate_device(self, coefficients):
            state['calls'] += 1
            return state['energy'], np.zeros_like(coefficients), dict(state['info'])
    def predictor(*args, **kwargs):
        def predict(vector,graph):
            state['forwards']+=1
            return reference
        def backward(vector,graph,gradient):
            state['backwards']+=1
            return state['gradient']
        return np.zeros(2),predict,backward,np.ones(2,dtype=bool)
    def directions(*args, **kwargs):
        state['directions'] += 1
        return [dict(structure_id='H2', passed=True)]
    monkeypatch.setattr(integrals, 'prepare_fixed_pool',
                        lambda *args, **kwargs: (Experiment(), {}))
    monkeypatch.setattr(trainer, 'make_mace_predictor', predictor)
    monkeypatch.setattr(graph, 'build_graph', lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow, 'scientific_source_hash', lambda root: 'same-source')
    monkeypatch.setattr(workflow, 'verify_source_imports', lambda root: {})
    monkeypatch.setattr(workflow, 'check_parameter_directions', directions)
    return workflow, state, dict(data=data, candidate=candidate_for(layout), seed=0)


def test_skip_parameter_gradient_checks_records_request_and_replaces_stale_progress(
        tmp_path, mocked_fixed_pool):
    workflow, state, args = mocked_fixed_pool
    output = tmp_path/'run'
    output.mkdir()
    (output/'gate-progress.json').write_text(json.dumps(
        dict(completed=500, total=500, checks=[dict(passed=True)])))
    result = workflow.run_fixed_pool(**args, cache_dir=tmp_path/'cache', output_dir=output,
                                    epochs=1, check_parameter_gradients=False)
    assert result['status'] == 'complete'
    assert state['directions'] == 0 and state['calls'] >= 4
    gate = json.loads((output/'gate.json').read_text())
    assert gate['passed'] and gate['hot_repeat_passed']
    assert gate['parameter_direction_checks'] == []
    assert gate['parameter_direction_checks_status'] == 'skipped_by_request'
    progress = json.loads((output/'gate-progress.json').read_text())
    assert progress['status'] == 'skipped_by_request'
    assert progress['completed'] == 0 and progress['checks'] == []
    metadata = json.loads((output/'metadata.json').read_text())
    assert metadata['check_parameter_gradients'] is False


@pytest.mark.parametrize('failure', ['scf', 'gradient', 'cold_hot'])
def test_skip_parameter_checks_preserves_other_gates(tmp_path, mocked_fixed_pool, failure):
    workflow, state, args = mocked_fixed_pool
    if failure == 'scf':
        state['info']['converged'] = False
    elif failure == 'gradient':
        state['gradient'][0] = np.nan
    else:
        class ChangingEnergy(float):
            def __float__(self):
                return float(state['calls'])
        state['energy'] = ChangingEnergy(-1.)
    with pytest.raises((RuntimeError, AssertionError)):
        workflow.run_fixed_pool(**args, cache_dir=tmp_path/'cache', output_dir=tmp_path/'run',
                                epochs=1, check_parameter_gradients=False)
    assert state['directions'] == 0
    assert not (tmp_path/'run/last.npz').exists()


def test_resume_may_skip_parameter_checks_but_still_requires_source_identity(
        tmp_path, mocked_fixed_pool, monkeypatch):
    workflow, state, args = mocked_fixed_pool
    args.update(cache_dir=tmp_path/'cache', output_dir=tmp_path/'run')
    first = workflow.run_fixed_pool(**args, epochs=1)
    assert state['directions'] == 1
    resumed = workflow.run_fixed_pool(**args, epochs=2, resume=True,
                                     check_parameter_gradients=False)
    assert resumed['status'] == 'complete' and resumed['accepted_epoch'] == 2
    assert first['identity'] == resumed['identity'] and state['directions'] == 1
    monkeypatch.setattr(workflow, 'scientific_source_hash', lambda root: 'changed-source')
    with pytest.raises(ValueError, match='identity differs'):
        workflow.run_fixed_pool(**args, epochs=3, resume=True, check_parameter_gradients=False)


def test_default_parameter_gate_still_rejects_a_failed_mock_check(
        tmp_path, mocked_fixed_pool, monkeypatch):
    workflow, _, args = mocked_fixed_pool
    monkeypatch.setattr(workflow, 'check_parameter_directions',
                        lambda *args, **kwargs: [dict(structure_id='H2', passed=False)])
    with pytest.raises(RuntimeError, match='parameter gradient gate failed'):
        workflow.run_fixed_pool(**args, cache_dir=tmp_path/'cache',
                                output_dir=tmp_path/'run', epochs=1)
    gate = json.loads((tmp_path/'run/gate.json').read_text())
    assert not gate['passed'] and gate['parameter_direction_checks_status'] == 'failed'
    assert not (tmp_path/'run/last.npz').exists()


def test_fixed_pool_cli_can_skip_parameter_gradient_checks(tmp_path, monkeypatch):
    from gradscf.model.nnao.workflows import fixed_pool as workflow
    candidate = tmp_path/'candidate.json'
    candidate.write_text('{}')
    observed = {}
    def run(*args, **kwargs):
        observed.update(kwargs)
        return dict(status='complete')
    monkeypatch.setattr(workflow, 'run_fixed_pool', run)
    assert workflow.main(['--candidate', str(candidate), '--cache-dir', str(tmp_path/'cache'),
                          '--output-dir', str(tmp_path/'run'), '--skip-gradient-checks']) == 0
    assert observed['check_parameter_gradients'] is False


def test_fixed_pool_cli_minibatch_defaults_to_single_pass(tmp_path,monkeypatch):
    from gradscf.model.nnao.workflows import fixed_pool as workflow
    candidate=tmp_path/'candidate.json';candidate.write_text('{}');observed={}
    def run(*args,**kwargs):
        observed.update(kwargs)
        return dict(status='complete')
    monkeypatch.setattr(workflow,'run_fixed_pool',run)
    assert workflow.main(['--candidate',str(candidate),'--cache-dir',str(tmp_path/'cache'),
        '--output-dir',str(tmp_path/'run'),'--batch-size','50'])==0
    assert observed['batch_size']==50 and observed['check_parameter_gradients'] is None


def test_single_pass_workflow_has_no_startup_candidate_or_epoch_replay(tmp_path,mocked_fixed_pool):
    workflow,state,args=mocked_fixed_pool
    result=workflow.run_fixed_pool(**args,cache_dir=tmp_path/'cache',output_dir=tmp_path/'run',
        epochs=2,batch_size=1)
    assert state['calls']==state['forwards']==state['backwards']==2 and state['directions']==0
    assert result['update_step']==2 and result['loss_kind']=='online_pre_update_mean'
    gate=json.loads((tmp_path/'run/gate.json').read_text())
    assert gate['passed'] is None and gate['validation_status']=='not_run_single_pass'
    assert not (tmp_path/'run/best.npz').exists()
    metadata=json.loads((tmp_path/'run/metadata.json').read_text())
    assert metadata['check_parameter_gradients'] is False
    assert metadata['training_policy']=='single_pass'


def test_single_pass_workflow_forwards_energy_stopping_without_extra_calls(tmp_path,mocked_fixed_pool):
    workflow,state,args=mocked_fixed_pool
    result=workflow.run_fixed_pool(**args,cache_dir=tmp_path/'cache',output_dir=tmp_path/'run',
        epochs=30,batch_size=1,early_stop_window=10,early_stop_tolerance=1e-4)
    assert result['status']=='early_stopped' and result['epoch']==11
    assert state['calls']==state['forwards']==state['backwards']==11
    assert state['directions']==0
    metadata=json.loads((tmp_path/'run/metadata.json').read_text())
    assert metadata['identity']['early_stopping']['tolerance_hartree']==1e-4


def test_single_pass_workflow_resume_keeps_identity_and_never_replays(tmp_path,mocked_fixed_pool,monkeypatch):
    workflow,state,args=mocked_fixed_pool
    args.update(cache_dir=tmp_path/'cache',output_dir=tmp_path/'run',epochs=3,batch_size=1)
    workflow.run_fixed_pool(**args,max_updates=1)
    result=workflow.run_fixed_pool(**args,resume=True)
    assert state['calls']==3 and state['directions']==0 and result['epoch']==3
    monkeypatch.setattr(workflow,'scientific_source_hash',lambda root:'changed-source')
    with pytest.raises(ValueError,match='identity differs'):
        workflow.run_fixed_pool(**dict(args,epochs=4),resume=True)
    assert state['calls']==3


def test_single_pass_workflow_rejects_requested_inline_differences(tmp_path,mocked_fixed_pool):
    workflow,state,args=mocked_fixed_pool
    with pytest.raises(ValueError,match='separate validation'):
        workflow.run_fixed_pool(**args,cache_dir=tmp_path/'cache',output_dir=tmp_path/'run',
            epochs=1,batch_size=1,check_parameter_gradients=True)
    assert state['calls']==state['directions']==0


def test_real_mace_single_pass_resume_evaluates_only_the_next_batch(tmp_path,monkeypatch):
    pytest.importorskip('cuequivariance')
    pytest.importorskip('mace_jax')
    from gradscf.model.nnao.training.pool import FreeContractionRHF
    from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
    data=load_references(ids=['H2'])
    layout=prepare_direct_basis('H 0 0 0; H 0 0 .74',basis_family='szp663_direct',core_primitives=6)
    calls=[];original=FreeContractionRHF.evaluate_device
    def evaluate(self,*args,**kwargs):
        calls.append(self.identifier)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(FreeContractionRHF,'evaluate_device',evaluate)
    args=dict(data=data,candidate=candidate_for(layout),cache_dir=tmp_path/'cache',
        output_dir=tmp_path/'run',epochs=2,batch_size=1,seed=0)
    run_fixed_pool(**args,max_updates=1)
    result=run_fixed_pool(**args,resume=True)
    assert calls==['H2','H2']
    assert result['status']=='complete' and result['epoch']==result['update_step']==2
    assert np.isfinite(result['parameters']).all() and np.isfinite(result['gradient']).all()
    assert result['gradient_update_step']==1 and result['parameters_evaluated'] is False
