import json
import numpy as np
import pytest
pytest.importorskip('optax')
from gradscf.model.nnao.training.trainer import Adam
from gradscf.model.nnao.training.minibatch import run_minibatch_training
from gradscf.model.nnao.training.checkpoints import load_checkpoint

def evaluators(fail=None):
    def make(i):
        def evaluate(x):
            x=np.asarray(x)
            if fail is not None and fail[0] and np.any(x!=0):raise RuntimeError('invalid candidate SCF')
            return float(((x-(i+1))**2).sum()),2*(x-(i+1)),dict(converged=True,stable=True,
                orbital_residual=0.,min_overlap_eigenvalue=1.,reconstruction_error=0.,fixed_point_residual=0.)
        return str(i),evaluate
    return [make(i) for i in range(4)]

def test_staircase_uses_accepted_update_count():
    opt=Adam(1e-3,decay_steps=1000,decay_factor=.5)
    assert opt.learning_rate_at(0)==pytest.approx(1e-3)
    assert opt.learning_rate_at(999)==pytest.approx(1e-3)
    assert opt.learning_rate_at(1000)==pytest.approx(5e-4)
    assert opt.learning_rate_at(9000)==pytest.approx(1.953125e-6)

def test_minibatch_restart_matches_uninterrupted(tmp_path):
    x=np.zeros(2);identity={'kind':'quadratic','batch_size':2,'seed':5}
    def opt():return Adam(.01,decay_steps=4,decay_factor=.5)
    a=run_minibatch_training(x,evaluators(),opt(),tmp_path/'full',identity,epochs=3,batch_size=2,seed=5)
    b=run_minibatch_training(x,evaluators(),opt(),tmp_path/'split',identity,epochs=3,batch_size=2,seed=5,max_updates=3)
    assert b['epoch']==1 and b['update_step']==3 and b['batch_cursor']==1
    b=run_minibatch_training(x,evaluators(),opt(),tmp_path/'split',identity,epochs=3,batch_size=2,seed=5,resume=True)
    assert a['epoch']==b['epoch']==3 and a['update_step']==b['update_step']==6
    np.testing.assert_array_equal(a['parameters'],b['parameters'])
    for directory in ['full','split']:
        saved=load_checkpoint(tmp_path/directory/'last.npz',identity,opt(),x)
        assert saved['update_step']==6 and saved['batch_cursor']==0

def test_next_failed_batch_does_not_advance_parameters_or_adam(tmp_path):
    x=np.zeros(2);identity={'kind':'failed','batch_size':2};flag=[True]
    opt=Adam(.01,decay_steps=2)
    with pytest.raises(RuntimeError,match='invalid candidate SCF'):
        run_minibatch_training(x,evaluators(flag),opt,tmp_path,identity,epochs=1,batch_size=2)
    saved=load_checkpoint(tmp_path/'last.npz',identity,opt,x)
    assert saved['update_step']==saved['batch_cursor']==1 and saved['epoch']==0
    ids=[int(i) for i in saved['last_batch_ids']]
    candidate,state=opt.update(x,np.mean([evaluators()[i][1](x)[1] for i in ids],axis=0),opt.init(x))
    np.testing.assert_array_equal(saved['parameters'],candidate)
    assert opt.encode(saved['optimizer'])==opt.encode(state)
    with np.load(tmp_path/'failed-update-000002.npz',allow_pickle=False) as failure:
        np.testing.assert_allclose(failure['parameters'],saved['parameters'])
    progress=json.loads((tmp_path/'progress.json').read_text())
    assert progress['status']=='failed' and progress['update_step']==1

@pytest.mark.parametrize('max_updates,phase,cursor',[
    (None,'batch_gradient',0),
    (1,'batch_gradient',1),
    (2,'batch_gradient',0),
])
def test_resumed_failure_snapshots_active_parameters_and_keeps_accepted_state(tmp_path,max_updates,phase,cursor):
    x=np.zeros(2);identity={'kind':'resume-failure','batch_size':2};opt=Adam(.01)
    accepted=run_minibatch_training(x,evaluators(),opt,tmp_path,identity,epochs=1,batch_size=2,
        max_updates=max_updates)
    assert accepted['batch_cursor']==cursor and np.any(accepted['parameters']!=x)
    before={'last.npz':(tmp_path/'last.npz').read_bytes()}
    with pytest.raises(RuntimeError,match='invalid candidate SCF'):
        run_minibatch_training(x,evaluators([True]),opt,tmp_path,identity,epochs=2,batch_size=2,resume=True)
    path=tmp_path/f'failed-update-{accepted["update_step"]+1:06d}.npz'
    with np.load(path,allow_pickle=False) as failure:
        np.testing.assert_array_equal(failure['parameters'],accepted['parameters'])
        assert json.loads(str(failure['metadata'].item()))['phase']==phase
    for name,content in before.items():assert (tmp_path/name).read_bytes()==content
    progress=json.loads((tmp_path/'progress.json').read_text())
    assert progress['status']=='failed'
    assert progress['accepted_epoch']==accepted['epoch']
    assert progress['update_step']==accepted['update_step'] and progress['next_batch']==cursor
    assert progress['loss_hartree']==accepted['loss_hartree']

def test_initial_failure_records_failed_progress_without_an_accepted_checkpoint(tmp_path):
    x=np.ones(2);identity={'kind':'initial-failure','batch_size':2};opt=Adam(.01)
    with pytest.raises(RuntimeError,match='invalid candidate SCF'):
        run_minibatch_training(x,evaluators([True]),opt,tmp_path,identity,epochs=1,batch_size=2)
    with np.load(tmp_path/'failed-update-000001.npz',allow_pickle=False) as failure:
        np.testing.assert_array_equal(failure['parameters'],x)
    progress=json.loads((tmp_path/'progress.json').read_text())
    assert progress['status']=='failed' and progress['update_step']==0
    assert not (tmp_path/'last.npz').exists()

def test_online_training_does_not_claim_a_best_full_dataset_checkpoint(tmp_path):
    x=np.zeros(1);identity={'kind':'best','batch_size':2};opt=Adam(.01)
    result=run_minibatch_training(x,evaluators(),opt,tmp_path,identity,epochs=2,batch_size=2)
    expected=np.mean([f(result['parameters'])[0] for _,f in evaluators()])
    assert result['loss_hartree']!=pytest.approx(expected)
    assert not (tmp_path/'best.npz').exists()
    assert result['batch_cursor']==0 and result['update_step']==4
