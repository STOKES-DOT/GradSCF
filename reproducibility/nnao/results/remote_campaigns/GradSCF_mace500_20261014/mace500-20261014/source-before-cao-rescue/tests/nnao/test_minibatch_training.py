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

def test_failed_candidate_does_not_advance_parameters_or_adam(tmp_path):
    x=np.zeros(2);identity={'kind':'failed','batch_size':2};flag=[True]
    opt=Adam(.01,decay_steps=2)
    with pytest.raises(RuntimeError,match='invalid candidate SCF'):
        run_minibatch_training(x,evaluators(flag),opt,tmp_path,identity,epochs=1,batch_size=2)
    saved=load_checkpoint(tmp_path/'last.npz',identity,opt,x)
    np.testing.assert_array_equal(saved['parameters'],x)
    assert saved['update_step']==saved['epoch']==saved['batch_cursor']==0
    assert opt.encode(saved['optimizer'])==opt.encode(opt.init(x))
    assert (tmp_path/'failed-update-000001.npz').exists()

def test_best_uses_full_dataset_not_changing_batches(tmp_path):
    x=np.zeros(1);identity={'kind':'best','batch_size':2};opt=Adam(.01)
    result=run_minibatch_training(x,evaluators(),opt,tmp_path,identity,epochs=2,batch_size=2)
    best=load_checkpoint(tmp_path/'best.npz',identity,opt,x)
    expected=np.mean([f(best['parameters'])[0] for _,f in evaluators()])
    assert best['loss_hartree']==pytest.approx(expected)
    assert best['batch_cursor']==0 and result['update_step']==4
