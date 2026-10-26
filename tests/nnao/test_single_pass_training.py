"""One measured energy/gradient per batch, with resumable Adam state."""
import json
import numpy as np
import pytest
pytest.importorskip('optax')

from gradscf.model.nnao.training.trainer import Adam
from gradscf.model.nnao.training.minibatch import run_minibatch_training
from gradscf.model.nnao.training.checkpoints import load_checkpoint,save_checkpoint


def problems(calls,fail=None):
    def make(i):
        def evaluate(x):
            calls.append((i,np.array(x,copy=True)))
            if fail is not None and fail[0]:raise RuntimeError('SCF did not converge')
            return float(np.sum((x-i-1)**2)),2*(x-i-1),dict(converged=True,stable=True,
                orbital_residual=0.,fixed_point_residual=0.,min_overlap_eigenvalue=1.,reconstruction_error=0.)
        return str(i),evaluate
    return [make(i) for i in range(4)]


def test_exactly_one_forward_backward_per_sample_and_epoch(tmp_path):
    calls=[];x=np.zeros(2);opt=Adam(.01);identity={'kind':'one-pass'}
    result=run_minibatch_training(x,problems(calls),opt,tmp_path,identity,epochs=2,batch_size=2,seed=7)
    assert len(calls)==8
    assert result['epoch']==2 and result['update_step']==4
    assert not (tmp_path/'best.npz').exists() and not (tmp_path/'last_full_epoch.npz').exists()
    assert result['loss_kind']=='online_pre_update_mean'
    assert result['gradient_update_step']==3 and result['parameters_evaluated'] is False
    for epoch in [1,2]:
        group=calls[4*(epoch-1):4*epoch]
        assert sorted(i for i,_ in group)==[0,1,2,3]
        expected=np.mean([np.sum((v-i-1)**2) for i,v in group])
        row=result['history'][epoch-1]
        assert row['loss_hartree']==pytest.approx(expected)
        assert row['sample_count']==4 and row['complete_epoch_coverage'] is True


def test_single_pass_resume_never_replays_samples_and_preserves_adam(tmp_path):
    x=np.zeros(2);identity={'kind':'resume'};calls=[]
    def opt():return Adam(.01,decay_steps=3)
    whole=run_minibatch_training(x,problems([]),opt(),tmp_path/'whole',identity,epochs=3,batch_size=2,seed=5)
    part=run_minibatch_training(x,problems(calls),opt(),tmp_path/'split',identity,
        epochs=3,batch_size=2,seed=5,max_updates=3)
    assert part['epoch']==1 and part['batch_cursor']==1 and len(calls)==6
    result=run_minibatch_training(x,problems(calls),opt(),tmp_path/'split',identity,
        epochs=3,batch_size=2,seed=5,resume=True)
    assert len(calls)==12 and result['history']==whole['history']
    np.testing.assert_array_equal(result['parameters'],whole['parameters'])
    assert opt().encode(result['optimizer'])==opt().encode(whole['optimizer'])


@pytest.mark.parametrize('budget',[1,2])
def test_single_pass_failure_preserves_last_update_and_snapshots_its_input(tmp_path,budget):
    x=np.zeros(2);identity={'kind':'failure'};opt=Adam(.01);calls=[]
    before=run_minibatch_training(x,problems(calls),opt,tmp_path,identity,
        epochs=2,batch_size=2,max_updates=budget)
    content=(tmp_path/'last.npz').read_bytes();calls.clear()
    with pytest.raises(RuntimeError,match='SCF did not converge'):
        run_minibatch_training(x,problems(calls,[True]),opt,tmp_path,identity,
            epochs=2,batch_size=2,resume=True)
    assert len(calls)==1 and (tmp_path/'last.npz').read_bytes()==content
    with np.load(tmp_path/f'failed-update-{budget+1:06d}.npz') as a:
        np.testing.assert_array_equal(a['parameters'],before['parameters'])
    progress=json.loads((tmp_path/'progress.json').read_text())
    assert progress['status']=='failed' and progress['update_step']==budget


def test_first_failed_forward_cannot_create_a_successful_checkpoint(tmp_path):
    with pytest.raises(RuntimeError,match='SCF did not converge'):
        run_minibatch_training(np.zeros(2),problems([],[True]),Adam(.01),tmp_path,{},epochs=1,batch_size=2)
    assert not (tmp_path/'last.npz').exists()
    assert json.loads((tmp_path/'progress.json').read_text())['status']=='failed'


def legacy_checkpoint(path,identity,opt,x,*,cursor):
    checkpoint=dict(identity=identity,parameters=x.copy(),gradient=np.ones_like(x),optimizer=opt.init(x),
        epoch=2,update_step=4+cursor,batch_cursor=cursor,loss_hartree=-12.,
        initial_loss_hartree=-10.,history=[dict(epoch=2,update_step=4,loss_hartree=-12.)])
    save_checkpoint(path,checkpoint,opt)


@pytest.mark.parametrize('cursor,observed',[(1,2),(2,0)])
def test_legacy_resume_does_not_invent_a_full_epoch_loss(tmp_path,cursor,observed):
    x=np.full(2,.1);identity={'kind':'legacy'};opt=Adam(.01)
    for name in ['last.npz','best.npz','last_full_epoch.npz']:
        legacy_checkpoint(tmp_path/name,identity,opt,x,cursor=cursor)
    old={n:(tmp_path/n).read_bytes() for n in ['best.npz','last_full_epoch.npz']}
    calls=[]
    result=run_minibatch_training(x,problems(calls),opt,tmp_path,identity,epochs=3,batch_size=2,resume=True)
    assert len(calls)==observed and result['epoch']==3
    row=result['history'][-1]
    assert row['sample_count']==observed and row['complete_epoch_coverage'] is False
    if not observed:assert row['loss_hartree'] is None
    for name,content in old.items():assert (tmp_path/name).read_bytes()==content


def test_nonfinite_optimizer_proposal_does_not_commit(tmp_path):
    x=np.zeros(2);opt=Adam(.01);identity={};calls=[]
    run_minibatch_training(x,problems(calls),opt,tmp_path,identity,epochs=2,batch_size=2,max_updates=1)
    content=(tmp_path/'last.npz').read_bytes()
    class BrokenAdam(Adam):
        def update(self,*args):
            x,state=super().update(*args)
            return np.full_like(x,np.inf),state
    with pytest.raises(RuntimeError,match='Nonfinite optimizer proposal'):
        run_minibatch_training(x,problems(calls),BrokenAdam(.01),tmp_path,identity,
            epochs=2,batch_size=2,resume=True)
    assert (tmp_path/'last.npz').read_bytes()==content
