"""Single-pass minibatch Adam with deterministic shuffle and resumable state."""
from pathlib import Path
from numbers import Integral
import time
import numpy as np
from ..runtime import atomic_json
from .trainer import evaluate_batch
from .checkpoints import save_checkpoint,load_checkpoint,save_failure_snapshot

def energy_stop_policy(window, tolerance):
    if window is None and tolerance is None:
        return None
    if (not isinstance(window, Integral) or isinstance(window, bool) or window < 1
            or tolerance is None or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Energy stopping requires a positive integer window and positive finite tolerance.')
    return dict(window=int(window), tolerance_hartree=float(tolerance),
                metric='sum_absolute_changes_of_complete_online_epoch_means')


def epoch_energy_change(history, window):
    rows=history[-(window+1):]
    if (len(rows)!=window+1 or any(not r.get('complete_epoch_coverage') for r in rows)
            or any(b['epoch']!=a['epoch']+1 for a,b in zip(rows,rows[1:]))):
        return None
    values=np.asarray([r['loss_hartree'] for r in rows],dtype=float)
    return float(np.abs(np.diff(values)).sum()) if np.isfinite(values).all() else None


def run_minibatch_training(initial,evaluators,optimizer,output_dir,identity,*,epochs,
                           batch_size,seed=0,resume=False,max_updates=None,
                           early_stop_window=None,early_stop_tolerance=None):
    """Measure each batch once, then update Adam without evaluating the proposal.

    Online epoch means combine pre-update energies at different parameter
    vectors. They do not select a best checkpoint. A saved gradient belongs to
    gradient_update_step, one step before the saved, unevaluated parameters.
    """
    if batch_size<1 or len(evaluators)%batch_size or not evaluators:
        raise ValueError('Use a positive batch size dividing the nonempty dataset.')
    if epochs<1:raise ValueError('epochs must be positive.')
    if max_updates is not None and max_updates<1:raise ValueError('max_updates must be positive.')
    stopping=energy_stop_policy(early_stop_window,early_stop_tolerance)
    if stopping is not None:
        identity=dict(identity,early_stopping=stopping)
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    last=directory/'last.npz'
    if last.exists()!=resume:raise ValueError('Use a fresh output directory or explicitly resume.')
    steps=len(evaluators)//batch_size;started=time.perf_counter();checkpoint=None
    active_parameters=np.asarray(initial);phase='initial';batch_ids=[]
    def evaluate(vector,subset):
        nonlocal active_parameters
        active_parameters=np.asarray(vector)
        def progress(records):
            atomic_json(directory/'batch_progress.json',dict(phase=phase,completed=len(records),total=len(subset),
                structure_id=records[-1]['structure_id'] if records else None))
        return evaluate_batch(vector,subset,progress=progress)
    def write_progress(status):
        current=checkpoint or dict(epoch=0,update_step=0,batch_cursor=0,loss_hartree=None)
        atomic_json(directory/'progress.json',dict(status=status,accepted_epoch=current['epoch'],
            update_step=current['update_step'],next_batch=current['batch_cursor'],updates_per_epoch=steps,
            batch_size=batch_size,target_epochs=epochs,loss_hartree=current['loss_hartree'],
            loss_epoch=current['epoch'],loss_kind=current.get('loss_kind','legacy_full_evaluation'),
            loss_sample_count=current.get('loss_sample_count'),
            parameters_evaluated=current.get('parameters_evaluated'),
            gradient_update_step=current.get('gradient_update_step'),
            early_stopping=stopping,
            energy_change_hartree=epoch_energy_change(current.get('history',[]),stopping['window']) if stopping else None,
            learning_rate=optimizer.learning_rate_at(current['update_step']),
            elapsed_seconds=time.perf_counter()-started))
    def finish_epoch():
        nonlocal checkpoint
        count=checkpoint['online_sample_count']
        loss=checkpoint['online_energy_sum']/count if count else None
        record=dict(epoch=checkpoint['epoch']+1,update_step=checkpoint['update_step'],loss_hartree=loss,
            loss_kind='online_pre_update_mean',sample_count=count,complete_epoch_coverage=count==len(evaluators))
        checkpoint=dict(checkpoint,epoch=record['epoch'],batch_cursor=0,loss_hartree=loss,
            loss_kind=record['loss_kind'],loss_sample_count=count,history=checkpoint['history']+[record],
            online_energy_sum=0.,online_sample_count=0)
        atomic_json(directory/'history.json',checkpoint['history'])
        print(f'Epoch {record["epoch"]}/{epochs}: online mean E {loss} Ha; '
              f'samples {count}/{len(evaluators)}; updates {checkpoint["update_step"]}',flush=True)
    def energy_converged():
        change=epoch_energy_change(checkpoint['history'],stopping['window']) if stopping else None
        return checkpoint['batch_cursor']==0 and change is not None and change<stopping['tolerance_hartree']
    def finish(status):
        write_progress(status)
        atomic_json(directory/'phase.json',dict(phase=status,status=status))
        return dict(checkpoint,status=status)
    try:
        if resume:
            checkpoint=load_checkpoint(last,identity,optimizer,initial)
        else:
            checkpoint=dict(identity=identity,parameters=np.asarray(initial),gradient=None,
                optimizer=optimizer.init(initial),epoch=0,update_step=0,batch_cursor=0,
                loss_hartree=None,history=[],initial_loss_hartree=None,loss_kind='online_pre_update_mean')
        # Legacy partial epochs lack online accumulators. Observe only the
        # remaining batches and record their actual coverage without replay.
        checkpoint.setdefault('online_energy_sum',0.)
        checkpoint.setdefault('online_sample_count',0)
        checkpoint['training_policy']='single_pass'
        write_progress('running')
        if energy_converged():return finish('early_stopped')
        while checkpoint['epoch']<epochs:
            if checkpoint['batch_cursor']==steps:
                finish_epoch();save_checkpoint(last,checkpoint,optimizer);write_progress('running')
                if energy_converged():return finish('early_stopped')
                continue
            if max_updates is not None and checkpoint['update_step']>=max_updates:
                write_progress('update_budget_reached');return dict(checkpoint,status='update_budget_reached')
            epoch=checkpoint['epoch']+1
            order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(evaluators))
            for batch_index in range(checkpoint['batch_cursor'],steps):
                indices=order[batch_index*batch_size:(batch_index+1)*batch_size]
                subset=[evaluators[int(i)] for i in indices];batch_ids=[i for i,_ in subset]
                phase='batch_gradient'
                atomic_json(directory/'phase.json',dict(status='running',phase=phase,epoch=epoch,batch=batch_index+1))
                loss,gradient,_=evaluate(checkpoint['parameters'],subset)
                phase='optimizer_update'
                candidate,state=optimizer.update(checkpoint['parameters'],gradient,checkpoint['optimizer'])
                active_parameters=np.asarray(candidate)
                if not np.isfinite(active_parameters).all():raise RuntimeError('Nonfinite optimizer proposal')
                checkpoint=dict(checkpoint,parameters=active_parameters,gradient=np.asarray(gradient),
                    optimizer=state,update_step=checkpoint['update_step']+1,batch_cursor=batch_index+1,
                    gradient_update_step=checkpoint['update_step'],parameters_evaluated=False,
                    last_batch_loss_hartree=loss,last_batch_ids=batch_ids,
                    online_energy_sum=checkpoint['online_energy_sum']+loss*len(subset),
                    online_sample_count=checkpoint['online_sample_count']+len(subset))
                if batch_index+1==steps:finish_epoch()
                save_checkpoint(last,checkpoint,optimizer);write_progress('running')
                if energy_converged():return finish('early_stopped')
                if max_updates is not None and checkpoint['update_step']>=max_updates:
                    write_progress('update_budget_reached');return dict(checkpoint,status='update_budget_reached')
        return finish('complete')
    except Exception as error:
        accepted_updates=checkpoint['update_step'] if checkpoint else 0
        step=accepted_updates+1 if checkpoint else 0
        save_failure_snapshot(directory/f'failed-update-{step:06d}.npz',active_parameters,error,
            dict(phase=phase,error=str(error),batch_ids=batch_ids,accepted_updates=accepted_updates))
        atomic_json(directory/'phase.json',dict(status='failed',phase=phase,error=str(error),accepted_updates=accepted_updates))
        write_progress('failed')
        raise
