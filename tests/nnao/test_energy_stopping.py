"""Epoch stopping uses complete online means without extra SCF evaluations."""
import json
import numpy as np
import pytest

pytest.importorskip('optax')
from gradscf.model.nnao.training.trainer import Adam
from gradscf.model.nnao.training.minibatch import run_minibatch_training


def problems(calls, values):
    def evaluate(x):
        epoch = len(calls)//4
        calls.append(epoch)
        return values[min(epoch, len(values)-1)], np.zeros_like(x), dict(
            converged=True, stable=True, orbital_residual=0., fixed_point_residual=0.,
            min_overlap_eigenvalue=1., reconstruction_error=0.)
    return [(str(i), evaluate) for i in range(4)]


def run(path, calls, values, **kwargs):
    return run_minibatch_training(np.zeros(1), problems(calls, values), Adam(.001),
        path, {'kind': 'energy-stop'}, epochs=30, batch_size=2,
        early_stop_window=10, early_stop_tolerance=1e-4, **kwargs)


def test_flat_energy_stops_after_ten_changes_without_replaying_samples(tmp_path):
    calls=[]
    result=run(tmp_path, calls, [-1.])
    assert result['status']=='early_stopped' and result['epoch']==11
    assert result['update_step']==22 and len(calls)==44
    progress=json.loads((tmp_path/'progress.json').read_text())
    assert progress['status']=='early_stopped'
    assert progress['energy_change_hartree']==0.
    assert result['parameters_evaluated'] is False


def test_opposite_energy_changes_do_not_cancel_in_stopping_window(tmp_path):
    result=run(tmp_path, [], [-1.+.001*(i%2) for i in range(30)])
    assert result['status']=='complete' and result['epoch']==30


def test_stopping_history_and_adam_survive_resume(tmp_path):
    calls=[]
    partial=run(tmp_path, calls, [-1.], max_updates=9)
    assert partial['epoch']==4 and partial['batch_cursor']==1
    result=run(tmp_path, calls, [-1.], resume=True)
    assert result['status']=='early_stopped' and result['epoch']==11
    assert len(calls)==44
    before=len(calls)
    replay=run(tmp_path, calls, [-1.], resume=True)
    assert replay['status']=='early_stopped' and len(calls)==before


def test_threshold_is_strict_and_applies_to_average_energy(tmp_path):
    calls=[]
    result=run_minibatch_training(np.zeros(1), problems(calls,[0.,.125,.25,.375]),
        Adam(.001), tmp_path, {}, epochs=4, batch_size=2,
        early_stop_window=2, early_stop_tolerance=.25)
    assert result['status']=='complete' and result['epoch']==4


def test_resume_cannot_silently_change_stopping_configuration(tmp_path):
    run(tmp_path, [], [-1.], max_updates=2)
    before=(tmp_path/'last.npz').read_bytes()
    with pytest.raises(ValueError, match='identity differs'):
        run_minibatch_training(np.zeros(1), problems([],[-1.]), Adam(.001), tmp_path,
            {'kind':'energy-stop'}, epochs=30, batch_size=2, resume=True,
            early_stop_window=10, early_stop_tolerance=2e-4)
    assert (tmp_path/'last.npz').read_bytes()==before


@pytest.mark.parametrize('window,tolerance', [(0,1e-4),(10,0.),(10,float('nan')),
                                            (10,None),(None,1e-4),(True,1e-4)])
def test_invalid_stopping_configuration_fails_before_evaluation(tmp_path,window,tolerance):
    calls=[]
    with pytest.raises(ValueError):
        run_minibatch_training(np.zeros(1), problems(calls,[-1.]), Adam(.001), tmp_path,
            {}, epochs=30, batch_size=2, early_stop_window=window,
            early_stop_tolerance=tolerance)
    assert not calls and not (tmp_path/'last.npz').exists()
