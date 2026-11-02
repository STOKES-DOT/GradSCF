"""Operational guard: only a timed-out valid training run may auto-resume."""
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

SCRIPT = Path(__file__).with_name('resume_guard.py')


def check(run, log, job='93538'):
    assert SCRIPT.is_file(), 'Timeout-only resume guard has not been implemented.'
    return runpy.run_path(str(SCRIPT))['resume_decision'](run, log, job, 1000)


def saved_run(tmp_path, epoch=500):
    run = tmp_path / 'training'; run.mkdir()
    for name in ('last.npz', 'best.npz', 'best_validation.npz'):
        np.savez(run / name, metadata=np.asarray(json.dumps(
            dict(epoch=epoch, identity={'run': 'same-source-data-and-config'}))))
    log = tmp_path / 'train-93538.out'
    log.write_text('slurmstepd: error: *** JOB 93538 ON n001 CANCELLED DUE TO TIME LIMIT ***\n')
    return run, log


def test_confirmed_timeout_with_complete_checkpoint_set_can_resume(tmp_path):
    run, log = saved_run(tmp_path)
    result = check(run, log)
    assert result['action'] == 'resume' and result['epoch'] == 500


def test_completed_target_does_not_restart(tmp_path):
    run, log = saved_run(tmp_path, epoch=1000)
    assert check(run, log)['action'] == 'skip'


def test_scientific_failure_does_not_auto_retry(tmp_path):
    run, log = saved_run(tmp_path)
    (run / 'summary.json').write_text(json.dumps({'status': 'failed', 'error': 'Nonfinite gradient'}))
    assert check(run, log)['action'] == 'skip'


def test_manual_cancel_does_not_restart(tmp_path):
    run, log = saved_run(tmp_path)
    log.write_text('slurmstepd: error: *** JOB 93538 ON n001 CANCELLED ***\n')
    assert check(run, log)['action'] == 'skip'


def test_timeout_for_another_job_is_not_authorization(tmp_path):
    run, log = saved_run(tmp_path)
    log.write_text('*** JOB 99999 ON n001 CANCELLED DUE TO TIME LIMIT ***\n')
    assert check(run, log)['action'] == 'skip'


@pytest.mark.parametrize('name', ['best.npz', 'best_validation.npz'])
def test_missing_required_best_checkpoint_blocks_resume(tmp_path, name):
    run, log = saved_run(tmp_path)
    (run / name).unlink()
    assert check(run, log)['action'] == 'blocked'


def test_mixed_checkpoint_identity_blocks_resume(tmp_path):
    run, log = saved_run(tmp_path)
    np.savez(run / 'best.npz', metadata=np.asarray(json.dumps(
        dict(epoch=499, identity={'run': 'different-data'}))))
    assert check(run, log)['action'] == 'blocked'
