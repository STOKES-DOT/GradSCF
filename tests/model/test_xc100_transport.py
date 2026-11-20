"""Transient SSH failures must not discard a running numerical campaign."""
import subprocess

import pytest

from gradscf.model.autoxc.workflows.xc100.distributed import RemoteEvaluator


def client(tmp_path):
    return RemoteEvaluator('jy_gpu_1', '/project', {}, receipt_directory=tmp_path)


@pytest.mark.parametrize('failure', [
    subprocess.CompletedProcess([], 255, '', 'Connection refused'),
    subprocess.TimeoutExpired('ssh', 90),
])
def test_read_recovers_from_a_transient_connection_failure(tmp_path, monkeypatch, failure):
    replies = iter([failure, subprocess.CompletedProcess([], 0, '{"phase":"ready"}', '')])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('time.sleep', lambda _: None)
    assert client(tmp_path).read('status.json') == {'phase': 'ready'}
    assert len(calls) == 2


def test_upload_recovers_without_changing_the_request(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 255 if len(calls) == 1 else 0,
                                           '', 'Connection reset' if len(calls) == 1 else '')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('time.sleep', lambda _: None)
    assert client(tmp_path)._transport(['scp', 'request.json', 'jy_gpu_1:/project/inbox/request.tmp']).returncode == 0
    assert calls[0] == calls[1]


def test_transient_connection_retries_have_a_finite_limit(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 255, '', 'Connection refused')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('time.sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='8 attempts'):
        client(tmp_path).read('status.json')
    assert len(calls) == 8


@pytest.mark.parametrize('code,error', [(2, 'Remote command failed'), (255, 'Permission denied (publickey).')])
def test_permanent_errors_are_not_retried(tmp_path, monkeypatch, code, error):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, code, '', error)
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(RuntimeError, match=error.split()[0]):
        client(tmp_path).read('status.json')
    assert len(calls) == 1


def test_missing_file_is_not_a_transport_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess([], 1, '', ''))
    assert client(tmp_path).read('ready.json') is None


def test_replayed_publication_preserves_an_existing_request(tmp_path, monkeypatch):
    from gradscf.model.autoxc import Candidate
    from gradscf.model.autoxc.history import source_digest
    ready = {'baseline_count': 100, 'source': source_digest(),
             'configuration': {'fingerprint': 'frozen'}}
    evaluator = RemoteEvaluator('jy_gpu_1', '/project', ready, receipt_directory=tmp_path)
    candidate = Candidate('n')
    destination = tmp_path/'inbox'
    destination.mkdir()
    existing = destination/(candidate.digest+'.json')
    existing.write_text('original request')
    def ssh(command, **kwargs):
        # Run the actual publication command locally after replacing the root.
        translated = command.replace('/project', str(tmp_path))
        subprocess.run(translated, shell=True, check=True, capture_output=True)
        return ''
    monkeypatch.setattr(evaluator, '_ssh', ssh)
    monkeypatch.setattr(evaluator, '_transport', lambda command: subprocess.CompletedProcess(command, 0))
    monkeypatch.setattr(evaluator, 'read', lambda name: {
        'digest': candidate.digest, 'fingerprint': 'frozen',
        'evaluation': {'reward': 0.1, 'valid': True}})
    assert evaluator.evaluate(candidate).reward == 0.1
    assert existing.read_text() == 'original request'
