"""Admission ordering and indefinite waits in the XC100 swarm."""
import asyncio
from collections import deque
from threading import Event
from types import SimpleNamespace

import pytest


def test_agent_queue_fills_one_freed_slot_and_requeues_at_the_tail():
    from gradscf.model.autoxc.swarm.queues import AgentQueue

    async def simulate():
        queue = AgentQueue(2)
        entered = [Event() for _ in range(4)]
        release = [Event() for _ in range(4)]

        def work(slot, *, agent_queue_seconds):
            assert agent_queue_seconds >= 0
            entered[slot].set()
            assert release[slot].wait(5)
            if slot == 1:
                raise ValueError('failed job')

        tasks = [asyncio.create_task(queue.run(work, slot)) for slot in range(4)]
        try:
            assert await asyncio.to_thread(entered[0].wait, 3)
            assert await asyncio.to_thread(entered[1].wait, 3)
            assert not entered[2].is_set()
            release[1].set()
            with pytest.raises(ValueError, match='failed job'):
                await tasks[1]
            assert await asyncio.to_thread(entered[2].wait, 3)
            assert not release[0].is_set() and not entered[3].is_set()
            release[2].set()
            assert await asyncio.to_thread(entered[3].wait, 3)
        finally:
            for event in release:
                event.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(simulate())


def test_evaluator_dispatches_fifo_after_a_failed_candidate():
    from gradscf.model.autoxc.swarm.queues import EvaluationQueue

    started = []
    ready = Event()
    release = Event()

    def evaluate(candidate):
        started.append(candidate)
        if candidate == 'a':
            ready.set()
            assert release.wait(5)
            raise ValueError('invalid score')
        return candidate

    with EvaluationQueue(evaluate) as queue:
        a = queue.submit('a')
        assert ready.wait(3)
        b, c = queue.submit('b'), queue.submit('c')
        assert started == ['a']
        release.set()
        with pytest.raises(ValueError, match='invalid score'):
            a.result(timeout=3)
        assert b.result(timeout=3)[0] == 'b'
        assert c.result(timeout=3)[0] == 'c'
    assert started == ['a', 'b', 'c']


def test_no_proposal_deadline_avoids_constructing_a_timer(tmp_path, monkeypatch):
    from gradscf.model.autoxc import codex

    def timer(*args):
        pytest.fail('created a timer with an unbounded proposal')

    monkeypatch.setattr(codex, 'Timer', timer)
    designer = codex.CodexDesigner(cwd=tmp_path, timeout=None)
    designer._active = True
    designer._thread = SimpleNamespace(run=lambda *args, **options: SimpleNamespace(final_response='{}'))
    assert designer.propose({}) == '{}'


def test_remote_unbounded_wait_still_checks_job_state(tmp_path, monkeypatch):
    from gradscf.model.autoxc import Candidate
    from gradscf.model.autoxc.workflows.xc100 import distributed

    c = Candidate('n')
    ready = {'baseline_count': 100, 'job_id': '42', 'source': 'test',
             'configuration': {'fingerprint': 'sample'}}
    events = deque([None, None, {'digest': c.digest, 'fingerprint': 'sample',
                    'evaluation': {'valid': True, 'reward': .1}}])
    ticks = iter(range(0, 100_000, 4_000))
    checked = []

    def read(self, relative):
        if relative == 'ready.json':
            return ready
        if relative == 'status.json':
            return {'phase': 'ready', 'job_id': '42'}
        return events.popleft()

    with monkeypatch.context() as patch:
        patch.setattr(distributed, 'source_digest', lambda: 'test')
        patch.setattr(distributed, 'time', SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None))
        patch.setattr(distributed.RemoteEvaluator, '_ssh', lambda *args, **kwargs: 'RUNNING')
        patch.setattr(distributed.RemoteEvaluator, '_transport', lambda *args: SimpleNamespace(returncode=0))
        patch.setattr(distributed.RemoteEvaluator, 'read', read)
        patch.setattr(distributed.RemoteEvaluator, '_running', lambda self, job: checked.append(job))
        worker = distributed.RemoteEvaluator.wait_ready('host', '/root', receipt_directory=tmp_path,
                                                        timeout=None, job_id='42')
        assert worker.timeout is None
        assert worker.evaluate(c).reward == .1
    assert len(checked) > 1


def test_no_deadline_worker_rejects_a_fit_wall_time(tmp_path):
    from gradscf.model.autoxc.workflows.xc100.campaign import Campaign

    worker = Campaign(tmp_path, candidate_mode='standalone_potential', unbounded_waits=True)
    assert worker.policy['unbounded_waits']
    assert worker.fit_options['max_seconds'] is None
    with pytest.raises(ValueError, match='max_seconds'):
        Campaign(tmp_path, unbounded_waits=True, max_seconds=1800.)
