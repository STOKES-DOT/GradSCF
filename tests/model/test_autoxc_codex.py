"""The thin official-SDK adapter is tested without authentication/model calls."""
import sys
import json
from types import SimpleNamespace

import pytest


def test_codex_uses_structured_output_and_reuses_a_thread(monkeypatch, tmp_path):
    from gradscf.model.autoxc import CodexDesigner
    calls = []
    class Client:
        def __init__(self, config): calls.append(('config', config))
        def thread_start(self, **kw):
            calls.append(('start', kw))
            return SimpleNamespace(id='thread-1', run=lambda *a, **kw: invoke(a, kw))
        def thread_resume(self, identifier, **kw):
            calls.append(('resume', identifier))
            return SimpleNamespace(id=identifier, run=lambda *a, **kw: invoke(a, kw))
        def close(self): calls.append(('close',))
    def invoke(args, kwargs):
        calls.append(('turn', kwargs))
        return SimpleNamespace(final_response='{"formula":"n","kind":"potential","parameters":[],"hypothesis":"test"}')
    sdk = SimpleNamespace(Codex=Client, CodexConfig=lambda **kw: kw,
                          Sandbox=SimpleNamespace(read_only='read-only'))
    monkeypatch.setitem(sys.modules, 'openai_codex', sdk)
    with CodexDesigner(cwd=tmp_path, model='configured-model') as designer:
        assert json.loads(designer.propose({'context': 'test'}))['formula'] == 'n'
        designer.propose({'context': 'feedback'})
        assert designer.session_id == 'thread-1'
    assert len([c for c in calls if c[0] == 'start']) == 1
    assert all('output_schema' in c[1] for c in calls if c[0] == 'turn')
    assert next(c[1] for c in calls if c[0] == 'start')['sandbox'] == 'read-only'
    assert calls[-1][0] == 'close'


def test_codex_resumes_an_explicit_saved_thread(monkeypatch, tmp_path):
    from gradscf.model.autoxc import CodexDesigner
    seen = []
    sdk = SimpleNamespace(Codex=lambda config: SimpleNamespace(
        thread_resume=lambda identifier, **kw: (seen.append(identifier) or SimpleNamespace(id=identifier)),
        close=lambda: None), CodexConfig=lambda **kw: kw,
        Sandbox=SimpleNamespace(read_only='read-only'))
    monkeypatch.setitem(sys.modules, 'openai_codex', sdk)
    with CodexDesigner(cwd=tmp_path, session_id='saved') as designer:
        assert designer.session_id == 'saved'
    assert seen == ['saved']


def test_designer_can_require_potential_only_without_mutating_default_schema(monkeypatch,tmp_path):
    from gradscf.model.autoxc import CodexDesigner
    from gradscf.model.autoxc.candidate import PROPOSAL_SCHEMA
    seen=[]
    def run(*args,**kwargs):
        seen.append(kwargs['output_schema'])
        return SimpleNamespace(final_response='{}')
    sdk=SimpleNamespace(Codex=lambda config:SimpleNamespace(
        thread_start=lambda **kwargs:SimpleNamespace(id='potential',run=run),close=lambda:None),
        CodexConfig=lambda **kwargs:kwargs,Sandbox=SimpleNamespace(read_only='read-only'))
    monkeypatch.setitem(sys.modules,'openai_codex',sdk)
    with CodexDesigner(cwd=tmp_path,kind='potential') as designer:designer.propose({})
    assert seen[0]['properties']['kind']['enum']==['potential']
    assert PROPOSAL_SCHEMA['properties']['kind']['enum']==['energy','potential']


def test_codex_timeout_closes_the_runtime_without_signals(monkeypatch, tmp_path):
    import threading
    from gradscf.model.autoxc import CodexDesigner
    closed = threading.Event()
    def blocked(*args, **kwargs):
        assert closed.wait(1.), 'test SDK did not receive cancellation'
        raise RuntimeError('transport closed')
    sdk = SimpleNamespace(Codex=lambda config: SimpleNamespace(
        thread_start=lambda **kw: SimpleNamespace(id='timed', run=blocked),
        close=closed.set), CodexConfig=lambda **kw: kw,
        Sandbox=SimpleNamespace(read_only='read-only'))
    monkeypatch.setitem(sys.modules, 'openai_codex', sdk)
    with CodexDesigner(cwd=tmp_path, timeout=.02) as designer:
        with pytest.raises(TimeoutError):
            designer.propose({'context': 'timeout'})
        assert designer.session_id == 'timed'


def test_sdk_schema_requires_the_definitions_array_without_changing_parser_optional_fields(monkeypatch,tmp_path):
    from gradscf.model.autoxc import CodexDesigner
    from gradscf.model.autoxc.candidate import PROPOSAL_SCHEMA
    seen=[]
    sdk=SimpleNamespace(Codex=lambda config:SimpleNamespace(
        thread_start=lambda **kwargs:SimpleNamespace(id='open',run=lambda *a,**kw:
            (seen.append(kw['output_schema']) or SimpleNamespace(final_response='{}'))),close=lambda:None),
        CodexConfig=lambda **kwargs:kwargs,Sandbox=SimpleNamespace(read_only='read-only'))
    monkeypatch.setitem(sys.modules,'openai_codex',sdk)
    with CodexDesigner(cwd=tmp_path,kind='potential') as designer:designer.propose({})
    assert 'definitions' in seen[0]['required']
    assert 'definitions' not in PROPOSAL_SCHEMA['required']
