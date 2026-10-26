"""S9 harness metadata and synchronization contracts; no numerical calculation."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _module():
    path = ROOT/'tools/libcint_migration/probes.py'
    assert path.is_file(), 'The thin S9 consumer probe catalogue/adapter is missing'
    spec = importlib.util.spec_from_file_location('s9_consumer_probes', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s9_counter_observes_completed_outputs_after_reset():
    module = _module()
    events = []
    class Pending:
        def block_until_ready(self):
            events.append('ready')
            return self
    pending = Pending()
    def execute():
        events.append('execute')
        return pending
    def snapshot():
        events.append('snapshot')
        return dict(schema='libcint-call-counts-v1', enabled=True,
                    build_kind='profile', library_sha256='fixture-only',
                    families={'eri4c': {'evaluations': 3, 'cache_queries': 2}})
    counter = module.CounterAdapter(lambda: events.append('reset'), snapshot)
    result, counts = module.counted_call(execute,
        ready=lambda value: value.block_until_ready(), counter=counter)
    assert result is pending
    assert events == ['reset', 'execute', 'ready', 'snapshot']
    assert counts['families']['eri4c']['evaluations'] == 3


def test_s9_disabled_or_unknown_counter_is_not_a_zero_measurement():
    module = _module()
    for payload in (dict(schema='libcint-call-counts-v1', enabled=False),
                    dict(schema='unrecognized', enabled=True)):
        counter = module.CounterAdapter(lambda: None, lambda: payload)
        with pytest.raises(RuntimeError, match='instrument|schema|enabled'):
            counter.snapshot()


def test_s9_catalogue_names_every_required_family_and_existing_gate():
    module = _module()
    groups = {row['group'] for row in module.CASES.values()}
    assert groups == {'scf', 'spin', 'excitation', 'ci_fci', 'mp2_ri', 'recursive_mp',
                      'cc_eom', 'gw_bse', 'ofdft', 'nnao', 'periodic_na'}
    for name, row in module.CASES.items():
        assert row['scope'] and row['outputs'], name
        assert row['driver'] in {'worker', 'pytest', 'nnao_step'}
        for node in row.get('nodes', ()):
            assert (ROOT/node.split('::', 1)[0]).is_file(), (name, node)
    periodic = [row for row in module.CASES.values() if row['group'] == 'periodic_na']
    assert periodic and all(not row['native_applicable'] for row in periodic)


def test_gpu_controller_preserves_cpu_libcint_platform(monkeypatch, tmp_path):
    path = ROOT/'tools/libcint_migration/run_cases.py'
    spec = importlib.util.spec_from_file_location('s9_controller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = {}
    class Completed:
        pid = 987654321
        def __init__(self, *args, env, **kwargs):
            observed.update(env)
        def poll(self):
            return 0
        def wait(self, **kwargs):
            return 0
    monkeypatch.setattr(module.subprocess, 'Popen', Completed)
    monkeypatch.setattr(module.os, 'sched_getaffinity', lambda _: {0})
    module.run(dict(case_id='gpu-contract', engine='gradscf', threads=1,
                    device='gpu'), tmp_path, deadline=module.time.time()+10)
    assert set(observed['JAX_PLATFORMS'].split(',')) == {'cpu', 'cuda'}
