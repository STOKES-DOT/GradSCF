"""Benchmark boundaries must wait for asynchronous facade result arrays."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path


def test_timer_ready_waits_for_all_facade_output_arrays():
    path = Path(__file__).parents[1]/'tools/libcint_migration/worker.py'
    spec = importlib.util.spec_from_file_location('libcint_benchmark_worker', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    completed = []
    class Pending:
        def __init__(self, name):
            self.name = name
        def block_until_ready(self):
            completed.append(self.name)
            return self
    @dataclass
    class Result:
        energy: object
        t1: object
        t2: object
    class Facade:
        result = Result(Pending('energy'), Pending('t1'), Pending('t2'))
    worker.ready(Facade())
    assert completed == ['energy', 't1', 't2']
