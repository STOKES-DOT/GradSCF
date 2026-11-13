"""Process exits must not interrupt the Linux process-group RSS sampler."""
import errno
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def sampler(monkeypatch, tmp_path):
    path = Path(__file__).parents[1]/"benchmarks/probes/tddft_perf/run.py"
    spec = importlib.util.spec_from_file_location("tddft_perf_run", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    proc = tmp_path/"proc"
    proc.mkdir()
    monkeypatch.setattr(runner, "Path", lambda path: proc if path == "/proc" else Path(path))
    monkeypatch.setattr(runner.os, "sysconf", lambda name: 4096)
    return runner, proc


def write_stat(proc, pid, pgrp, rss_pages):
    fields = ["0"]*22
    fields[0], fields[2], fields[21] = "S", str(pgrp), str(rss_pages)
    folder = proc/str(pid)
    folder.mkdir()
    (folder/"stat").write_text(f"{pid} (python worker) "+" ".join(fields))


def test_rss_group_survives_process_exit_during_glob_enumeration(sampler, monkeypatch):
    runner, proc = sampler
    write_stat(proc, 123, 123, 7)
    original_glob = Path.glob

    def disappearing_glob(path, pattern):
        if path == proc:
            yield proc/"123"/"stat"
            # Recursive glob descends into a PID directory after it disappears.
            raise ProcessLookupError(errno.ESRCH, "No such process")
        yield from original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", disappearing_glob)
    assert runner.rss_group(123) == 7*4096


@pytest.mark.parametrize("error", [
    FileNotFoundError(errno.ENOENT, "No such file or directory"),
    ProcessLookupError(errno.ESRCH, "No such process"),
    PermissionError(errno.EACCES, "Permission denied"),
])
def test_rss_group_skips_unreadable_process_stat(sampler, monkeypatch, error):
    runner, proc = sampler
    write_stat(proc, 123, 123, 7)
    write_stat(proc, 124, 123, 11)
    original_read_text = Path.read_text

    def disappearing_stat(path, *args, **kwargs):
        if path == proc/"124"/"stat":
            raise error
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", disappearing_stat)
    assert runner.rss_group(123) == 7*4096


def test_rss_group_counts_only_numeric_members_of_requested_group(sampler):
    runner, proc = sampler
    write_stat(proc, 123, 123, 7)
    write_stat(proc, 124, 123, 11)
    write_stat(proc, 125, 999, 101)
    write_stat(proc, "self", 123, 1000)
    assert runner.rss_group(123) == (7+11)*4096
