"""Private opt-in call counts; never substitute them for clean timing/RSS.

Use a separate source/library build with ``build --diagnostics``. Complete all
outstanding JAX work before reset, then execute, synchronize every result and
read the counters. The production build has no per-call counter updates.
"""
import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
from . import register_integrals

_FAMILIES = ('one_electron', 'eri4c', 'ri3c', 'ri2c', 'ecp')
_COUNTERS = ('evaluations', 'cache_queries', 'scheduled_quartets', 'emissions')


@lru_cache(maxsize=1)
def _library():
    handle = register_integrals()
    enabled = getattr(handle, 'GradSCFLibcintDiagnosticsEnabled', None)
    if enabled is not None:
        enabled.argtypes, enabled.restype = [], ctypes.c_int
    active = enabled is not None and bool(enabled())
    if active:
        handle.GradSCFLibcintDiagnosticsReset.argtypes = []
        handle.GradSCFLibcintDiagnosticsReset.restype = None
        handle.GradSCFLibcintDiagnosticsRead.argtypes = [ctypes.c_int, ctypes.c_int]
        handle.GradSCFLibcintDiagnosticsRead.restype = ctypes.c_uint64
    path = Path(handle._name).resolve()
    return handle, active, str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def reset_counts():
    handle, active, _, _ = _library()
    if not active:
        raise RuntimeError('libcint call counts require a separate build with --diagnostics.')
    handle.GradSCFLibcintDiagnosticsReset()


def get_counts():
    handle, active, path, sha = _library()
    return dict(schema='libcint-call-counts-v1', enabled=active,
        build_kind='diagnostic' if active else 'production', library=path, library_sha256=sha,
        families={name: {counter: int(handle.GradSCFLibcintDiagnosticsRead(i,j)) if active else None
                         for j,counter in enumerate(_COUNTERS)} for i,name in enumerate(_FAMILIES)})
