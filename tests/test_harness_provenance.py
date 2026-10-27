"""Runtime provenance checks with metadata-only module/library fixtures."""
from functools import lru_cache
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).parents[1]


def _tool(name):
    path = ROOT/'tools/libcint_migration'/(name+'.py')
    assert path.is_file(), f'The shared {name} provenance helper is missing'
    spec = importlib.util.spec_from_file_location('test_harness_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package(tmp_path, name):
    directory = tmp_path/name
    directory.mkdir()
    source = directory/'__init__.py'
    source.write_text('# actual imported fixture\n')
    module = ModuleType(name)
    module.__file__ = str(source)
    module.__version__ = name+'-fixture-version'
    return module, directory


def _forbid_numeric_imports(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'gradscf', 'pyscf', 'jax', 'numpy', 'scipy'}, name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)


def test_actual_tree_cpp_and_cached_native_identity(tmp_path, monkeypatch):
    helper = _tool('provenance')
    package, directory = _package(tmp_path, 'gradscf')
    cpp = directory/'integrals/_native/csrc'
    cpp.mkdir(parents=True)
    (cpp/'product.cc').write_text('// actual implementation A\n')
    library = tmp_path/'libgradscf_integrals.so'
    library.write_bytes(b'actual loaded library fixture')
    loads = []
    @lru_cache(maxsize=1)
    def register():
        loads.append('load')
        return SimpleNamespace(_name=str(library))
    register()  # Already loaded before metadata collection.
    native = ModuleType('gradscf.integrals._native')
    native.register_integrals = register
    modules = {'gradscf': package, 'gradscf.integrals._native': native,
               **{name: SimpleNamespace(__version__=name+'-already-loaded') for name in ('jax', 'numpy', 'scipy')}}
    probe = tmp_path/'probe.py'
    probe.write_text('# actual probe\n')
    _forbid_numeric_imports(monkeypatch)
    first = helper.collect(probe, modules=modules)
    assert first['python_executable'] == sys.executable and first['python_version'] == sys.version
    assert first['hostname']
    assert first['sources']['gradscf']['path'] == str(Path(package.__file__).resolve())
    assert first['sources']['gradscf']['version'] == package.__version__
    assert first['versions']['jax'] == 'jax-already-loaded'
    assert first['probes'][str(probe.resolve())] == hashlib.sha256(probe.read_bytes()).hexdigest()
    assert first['libraries'] == [dict(consumer='gradscf', path=str(library.resolve()),
        sha256=hashlib.sha256(library.read_bytes()).hexdigest())]
    (cpp/'product.cc').write_text('// actual implementation B\n')
    changed = helper.collect(probe, modules=modules)
    assert changed['sources']['gradscf']['tree_sha256'] != first['sources']['gradscf']['tree_sha256']
    Path(package.__file__).write_text('# actual Python implementation changed\n')
    assert helper.collect(probe, modules=modules)['sources']['gradscf']['tree_sha256'] != changed['sources']['gradscf']['tree_sha256']
    assert loads == ['load'], 'Metadata must only hit the existing register_integrals cache'


def test_unloaded_native_and_missing_packages_are_not_imported(tmp_path, monkeypatch):
    helper = _tool('provenance')
    package, _ = _package(tmp_path, 'gradscf')
    def forbidden():
        raise AssertionError('PBC/N/A metadata caused a native library load')
    forbidden.cache_info = lambda: SimpleNamespace(currsize=0)
    native = ModuleType('gradscf.integrals._native')
    native.register_integrals = forbidden
    _forbid_numeric_imports(monkeypatch)
    result = helper.collect(package.__file__, modules={'gradscf': package, 'gradscf.integrals._native': native})
    assert result['libraries'] == []
    assert result['versions'] == dict(jax=None, numpy=None, scipy=None)
    assert set(result['sources']) == {'gradscf'}


def test_reference_reads_loaded_pyscf_and_mapped_libraries_only(tmp_path, monkeypatch):
    helper = _tool('provenance')
    package, _ = _package(tmp_path, 'pyscf')
    cgto, cint = tmp_path/'libcgto.so', tmp_path/'libcint.so'
    cgto.write_bytes(b'loaded cgto')
    cint.write_bytes(b'loaded libcint')
    intor = ModuleType('pyscf.gto.moleintor')
    intor.libcgto = SimpleNamespace(_name=str(cgto))
    maps = tmp_path/'maps'
    maps.write_text('\n'.join(f'1000-2000 r-xp 0000 00:00 1 {p}' for p in (cgto, cint, cint)))
    _forbid_numeric_imports(monkeypatch)
    result = helper.collect(package.__file__, modules={'pyscf': package, 'pyscf.gto.moleintor': intor}, maps_path=maps)
    assert set(result['sources']) == {'pyscf'} and result['versions']['jax'] is None
    assert result['sources']['pyscf']['version'] == package.__version__
    assert {row['path'] for row in result['libraries']} == {str(cgto.resolve()), str(cint.resolve())}
    assert all(row['consumer'] == 'pyscf' for row in result['libraries'])
    assert len(result['libraries']) == 2


def test_worker_collects_once_after_all_four_samples(tmp_path, monkeypatch):
    worker = _tool('worker')
    events = []
    def calculate(cfg, folder, repeat):
        events.append(repeat)
        return dict(converged=True), {}
    def collect(probe):
        events.append('metadata')
        return dict(fixture='after-four-samples')
    fake = ModuleType('provenance')
    fake.collect = collect
    monkeypatch.setitem(sys.modules, 'provenance', fake)
    monkeypatch.setattr(worker, 'calculate', calculate)
    config, output = tmp_path/'input.json', tmp_path/'output'
    config.write_text(json.dumps(dict(case_id='metadata-only', repeats=4)))
    monkeypatch.setattr(sys, 'argv', ['worker.py', str(config), str(output)])
    worker.main()
    assert events == [0, 1, 2, 3, 'metadata']
    rows = json.loads((output/'records.json').read_text())
    assert len(rows) == 4 and all(row['provenance']['fixture'] == 'after-four-samples' for row in rows)
