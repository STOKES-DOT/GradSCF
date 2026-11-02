"""Source-transition guards are exercised without importing a GPU runtime."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def source_transition(tmp_path):
    script = Path(__file__).with_name('regress.py')
    tree = ast.parse(script.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {'manifest_scientific_hash', 'verify_sources'}]
    allowed = frozenset(('src/gradscf/solvers/linear/gmres.py',
                         'tools/train_nnao_pilot.py', 'tools/optimize_methane_nnao.py'))
    namespace = dict(hashlib=hashlib, json=json, Path=Path, ROOT=tmp_path, ALLOWED=allowed)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(script), 'exec'), namespace)
    names = sorted(allowed | {'src/gradscf/model/example.py', 'tools/validate_nnao_pilot.py',
                             'native/CMakeLists.txt'})
    for name in names:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original ' + name)
    parent = dict(files={name: file_hash(tmp_path / name) for name in names},
                  allowed_changes=sorted(allowed))
    parent['scientific_source_sha256'] = namespace['manifest_scientific_hash'](parent)
    namespace['PARENT_SOURCE'] = parent['scientific_source_sha256']
    identity = dict(scientific_source_sha256=parent['scientific_source_sha256'],
                    source_sha256=parent['files']['tools/train_nnao_pilot.py'],
                    experiment_source_sha256=parent['files']['tools/optimize_methane_nnao.py'],
                    validation_source_sha256=parent['files']['tools/validate_nnao_pilot.py'])
    for name in allowed:
        (tmp_path / name).write_text('repaired ' + name)
    repaired = dict(files={name: file_hash(tmp_path / name) for name in names})
    repaired['scientific_source_sha256'] = namespace['manifest_scientific_hash'](repaired)
    namespace['train'] = SimpleNamespace(_pilot=SimpleNamespace(
        file_hash=file_hash, scientific_source_hash=lambda root: namespace['manifest_scientific_hash'](
            dict(files={name: file_hash(root / name) for name in names}))))
    parent_path = tmp_path / 'parent.json'; new_path = tmp_path / 'new.json'
    parent_path.write_text(json.dumps(parent)); new_path.write_text(json.dumps(repaired))
    return namespace, parent_path, new_path, identity, repaired


def test_explicit_repair_accepts_only_frozen_three_file_transition(source_transition):
    namespace, parent, repaired, identity, _ = source_transition
    result = namespace['verify_sources'](parent, repaired, identity)
    assert set(result['changed_paths']) == namespace['ALLOWED']
    assert result['verified_file_count'] == 6


def test_mutated_native_configuration_is_rejected(source_transition):
    namespace, parent, repaired, identity, _ = source_transition
    (namespace['ROOT'] / 'native/CMakeLists.txt').write_text('unapproved native change')
    with pytest.raises(RuntimeError, match='Live repaired file differs'):
        namespace['verify_sources'](parent, repaired, identity)


def test_unrelated_source_change_is_rejected_even_when_new_manifest_is_consistent(source_transition):
    namespace, parent, repaired, identity, manifest = source_transition
    name = 'src/gradscf/model/example.py'
    (namespace['ROOT'] / name).write_text('unapproved model change')
    manifest['files'][name] = file_hash(namespace['ROOT'] / name)
    manifest['scientific_source_sha256'] = namespace['manifest_scientific_hash'](manifest)
    repaired.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='explicit three-file scope'):
        namespace['verify_sources'](parent, repaired, identity)


def test_checkpoint_cannot_silently_adopt_repaired_identity(source_transition):
    namespace, parent, repaired, identity, manifest = source_transition
    identity['scientific_source_sha256'] = manifest['scientific_source_sha256']
    with pytest.raises(RuntimeError, match='Checkpoint was not produced'):
        namespace['verify_sources'](parent, repaired, identity)
