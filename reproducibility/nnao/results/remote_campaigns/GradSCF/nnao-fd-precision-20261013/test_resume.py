import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('fixed_pool_resume', Path(__file__).with_name('resume.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_migration_preserves_arrays_and_rejects_non_source_changes(tmp_path):
    source = tmp_path/'old.npz'
    old = dict(scientific_source_sha256='old', candidate='same', learning_rate=1e-4)
    new = dict(old, scientific_source_sha256='new')
    metadata = dict(identity=old, epoch=22, loss_hartree=-2., history=[{'epoch': 22}])
    arrays = dict(parameters=np.array([.3, .5]), gradient=np.array([.02, -.04]),
                  optimizer=np.arange(17, dtype=np.uint8))
    np.savez(source, **arrays, metadata=np.asarray(json.dumps(metadata)))
    result = m.migrate_checkpoint(source, tmp_path/'new.npz', new, parent_source='old')
    assert result['epoch'] == 22 and result['optimizer_bytes_preserved']
    with pytest.raises(ValueError, match='only source'):
        m.migrate_checkpoint(source, tmp_path/'bad.npz', dict(new, candidate='changed'), parent_source='old')
    with pytest.raises(ValueError, match='overwrite'):
        m.migrate_checkpoint(source, tmp_path/'new.npz', new, parent_source='old')


def test_changed_initializer_fingerprint_requires_explicit_template_migration(tmp_path):
    old = dict(scientific_source_sha256='old', initialization_sha256='a', model_config={'channels': 8})
    new = dict(old, scientific_source_sha256='new', initialization_sha256='b')
    source = tmp_path/'old.npz'
    np.savez(source, parameters=np.array([.4]), gradient=np.array([.2]),
        optimizer=np.arange(8, dtype=np.uint8), metadata=np.asarray(json.dumps(dict(
            identity=old, epoch=22, loss_hartree=-1., history=[{'epoch': 22}]))))
    with pytest.raises(ValueError, match='only source'):
        m.migrate_checkpoint(source, tmp_path/'rejected.npz', new, parent_source='old')
    result = m.migrate_checkpoint(source, tmp_path/'new.npz', new,
        parent_source='old', allow_template_change=True)
    assert result['parameters_preserved'] and result['optimizer_bytes_preserved']
    assert result['initializer_fingerprint_changed']
    with pytest.raises(ValueError, match='only source'):
        m.migrate_checkpoint(source, tmp_path/'wrong.npz', dict(new, model_config={'channels': 16}),
            parent_source='old', allow_template_change=True)


def test_replay_catches_molecular_errors_that_cancel_in_the_mean():
    parent = dict(epoch=22, loss_hartree=-3., molecules=[
        dict(structure_id='a', energy_hartree=-2.), dict(structure_id='b', energy_hartree=-4.)])
    arrays = dict(gradient=np.array([.3]))
    assert m.verify_replay(parent, arrays, (-3., arrays['gradient'], parent['molecules']))['passed']
    changed = [dict(structure_id='a', energy_hartree=-1.), dict(structure_id='b', energy_hartree=-5.)]
    with pytest.raises(ValueError, match='Per-molecule'):
        m.verify_replay(parent, arrays, (-3., arrays['gradient'], changed))


def test_mask_migration_preserves_arrays_and_requires_zero_frozen_gradient(tmp_path):
    old = dict(scientific_source_sha256='old')
    new = dict(scientific_source_sha256='new', frozen_parameter_indices=[0],
               trainable_parameter_count=1)
    source = tmp_path/'old.npz'
    metadata = dict(identity=old, epoch=22, loss_hartree=-1., history=[{'epoch': 22}])
    def write(g):
        np.savez(source, parameters=np.array([1.7, .4]), gradient=np.array(g),
            optimizer=np.arange(8, dtype=np.uint8), metadata=np.asarray(json.dumps(metadata)))
    write([0., .2])
    report = m.migrate_checkpoint(source, tmp_path/'new.npz', new,
        parent_source='old', trainable_mask=[False, True])
    assert report['optimizer_bytes_preserved'] and report['parameters_preserved']
    with pytest.raises(ValueError, match='mask'):
        m.migrate_checkpoint(source, tmp_path/'wrong.npz', new,
            parent_source='old', trainable_mask=[True, False])
    write([.01, .2])
    with pytest.raises(ValueError, match='immutable'):
        m.migrate_checkpoint(source, tmp_path/'nonzero.npz', new,
            parent_source='old', trainable_mask=[False, True])
