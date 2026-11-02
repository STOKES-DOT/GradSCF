"""Checkpoint and atom-specific export guards, without initializing JAX."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def api():
    path = Path(__file__).with_name('native_forward.py')
    assert path.exists(), 'Frozen native forward benchmark is not implemented.'
    spec = importlib.util.spec_from_file_location('nnao_native_forward', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_checkpoint(path, *, epoch=41, dtype=np.float64, count=7171, seed=0,
                    source='7f066083658982a552fda47e0144e762796f69788c6a1f1c7e70c77fbfec403a'):
    config = dict(elements=[1, 6, 7, 8, 9], channels=8, num_interactions=2,
                  max_ell=1, correlation=2, zero_init=True, basis_family='szp663_direct')
    metadata = dict(epoch=epoch, identity=dict(model_config=config, seed=seed,
        dtype='float64', scientific_source_sha256=source))
    np.savez(path, metadata=np.asarray(json.dumps(metadata)),
             parameters=np.ones(count, dtype=dtype))


def test_checkpoint_is_frozen_epoch41_and_hash_is_checked(api, tmp_path):
    path = tmp_path / 'last.npz'; save_checkpoint(path)
    metadata, vector = api.read_checkpoint(path, expected_hash=api.file_hash(path))
    assert metadata['epoch'] == 41 and vector.shape == (7171,)
    assert vector.dtype == np.float64 and not vector.flags.writeable
    with pytest.raises(ValueError, match='bytes'):
        api.read_checkpoint(path, expected_hash='0' * 64)


@pytest.mark.parametrize('kwargs', [dict(epoch=42), dict(dtype=np.float32),
                                  dict(count=7170), dict(seed=1)])
def test_rejects_incompatible_checkpoint(api, tmp_path, kwargs):
    path = tmp_path / 'last.npz'; save_checkpoint(path, **kwargs)
    with pytest.raises(ValueError):
        api.read_checkpoint(path, expected_hash=api.file_hash(path))


def test_atom_labels_do_not_merge_equal_elements(api):
    row = dict(structure_id='A/x', symbols=['C', 'H', 'H'],
               coords_angstrom=[[0, 0, 0], [0, 0, 1], [0, 0, -1]], charge=0, spin=0)
    shells = [[[0, [1., .5]]], [[0, [2., .6]]], [[0, [2., .7]]]]
    exported = api.frozen_record(row, 'NNAO-epoch41', shells, nao=3)
    assert [atom[0] for atom in exported['atom']] == ['C0', 'H1', 'H2']
    assert exported['basis_by_atom']['H1'] != exported['basis_by_atom']['H2']
    assert exported['basis'] == 'NNAO-epoch41'
    assert exported['cart'] is False and exported['unit'] == 'Angstrom'
    assert exported['symbols'] == row['symbols']
    assert exported['coords_angstrom'] == row['coords_angstrom']


def test_explicit_native_parameter_device_moves_every_registered_leaf(api):
    assert hasattr(api, 'place_parameters'), 'Native parameters are not explicitly placed on CPU.'
    import jax
    import jax.numpy as jnp
    from gradscf.integrals.basis import BasisParameters
    jax.config.update('jax_enable_x64', True)
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('Run with XLA_FLAGS=--xla_force_host_platform_device_count=2.')
    source, target = devices[:2]
    params = BasisParameters((jax.device_put(jnp.asarray([1., .5]), source),),
        (jax.device_put(jnp.asarray([[.6], [.4]]), source),),
        jax.device_put(jnp.asarray([[0., 0., 0.]]), source),
        jax.device_put(jnp.asarray([[0., 0., 0.]]), source))
    moved = api.place_parameters(params, target)
    assert isinstance(moved, BasisParameters)
    for original, actual in zip(jax.tree.leaves(params), jax.tree.leaves(moved)):
        assert actual.device == target and original.device == source
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(original))
        assert actual.dtype == original.dtype == jnp.float64


def test_pilot_best975_profile_accepts_only_exact_frozen_best(api, tmp_path):
    source = '7df11aecdbf295818de02c0db67da3de8d3db5b1ccef29798114308e28d08ebb'
    path = tmp_path / 'best.npz'; save_checkpoint(path, epoch=975, source=source)
    metadata, vector = api.read_checkpoint(path, profile='pilot-best975', expected_hash=api.file_hash(path))
    assert metadata['epoch'] == 975 and not vector.flags.writeable
    with pytest.raises(ValueError):
        api.read_checkpoint(path, profile='stage2-epoch41', expected_hash=api.file_hash(path))
    save_checkpoint(path, epoch=1000, source=source)
    with pytest.raises(ValueError):
        api.read_checkpoint(path, profile='pilot-best975', expected_hash=api.file_hash(path))


def test_legacy_model_manifest_proves_old_source_and_current_mapping(api, tmp_path):
    assert hasattr(api, 'verify_legacy_model'), 'Legacy model mapping is not audited.'
    legacy = Path(api.__file__).with_name('pilot-source-inventory.json')
    identity = {'scientific_source_sha256': '7df11aecdbf295818de02c0db67da3de8d3db5b1ccef29798114308e28d08ebb'}
    evidence = api.verify_legacy_model(legacy, identity)
    assert evidence['verified_old_model_file_count'] == 127
    assert evidence['old_scientific_file_count'] == 700
    bad = json.loads(legacy.read_text()); bad['remote_sources']['model/nnao/encoder.py'] = '0' * 64
    changed = tmp_path / 'changed.json'; changed.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError):
        api.verify_legacy_model(changed, identity)
    with pytest.raises(RuntimeError):
        api.verify_legacy_model(legacy, {'scientific_source_sha256': '0' * 64})


def test_current_scientific_manifest_is_required_for_legacy_inference(api, tmp_path, monkeypatch):
    assert hasattr(api, 'verify_current_sources'), 'Legacy inference skips the current source audit.'
    # An isolated 764-file tree exercises actual disk hashes, without changing
    # the user's local scientific tree or relaxing the deployed source pin.
    files = {}
    for index in range(764):
        name = f'src/gradscf/fixture_{index:04d}.py'
        item = tmp_path / name; item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text(f'value = {index}\n'); files[name] = api.file_hash(item)
    source = api.manifest_source_hash(files, 'src/gradscf/')
    monkeypatch.setattr(api, 'CURRENT_SOURCE', source)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(scientific_source_sha256=source, files=files)))
    evidence = api.verify_current_sources(path, root=tmp_path)
    assert evidence['repaired_scientific_source_sha256'] == api.CURRENT_SOURCE
    assert evidence['verified_file_count'] == 764
    item.write_text('changed = True\n')
    with pytest.raises(RuntimeError, match='Live current source'):
        api.verify_current_sources(path, root=tmp_path)
    bad = json.loads(path.read_text()); bad['scientific_source_sha256'] = '0' * 64
    changed = tmp_path / 'changed.json'; changed.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError):
        api.verify_current_sources(changed, root=tmp_path)


def test_trained_only_baseline_must_match_verified_input_and_source(api):
    assert hasattr(api, 'reference_baseline'), 'Trained-only evaluation does not audit the reused reference baseline.'
    path = api.ROOT / 'artifacts/nnao-heldout-benchmark-20261010/job-95106/native/forward.json'
    expected = '60c3a44e19214decdbc07701c27460b8098e38dc5184e4f44e0e8e14ceba8987'
    evidence = api.reference_baseline(path, expected, ['W4-11/n2o'])
    assert evidence['reference_recomputed'] is False
    assert evidence['reference_energies_hartree']['W4-11/n2o'] == -183.14891250720433
    with pytest.raises(ValueError):
        api.reference_baseline(path, '0' * 64, ['W4-11/n2o'])
    with pytest.raises(ValueError):
        api.reference_baseline(path, expected, ['not-in-baseline'])
