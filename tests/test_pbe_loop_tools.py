import csv
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft import iks
from gradscf_tools.pbe_loop import (
    AffineTerm,
    build_affine_cache,
    load_teacher_reference,
)
from gradscf_tools.pbe_recording import (
    compute_reward,
    freeze_molecule_split,
    freeze_reward_config,
    write_candidate_bundle,
)


def _artifact(tmp_path):
    target_density = np.array([[1.7, .1], [.1, .3]])
    overlap = np.eye(2)
    reference = dict(
        coefficients=np.eye(2), occupations=np.array([2., 0.]),
        energies=np.array([-.5, .5]), density_target=target_density,
        density_ks=np.diag([2., 0.]), overlap=overlap,
        fock=np.diag([-.5, .5]), hcore=np.diag([-.7, .2]),
        grid_coords=np.array([[0., 0., 0.], [1., 0., 0.]]),
        grid_weights=np.array([1., 1.]), energy_total=np.array(.2),
        atomic_numbers=np.array([1, 1]), positions_angstrom=np.zeros((2, 3)),
    )
    npz = tmp_path / "ref.npz"
    np.savez_compressed(npz, **reference)
    digest = hashlib.sha256(npz.read_bytes()).hexdigest()
    teacher = {
        "valid": True,
        "reference_method": "wu_yang_test",
        "reference_family": "CAFES-QM9_CCSD_density_WuYang",
        "reference_artifact": {"path": str(npz), "sha256": digest},
        "source_basis": "QH9 spherical",
        "inversion_basis": "GradSCF Cartesian",
        "guide_settings": {"kind": "fixed converged PBE local KS guide"},
        "smooth_pbs": {"exponents": [.2, 1., 5., 25.]},
        "grid_level": 1,
        "quality_settings": {"diagnostic_tolerance": 1e-3},
    }
    path = tmp_path / "teacher.json"
    path.write_text(json.dumps(teacher))
    return path, npz


def test_invalid_or_missing_teacher_is_refused(tmp_path):
    teacher = tmp_path / "bad.json"
    teacher.write_text(json.dumps({"valid": False, "reference_artifact": None}))
    with pytest.raises(ValueError, match="valid"):
        load_teacher_reference(teacher)


def test_valid_teacher_loader_preserves_total_label_and_provenance(tmp_path):
    teacher_path, npz = _artifact(tmp_path)
    loaded = load_teacher_reference(teacher_path)
    assert loaded.target.energy == pytest.approx(.2)
    assert loaded.reference.method == "wu_yang_test"
    assert loaded.provenance["source_basis"] == "QH9 spherical"
    assert loaded.provenance["inversion_basis"] == "GradSCF Cartesian"
    assert loaded.reference_artifact == npz


def test_affine_cached_loss_and_gradient_match_direct_iks_loss(tmp_path):
    teacher_path, _ = _artifact(tmp_path)
    loaded = load_teacher_reference(teacher_path)
    density = loaded.reference.density

    def base(_params, d):
        return .2 + .13 * jnp.sum(d * d)

    def term(_params, d):
        return .07*jnp.sum(d*d*d)+.11*(d[0,1]+d[1,0])*(1+d[0,0])

    cache = build_affine_cache(
        loaded, baseline_energy_fn=base,
        terms=(AffineTerm("cubic", term, None),),
    )
    direct = lambda p, d: base(None, d) + p[0] * term(None, d)
    theta = jnp.array([.17])
    weights = {"energy": {"mse": 1., "mae": 1.}, "stationarity": {"norm2": .01}}
    scales = {"energy": 1., "stationarity": 1.}
    direct_value, direct_score = iks.loss(loaded.reference, direct, theta,
                                          weights=weights, scales=scales)
    assert direct_score.valid
    np.testing.assert_allclose(cache.loss(theta), direct_value, atol=2e-10)
    np.testing.assert_allclose(
        jax.grad(cache.loss)(theta), jax.grad(
            lambda p: iks.loss(loaded.reference, direct, p,
                               weights=weights, scales=scales)[0]
        )(theta), atol=2e-9,
    )


def test_reward_freezes_positive_baseline_and_penalizes_declared_failures():
    cfg = freeze_reward_config(2.0, epsilon=1e-12)
    result = compute_reward(1.0, cfg, {"finite": True, "spin": False})
    assert result["raw"] == pytest.approx(1.7)
    assert result["penalized"] == pytest.approx(1.7 * .9)
    with pytest.raises(ValueError, match="J_target"):
        freeze_reward_config(0.0)
    with pytest.raises(ValueError, match="finite"):
        freeze_reward_config(float("nan"))


def test_candidate_bundle_is_immutable_and_split_is_20_5_5(tmp_path):
    split, split_hash = freeze_molecule_split([f"m{i:02d}" for i in range(30)])
    assert {key: len(value) for key, value in split.items()} == {
        "train": 20, "validation": 5, "test": 5,
    }
    record = {"candidate_id": "c000", "status": "declared", "metrics": None}
    write_candidate_bundle(tmp_path, record, formula="E_xc=PBE", feedback="", predictions=[])
    with pytest.raises(FileExistsError):
        write_candidate_bundle(tmp_path, record, formula="changed", feedback="", predictions=[])
    assert split_hash
    with (tmp_path / "c000" / "predictions.csv").open() as handle:
        assert list(csv.reader(handle)) == [["molecule_id", "split", "reference", "prediction", "signed_error", "absolute_error", "units"]]


def _update_artifact(teacher_path, npz, **changes):
    with np.load(npz, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays.update(changes)
    np.savez_compressed(npz, **arrays)
    payload = json.loads(teacher_path.read_text())
    payload['reference_artifact']['sha256'] = hashlib.sha256(npz.read_bytes()).hexdigest()
    teacher_path.write_text(json.dumps(payload))


def test_loader_rejects_fock_inconsistent_with_supplied_spectrum(tmp_path):
    teacher_path, npz = _artifact(tmp_path)
    _update_artifact(teacher_path, npz, fock=np.array([[-.5,.1],[.1,.5]]))
    with pytest.raises(ValueError, match='Fock|fock|spectrum'):
        load_teacher_reference(teacher_path)


def test_loader_compares_total_energy_with_an_absolute_tolerance(tmp_path):
    teacher_path, _ = _artifact(tmp_path)
    payload = json.loads(teacher_path.read_text())
    payload['target_energy'] = .2000001
    teacher_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='energies'):
        load_teacher_reference(teacher_path)


def test_loader_rejects_invalid_quadrature_before_scoring(tmp_path):
    teacher_path, npz = _artifact(tmp_path)
    _update_artifact(teacher_path, npz, grid_weights=np.array([np.nan, 1.]))
    with pytest.raises(ValueError, match='grid|finite'):
        load_teacher_reference(teacher_path)


def test_affine_cache_roundtrip_and_baseline_reuse(tmp_path):
    from gradscf_tools.pbe_loop import save_affine_cache, load_affine_cache
    teacher_path, _ = _artifact(tmp_path)
    teacher = load_teacher_reference(teacher_path)
    base = lambda _, d: .2+.1*jnp.sum(d*d)
    baseline = build_affine_cache(teacher, baseline_energy_fn=base, terms=())
    path = tmp_path/'baseline_cache.npz'
    save_affine_cache(baseline, path, teacher_json=teacher_path)
    restored = load_affine_cache(path)
    np.testing.assert_allclose(restored.base_fock, baseline.base_fock, atol=1e-12)
    def forbidden(*args):
        pytest.fail('The verified PBE baseline must not be recalculated for each new term')
    term = lambda _, d: .03*jnp.sum(d*d*d)
    fresh = build_affine_cache(teacher, baseline_energy_fn=forbidden,
        terms=(AffineTerm('cubic', term),), baseline_cache=restored)
    expected = build_affine_cache(teacher, baseline_energy_fn=base,
        terms=(AffineTerm('cubic', term),))
    np.testing.assert_allclose(fresh.loss(jnp.array([.1])), expected.loss(jnp.array([.1])), atol=1e-12)
