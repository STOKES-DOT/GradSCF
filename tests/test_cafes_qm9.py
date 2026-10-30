"""Pure-data CAFES-QM9 selection and fixed-HF reference qualification."""
import numpy as np
import pytest

from gradscf_tools.cafes_qm9 import (
    ao_rows_to_canonical,
    ao_square_to_canonical,
    deterministic_split,
    qualify_ccsd_sample,
    qualify_hf_sample,
    select_molecule_ids,
    select_one_sample_per_molecule,
)


def sample():
    overlap = np.eye(2)
    coeff = np.eye(2)
    return {
        "atomic_numbers": np.array([1, 1]),
        "positions_angstrom": np.zeros((2, 3)),
        "positions_dft_angstrom": np.zeros((2, 3)),
        "E_CCSD": np.array(-1.0),
        "E_HF": np.array(-0.9),
        "F_HF": np.eye(2),
        "S": overlap,
        "S_DFT": overlap,
        "D_CCSD": np.diag([1.0, 1.0]),
        "C_HF": coeff,
        "epsilon_HF": np.array([-0.5, 0.2]),
        "mo_occ": np.array([2.0, 0.0]),
        "occupied_mo_indices": np.array([0]),
        "virtual_mo_indices": np.array([1]),
        "ao_source_indices": np.array([0, 1]),
        "ao_phase_signs": np.array([1.0, -1.0]),
        "charge": np.array(0),
        "spin": np.array(0),
        "sample_id": np.array("s1"),
        "molecule_id": np.array("m1"),
    }


def test_select_molecule_ids_is_manifest_only_and_deterministic():
    manifest = {"dataset_files": [{"file": "part1.h5", "molecule_ids": [f"m{i:03d}" for i in range(100)]}]}

    selected = select_molecule_ids(manifest, count=30)

    assert len(selected) == 30
    assert len(set(selected)) == 30
    assert selected[0] == "m000"
    assert selected[-1] == "m099"


def test_select_one_sample_per_molecule_uses_stable_index_not_energy():
    index = [
        {"sample_id": "b", "molecule_id": "m1", "frame_index": 20},
        {"sample_id": "a", "molecule_id": "m1", "frame_index": 10},
        {"sample_id": "c", "molecule_id": "m2", "frame_index": 5},
    ]

    chosen = select_one_sample_per_molecule(index, ["m1", "m2"])

    assert chosen == {"m1": "a", "m2": "c"}


def test_qualify_hf_sample_checks_total_labels_and_hf_reference_but_excludes_ccsd_rdm():
    result = qualify_hf_sample(sample())

    assert result["status"] == "qualified"
    assert result["target_kind"] == "total_energy"
    assert result["reference_family"] == "CAFES-QM9_CCSD_fixed_HF"
    assert result["nelec"] == 2
    assert result["hf_trace_s"] == 2.0
    assert result["ccsd_density_used_as_reference"] is False


def test_qualify_hf_sample_rejects_bad_occupation_or_missing_total_label():
    bad = sample()
    bad["mo_occ"] = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="occupation"):
        qualify_hf_sample(bad)

    missing = sample()
    del missing["E_CCSD"]
    with pytest.raises(ValueError, match="E_CCSD"):
        qualify_hf_sample(missing)


def test_deterministic_split_is_molecule_level_and_20_5_5():
    ids = [f"qm9_{i:06d}" for i in range(30)]
    split = deterministic_split(ids)
    assert {name: len(values) for name, values in split.items()} == {
        "train": 20, "validation": 5, "test": 5,
    }
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])


def test_qh9_ao_transform_uses_source_indices_and_phase_signs_without_rescaling():
    source_indices = np.array([2, 0, 1])
    phases = np.array([1.0, -1.0, 1.0])
    matrix = np.arange(9.0).reshape(3, 3)
    rows = np.arange(6.0).reshape(3, 2)

    square = ao_square_to_canonical(matrix, source_indices, phases)
    transformed_rows = ao_rows_to_canonical(rows, source_indices, phases)

    order = np.argsort(source_indices)
    expected_square = matrix[np.ix_(order, order)] * phases[order, None] * phases[None, order]
    expected_rows = rows[order] * phases[order, None]
    np.testing.assert_allclose(square, expected_square)
    np.testing.assert_allclose(transformed_rows, expected_rows)


def test_ccsd_density_is_allowed_to_be_nonidempotent_but_natural_occupations_must_be_bounded():
    data = sample()
    data["D_CCSD"] = np.diag([1.2, 0.8])
    result = qualify_hf_sample(data)
    assert result["ccsd_density_used_as_reference"] is False
    assert result["ccsd_natural_occupation_max"] == 1.2

    bad = sample()
    bad["D_CCSD"] = np.diag([2.2, -0.2])
    with pytest.raises(ValueError, match="natural|occupation"):
        qualify_hf_sample(bad)


def test_ccsd_route_relabels_hf_fields_as_source_validation_only():
    result = qualify_ccsd_sample(sample())
    assert result["reference_family"] == "CAFES-QM9_CCSD_density_WuYang"
    assert result["ccsd_density_used_as_target"] is True
    assert result["hf_source_validation_only"] is True
    assert result["hf_fock_used_as_local_base"] is False


def test_local_delivery_file_requires_official_size_and_sha256(tmp_path):
    import hashlib
    from gradscf_tools.cafes_qm9 import qualify_delivery_file
    path = tmp_path/'part.h5'
    payload = b'public dataset shard bytes'
    path.write_bytes(payload)
    manifest = {'dataset_files': [{'file':'part.h5', 'bytes':len(payload),
        'sha256':hashlib.sha256(payload).hexdigest()}]}
    result = qualify_delivery_file(path, manifest)
    assert result['archive_sha256_verified'] and result['bytes'] == len(payload)
    path.write_bytes(b'x'+payload[1:])
    with pytest.raises(ValueError, match='SHA|checksum'):
        qualify_delivery_file(path, manifest)
