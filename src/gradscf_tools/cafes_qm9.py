"""Pure-data CAFES-QM9 metadata, selection, and source/target qualification."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _ao_map(source_indices: np.ndarray, phase_signs: np.ndarray, nao: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(source_indices)
    phases = np.asarray(phase_signs, dtype=float)
    if indices.shape != (nao,) or indices.dtype.kind not in "iu":
        raise ValueError("ao_source_indices must be an integer AO vector")
    if not np.array_equal(np.sort(indices), np.arange(nao)):
        raise ValueError("ao_source_indices must be a bijective AO permutation")
    if phases.shape != (nao,) or not np.allclose(np.abs(phases), 1.):
        raise ValueError("ao_phase_signs must be explicit +/-1 values")
    return np.argsort(indices), phases[np.argsort(indices)]


def ao_square_to_canonical(matrix: np.ndarray, source_indices: np.ndarray,
                           phase_signs: np.ndarray) -> np.ndarray:
    """Map a QH9 source-order square AO matrix into canonical AO order."""
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("AO square matrix must be square")
    order, phases = _ao_map(source_indices, phase_signs, value.shape[0])
    return value[np.ix_(order, order)] * phases[:, None] * phases[None, :]


def ao_rows_to_canonical(matrix: np.ndarray, source_indices: np.ndarray,
                         phase_signs: np.ndarray) -> np.ndarray:
    """Map AO rows of a coefficient matrix into canonical AO order."""
    value = np.asarray(matrix)
    if value.ndim != 2:
        raise ValueError("AO-row matrix must be two-dimensional")
    order, phases = _ao_map(source_indices, phase_signs, value.shape[0])
    return value[order] * phases[:, None]


def select_molecule_ids(manifest: dict[str, Any], *, count: int = 30) -> list[str]:
    """Select evenly spaced molecule IDs from the first delivery shard.

    This uses only manifest order and never reads energies, densities, or
    functional errors.  Restricting the first 100-molecule shard keeps the
    remote input request bounded while retaining a deterministic 30-molecule
    control set.
    """
    files = manifest.get("dataset_files", [])
    if not files or len(files[0].get("molecule_ids", [])) != 100:
        raise ValueError("CAFES manifest must expose a first shard of 100 molecules")
    if count < 1 or count > 100:
        raise ValueError("count must be between 1 and 100")
    ids = list(files[0]["molecule_ids"])
    positions = np.rint(np.linspace(0, len(ids) - 1, count)).astype(int)
    selected = [ids[int(position)] for position in positions]
    if len(set(selected)) != count:
        raise ValueError("manifest selection produced duplicate molecule IDs")
    return selected


def select_one_sample_per_molecule(index: Iterable[dict[str, Any]], molecule_ids: Iterable[str]) -> dict[str, str]:
    """Choose the earliest indexed sample per molecule, without energy screening."""
    wanted = set(molecule_ids)
    candidates: dict[str, list[dict[str, Any]]] = {molecule: [] for molecule in wanted}
    for row in index:
        molecule = str(row.get("molecule_id", ""))
        if molecule in candidates:
            candidates[molecule].append(row)
    chosen = {}
    for molecule in molecule_ids:
        rows = candidates.get(molecule, [])
        if not rows:
            raise ValueError(f"No samples indexed for molecule {molecule}")
        row = min(rows, key=lambda value: (int(value.get("frame_index", 10**12)),
                                           str(value["sample_id"])))
        chosen[molecule] = str(row["sample_id"])
    return chosen


def deterministic_split(molecule_ids: Iterable[str]) -> dict[str, list[str]]:
    """Create the fixed 20/5/5 molecule-level split without label screening."""
    ids = sorted(str(value) for value in molecule_ids)
    if len(ids) != 30 or len(set(ids)) != 30:
        raise ValueError("CAFES control loop requires 30 distinct molecules")
    result = {"train": [], "validation": [], "test": []}
    for start in range(0, 30, 6):
        result["train"].extend(ids[start:start + 4])
        result["validation"].append(ids[start + 4])
        result["test"].append(ids[start + 5])
    return result


def _require(sample: dict[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in sample]
    if missing:
        raise ValueError(f"CAFES sample missing required field(s): {', '.join(missing)}")


def qualify_hf_sample(sample: dict[str, Any], *, tolerance: float = 1e-7) -> dict[str, Any]:
    """Legacy validation of supplied RHF source fields and CCSD label.

    ``D_CCSD`` is intentionally checked only as a present/data-shaped field;
    it is never used as a noninteracting reference.  The returned reference
    method is supplied HF, not inverse KS/GKS.
    """
    _require(sample, (
        "atomic_numbers", "positions_angstrom", "positions_dft_angstrom",
        "E_CCSD", "E_HF", "F_HF", "S", "S_DFT", "D_CCSD", "C_HF",
        "epsilon_HF", "mo_occ", "occupied_mo_indices", "virtual_mo_indices",
        "ao_source_indices", "ao_phase_signs", "charge", "spin",
    ))
    s = np.asarray(sample["S"], dtype=float)
    c = np.asarray(sample["C_HF"], dtype=float)
    eps = np.asarray(sample["epsilon_HF"], dtype=float)
    occ = np.asarray(sample["mo_occ"], dtype=float)
    atomic_numbers = np.asarray(sample["atomic_numbers"])
    positions = np.asarray(sample["positions_angstrom"], dtype=float)
    positions_dft = np.asarray(sample["positions_dft_angstrom"], dtype=float)
    fock = np.asarray(sample["F_HF"], dtype=float)
    d_ccsd = np.asarray(sample["D_CCSD"], dtype=float)
    ao_indices = np.asarray(sample["ao_source_indices"])
    phase_signs = np.asarray(sample["ao_phase_signs"], dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("S must be a square AO overlap matrix")
    nao = s.shape[0]
    if c.shape != (nao, nao) or fock.shape != (nao, nao) or d_ccsd.shape != (nao, nao):
        raise ValueError("HF/CCSD AO matrices must share the S dimensions")
    if eps.shape != (nao,) or occ.shape != (nao,):
        raise ValueError("HF orbital arrays must match AO/MO dimension")
    if positions.ndim != 2 or positions.shape[1] != 3 or positions_dft.shape != positions.shape:
        raise ValueError("CAFES coordinates must be finite (N,3) arrays")
    if (atomic_numbers.ndim != 1 or atomic_numbers.size != positions.shape[0]
            or atomic_numbers.dtype.kind not in "iu"
            or np.any(atomic_numbers <= 0)):
        raise ValueError("atomic_numbers must be a positive integer vector matching coordinates")
    if not np.all(np.isfinite(np.concatenate((s.ravel(), c.ravel(), eps, occ,
                                               positions.ravel(), positions_dft.ravel(),
                                               fock.ravel(), d_ccsd.ravel())))):
        raise ValueError("CAFES numerical fields must be finite")
    s_dft = np.asarray(sample["S_DFT"], dtype=float)
    if s_dft.shape != s.shape or not np.all(np.isfinite(s_dft)):
        raise ValueError("S_DFT must be a finite AO square matrix matching S")
    if not np.allclose(s, s.T, atol=tolerance, rtol=0.):
        raise ValueError("S must be symmetric")
    if not np.allclose(s_dft, s_dft.T, atol=tolerance, rtol=0.):
        raise ValueError("S_DFT must be symmetric")
    if np.min(np.linalg.eigvalsh(s)) <= 0.:
        raise ValueError("S must be positive definite")
    if not np.allclose(fock, fock.T, atol=tolerance, rtol=0.):
        raise ValueError("F_HF must be symmetric")
    if not np.allclose(d_ccsd, d_ccsd.T, atol=tolerance, rtol=0.):
        raise ValueError("D_CCSD must be symmetric")
    if not np.all((np.isclose(occ, 0., atol=tolerance) | np.isclose(occ, 2., atol=tolerance))):
        raise ValueError("HF occupations must be exactly 0 or 2")
    if int(sample["charge"]) != 0 or int(sample["spin"]) != 0:
        raise ValueError("The fixed-HF control loop requires neutral closed-shell samples")
    if ao_indices.shape != (nao,) or len(set(map(int, ao_indices))) != nao:
        raise ValueError("ao_source_indices must be a unique AO permutation")
    if phase_signs.shape != (nao,) or not np.allclose(np.abs(phase_signs), 1., atol=tolerance, rtol=0.):
        raise ValueError("ao_phase_signs must be explicit +/-1 phase factors")
    if not np.isfinite(float(np.asarray(sample["E_CCSD"]))) or not np.isfinite(float(np.asarray(sample["E_HF"]))):
        raise ValueError("E_CCSD and E_HF must be finite scalar total energies")
    if not np.allclose(c.T @ s @ c, np.eye(nao), atol=tolerance, rtol=0.):
        raise ValueError("C_HF is not S-orthonormal")
    d_hf = (c * occ[None, :]) @ c.T
    hf_trace = float(np.trace(d_hf @ s))
    nelec = int(round(float(np.sum(occ))))
    ccsd_trace = float(np.trace(d_ccsd @ s))
    if abs(ccsd_trace - nelec) > tolerance:
        raise ValueError("D_CCSD electron-count contraction Tr(D_CCSD S) is invalid")
    s_values, s_vectors = np.linalg.eigh(s)
    s_root = (s_vectors * np.sqrt(s_values)) @ s_vectors.T
    natural_occupations = np.linalg.eigvalsh(s_root @ d_ccsd @ s_root)
    if natural_occupations[0] < -tolerance or natural_occupations[-1] > 2. + tolerance:
        raise ValueError("D_CCSD metric natural occupations are outside [0, 2]")
    return {
        "status": "qualified",
        "reference_family": "CAFES-QM9_CCSD_fixed_HF",
        "reference_method": "supplied_converged_RHF_orbitals",
        "target_kind": "total_energy",
        "target_label": "E_CCSD",
        "target_includes_nuclear_repulsion": True,
        "basis": "def2-SVP",
        "coordinate_units": "angstrom",
        "nelec": nelec,
        "hf_trace_s": hf_trace,
        "hf_trace_error": hf_trace - nelec,
        "ccsd_trace_s": ccsd_trace,
        "ccsd_trace_error": ccsd_trace - nelec,
        "ccsd_natural_occupation_min": float(natural_occupations[0]),
        "ccsd_natural_occupation_max": float(natural_occupations[-1]),
        "ccsd_density_used_as_reference": False,
        "ccsd_density_field_present": True,
        "ao_source_indices_preserved": True,
        "ao_phase_signs_preserved": True,
    }


def qualify_ccsd_sample(sample: dict[str, Any], *, tolerance: float = 1e-7) -> dict[str, Any]:
    """Qualify one sample for the PBE/CCSD-density Wu--Yang route.

    The numerical checks are shared with the historical HF-source validator,
    but the returned provenance is deliberately different: ``D_CCSD`` is the
    correlated target density and ``E_CCSD`` is the total-energy label.  The
    supplied HF orbitals, eigenvalues, occupations and overlap remain source
    validation fields only; no HF density or Fock is an inverse target.
    """
    result = qualify_hf_sample(sample, tolerance=tolerance)
    result.update({
        "reference_family": "CAFES-QM9_CCSD_density_WuYang",
        "reference_method": "unrelaxed_CCSD_Lambda_1RDM_target",
        "target_kind": "total_energy",
        "target_label": "E_CCSD",
        "target_includes_nuclear_repulsion": True,
        "ccsd_density_used_as_target": True,
        "ccsd_density_used_as_reference": False,
        "hf_source_validation_only": True,
        "hf_fock_used_as_local_base": False,
        "inverse_orbitals_source": "none_supplied; generated only by Wu-Yang if diagnostics pass",
    })
    return result


def qualify_delivery_file(path, manifest):
    """Verify the complete local delivery shard against published metadata."""
    import hashlib
    from pathlib import Path

    path = Path(path)
    entries = [row for row in manifest['dataset_files'] if row['file'] == path.name]
    if len(entries) != 1 or not path.is_file():
        raise ValueError('The local shard must match one published delivery filename')
    expected = entries[0]
    size = path.stat().st_size
    if size != int(expected['bytes']):
        raise ValueError('The local delivery shard has an incorrect byte size')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    value = digest.hexdigest()
    if value != expected['sha256']:
        raise ValueError('The local delivery shard SHA256 checksum does not match metadata')
    return dict(source_file=str(path.resolve()), bytes=size, source_archive_sha256=value,
                archive_sha256_verified=True, partial_archive_sha256_verified=False)
