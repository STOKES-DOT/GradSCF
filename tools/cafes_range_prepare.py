#!/usr/bin/env python3
"""Range-read selected CAFES HDF5 groups and qualify fixed HF references."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from gradscf_tools.cafes_qm9 import (
    deterministic_split,
    qualify_ccsd_sample,
    qualify_delivery_file,
    select_molecule_ids,
)
from gradscf_tools.range_http import BoundedRangeHTTPFile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--hdf5-url")
    source_group.add_argument("--hdf5-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-bytes", type=int, default=250 * 1024 * 1024)
    parser.add_argument("--ids", default=None,
                        help="Comma-separated molecule IDs for a bounded qualification probe")
    args = parser.parse_args()

    import h5py

    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    subset_dir = output_dir / "raw_subset"
    qualification_dir = output_dir / "qualification"
    subset_dir.mkdir(parents=True, exist_ok=True)
    qualification_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = metadata_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    molecule_ids = (args.ids.split(",") if args.ids else
                    select_molecule_ids(manifest, count=30))
    selected_groups = {}

    local_provenance = (qualify_delivery_file(args.hdf5_file, manifest)
                        if args.hdf5_file else None)
    source = (nullcontext(args.hdf5_file) if args.hdf5_file else
              BoundedRangeHTTPFile(args.hdf5_url, block_size=args.block_size,
                                   max_bytes=args.max_bytes))
    with source as remote:
        with h5py.File(remote, "r") as handle:
            keys = sorted(str(key) for key in handle.keys())
            for molecule_id in molecule_ids:
                candidates = [key for key in keys if key.startswith(molecule_id + ":")]
                if not candidates:
                    raise ValueError(f"No HDF5 sample group for {molecule_id}")
                selected_groups[molecule_id] = candidates[0]

            sys.path.insert(0, str(metadata_dir))
            from dict_loader import read_sample

            rows = []
            for molecule_id in molecule_ids:
                sample_id = selected_groups[molecule_id]
                sample = read_sample(handle[sample_id])
                qualification = qualify_ccsd_sample(sample)
                array_payload = {
                    key: value for key, value in sample.items()
                    if isinstance(value, np.ndarray) and key != "metadata"
                }
                npz_path = subset_dir / f"{_safe_name(sample_id)}.npz"
                np.savez_compressed(npz_path, **array_payload)
                metadata = sample.get("metadata", {})
                (subset_dir / f"{_safe_name(sample_id)}.metadata.json").write_text(
                    json.dumps(metadata, indent=2, default=str), encoding="utf-8"
                )
                rows.append({
                    "molecule_id": molecule_id,
                    "sample_id": sample_id,
                    "E_CCSD": float(sample["E_CCSD"]),
                    "E_HF": float(sample["E_HF"]),
                    "qualification": qualification,
                    "npz": str(npz_path.name),
                    "npz_sha256": _sha256(npz_path),
                })

    status = "qualified" if all(row["qualification"]["status"] == "qualified" for row in rows) else "blocked"
    result = {
        "status": status,
        "reference_family": "CAFES-QM9_CCSD_density_WuYang",
        "target_kind": "total_energy",
        "reference_method": "unrelaxed_CCSD_Lambda_1RDM_target",
        "basis": "def2-SVP",
        "coordinate_units": "angstrom",
        "total_energy_includes_nuclear_repulsion": True,
        "nuclear_repulsion_field_present": False,
        "nuclear_repulsion_action": "recompute from atomic_numbers/positions with GradSCF remotely",
        "ccsd_unrelaxed_density_used_as_target": True,
        "ccsd_unrelaxed_density_used_as_reference": False,
        "hf_source_validation_only": True,
        "hf_fock_used_as_local_base": False,
        "selection_rule": "first manifest shard, 30 evenly spaced molecule IDs, first sorted HDF5 group per molecule",
        "source_url": args.hdf5_url,
        "source_manifest_sha256": _sha256(manifest_path),
        "molecule_ids": molecule_ids,
        "sample_groups": selected_groups,
        "rows": rows,
        "range_provenance": None if local_provenance else remote.stats.as_dict(),
        "local_file_provenance": local_provenance,
        "partial_archive_sha256_verified": False,
    }
    qualification_path = qualification_dir / "qualification.json"
    qualification_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if status == "qualified" and len(molecule_ids) == 30:
        frozen = {
            "reference_family": result["reference_family"],
            "target_kind": result["target_kind"],
            "molecule_ids": molecule_ids,
            "sample_groups": selected_groups,
            "split": deterministic_split(molecule_ids),
            "split_rule": "sorted molecule IDs, five blocks of four train/one validation/one test",
            "qualification_sha256": _sha256(qualification_path),
        }
        (qualification_dir / "sample_manifest.json").write_text(
            json.dumps(frozen, indent=2), encoding="utf-8"
        )
    print(json.dumps({"status": status, "molecules": len(rows), "output_dir": str(output_dir)}))
    return 0 if status == "qualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
