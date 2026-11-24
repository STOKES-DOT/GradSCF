from __future__ import annotations

from dataclasses import fields, is_dataclass
from dataclasses import replace
from typing import Any, Literal

import jax

from ..scf.molecules import QuadratureGrid, RestrictedMolecule


_DEVICE_ARRAY_FIELDS = (
    "ao", "dipole_integrals", "rep_tensor", "mo_coeff", "mo_occ", "mo_energy",
    "rdm1", "h1e", "overlap_matrix", "fock_matrix", "atom_coords", "atom_charges",
    "ao_deriv1", "ao_laplacian", "hfx_local", "hfx_fxx", "hfx_nu", "pt2_local",
    "df_factors", "eri_pair_matrix", "eri_ovov", "eri_ovvo", "eri_oovv",
)


def _put_molecule_arrays(molecule, device):
    if device is None or not is_dataclass(molecule):
        return molecule

    moved: dict[int, tuple[Any, Any]] = {}

    def put(value):
        if value is None:
            return None
        key = id(value)
        if key not in moved:
            # Keep the original leaf alive for the extent of this conversion.
            moved[key] = value, jax.device_put(value, device)
        return moved[key][1]

    molecule_fields = {f.name for f in fields(molecule)}
    updates = {name: put(getattr(molecule, name))
               for name in _DEVICE_ARRAY_FIELDS if name in molecule_fields}
    if "grid" in molecule_fields and isinstance(molecule.grid, QuadratureGrid):
        grid = molecule.grid
        updates["grid"] = replace(grid, weights=put(grid.weights), coords=put(grid.coords))
    if "grid_ao" in molecule_fields and molecule.grid_ao is not None:
        from ..integrals.grids.blocks import AOGrid

        provider = molecule.grid_ao
        # The raw streaming provider may share the quadrature leaves above.
        updates["grid_ao"] = (jax.tree_util.tree_map(put, provider) if type(provider) is AOGrid
                              else put(provider))
    return replace(molecule, **updates)


def put_restricted_molecule_on_device(
    molecule: RestrictedMolecule,
    device: Any | None = None,
) -> RestrictedMolecule:
    """Move a restricted molecule dataclass onto an explicit JAX device."""
    return _put_molecule_arrays(molecule, device)


def resolve_execution_device(
    preference: Literal["auto", "cpu", "gpu"] = "auto",
) -> Any | None:
    """Resolve a JAX execution device from a high-level preference.

    Returns ``None`` for ``auto`` when no explicit override is required.
    """

    pref = preference.lower()
    if pref == "auto":
        return None
    if pref == "gpu":
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("execution_device='gpu' requested but no GPU is visible to JAX.")
        return gpus[0]
    if pref == "cpu":
        cpus = jax.devices("cpu")
        if not cpus:
            raise RuntimeError("execution_device='cpu' requested but no CPU is visible to JAX.")
        return cpus[0]
    raise ValueError(f"Unsupported execution device preference: {preference!r}")


def put_molecule_on_device(
    molecule: Any,
    device: Any | None = None,
) -> Any:
    """Move a molecule-like dataclass onto an explicit JAX device.

    Supports both restricted and unrestricted molecules that share the same
    field names used by GradSCF workflow code.
    """

    if isinstance(molecule, RestrictedMolecule):
        return put_restricted_molecule_on_device(molecule, device=device)
    return _put_molecule_arrays(molecule, device)
