"""Public integral-input assembly API.

Backend helpers receive explicit callables so the compatibility module
``gradscf.scf.inputs`` shares the same patch points as this public entry point.
SCF configuration and initial-guess implementations are imported on demand.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, TypeVar
from .basis import basis_from_molecule_spec
from ..data.grid import build_molecular_grid_from_spec
from ..data.grid_ao import evaluate_cartesian_ao, evaluate_cartesian_ao_with_derivatives
from ..data.molecule import MoleculeSpec
from .backends.jax_reference import (
    build_hcore, dipole_matrix, eri_pair_matrix_packed, eri_tensor,
    overlap_matrix, overlap_hcore_matrices, precompile_eri_kernels,
)
from .autodiff import LibcintGeometryGradPolicy
from .input_types import RKSIntegralInputs, UKSIntegralInputs
from .input_grid import _GRID_AO_INPUT_CACHE
from .input_libcint import (
    _LIBCINT_HOST_INTEGRAL_CACHE, _cached_libcint_host_integral,
    _gpu4pyscf_eri,
)
from . import input_libcint as _libcint
from . import input_grid as _grid
from . import assembly_cpu as _cpu
from . import assembly_jax as _jax
from ..xc_backend.jax_libxc import parse_xc

if TYPE_CHECKING:
    from ..scf.rks import RKSConfig
    from ..scf.uks import UKSConfig

_SCFConfig = TypeVar("_SCFConfig", "RKSConfig", "UKSConfig")


def restricted_init_guess_from_pyscf(**kwargs):
    from ..scf.init_guess import restricted_init_guess_from_pyscf as build_guess
    return build_guess(**kwargs)


def unrestricted_init_guess_from_pyscf(**kwargs):
    from ..scf.init_guess import unrestricted_init_guess_from_pyscf as build_guess
    return build_guess(**kwargs)


def _libcint_one_electron_from_mol(**kwargs):
    return _libcint._libcint_one_electron_from_mol(
        **kwargs,
        _cached_libcint_host_integral=_cached_libcint_host_integral,
    )


def _prepare_basis_grid_context(**kwargs):
    return _grid._prepare_basis_grid_context(
        **kwargs,
        basis_from_molecule_spec=basis_from_molecule_spec,
        build_molecular_grid_from_spec=build_molecular_grid_from_spec,
        evaluate_cartesian_ao_with_derivatives=evaluate_cartesian_ao_with_derivatives,
    )


def _grid_ao_payload(context, **kwargs):
    return _grid._grid_ao_payload(
        context, **kwargs,
        evaluate_cartesian_ao_with_derivatives=evaluate_cartesian_ao_with_derivatives,
    )


def _build_rks_inputs_from_cpu_backbone(**kwargs):
    return _cpu._build_rks_inputs_from_cpu_backbone(
        **kwargs,
        _prepare_basis_grid_context=_prepare_basis_grid_context,
        _grid_ao_payload=_grid_ao_payload,
        _cached_libcint_host_integral=_cached_libcint_host_integral,
        _libcint_one_electron_from_mol=_libcint_one_electron_from_mol,
        restricted_init_guess_from_pyscf=restricted_init_guess_from_pyscf,
    )


def _build_uks_inputs_from_cpu_backbone(**kwargs):
    return _cpu._build_uks_inputs_from_cpu_backbone(
        **kwargs,
        _prepare_basis_grid_context=_prepare_basis_grid_context,
        _grid_ao_payload=_grid_ao_payload,
        _cached_libcint_host_integral=_cached_libcint_host_integral,
        _libcint_one_electron_from_mol=_libcint_one_electron_from_mol,
        unrestricted_init_guess_from_pyscf=unrestricted_init_guess_from_pyscf,
    )


def _build_rks_inputs_from_jax_backbone(**kwargs):
    return _jax._build_rks_inputs_from_jax_backbone(
        **kwargs,
        _prepare_basis_grid_context=_prepare_basis_grid_context,
        _grid_ao_payload=_grid_ao_payload,
        restricted_init_guess_from_pyscf=restricted_init_guess_from_pyscf,
        overlap_hcore_matrices=overlap_hcore_matrices,
        dipole_matrix=dipole_matrix,
        eri_pair_matrix_packed=eri_pair_matrix_packed,
    )


def _build_uks_inputs_from_jax_backbone(**kwargs):
    return _jax._build_uks_inputs_from_jax_backbone(
        **kwargs,
        _prepare_basis_grid_context=_prepare_basis_grid_context,
        _grid_ao_payload=_grid_ao_payload,
        unrestricted_init_guess_from_pyscf=unrestricted_init_guess_from_pyscf,
        overlap_matrix=overlap_matrix,
        build_hcore=build_hcore,
        dipole_matrix=dipole_matrix,
        eri_tensor=eri_tensor,
    )


def _resolve_config(
    config: _SCFConfig | None,
    xc_spec: str | None,
    config_type: type[_SCFConfig],
) -> tuple[_SCFConfig, str]:
    xc_spec_resolved = str(xc_spec if xc_spec is not None else (config.xc_spec if config is not None else "pbe"))
    parse_xc(xc_spec_resolved)
    cfg = config_type(xc_spec=xc_spec_resolved) if config is None else config
    if cfg.xc_spec != xc_spec_resolved:
        cfg = replace(cfg, xc_spec=xc_spec_resolved)
    return cfg, xc_spec_resolved


def _resolve_integral_input_modes(
    *,
    integral_backend: Literal["jax", "cpu", "gpu", "libcint"],
    grid_ao_backend: Literal["jax"],
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy,
) -> tuple[str, str, str]:
    integral_backend_mode = str(integral_backend).lower()
    if integral_backend_mode == "libcint":
        integral_backend_mode = "cpu"
    if integral_backend_mode not in {"jax", "cpu", "gpu"}:
        raise ValueError(
            f"Unsupported integral_backend={integral_backend!r}. Expected 'jax', 'cpu', or 'gpu'."
        )
    grid_ao_backend_mode = str(grid_ao_backend).lower()
    if grid_ao_backend_mode != "jax":
        raise ValueError(
            f"Unsupported grid_ao_backend={grid_ao_backend!r}. "
            "Only grid_ao_backend='jax' is supported."
        )
    libcint_grad_policy_mode = str(libcint_geometry_grad_policy).lower()
    if libcint_grad_policy_mode not in {"analytic", "error", "zero"}:
        raise ValueError(
            f"Unsupported libcint_geometry_grad_policy={libcint_geometry_grad_policy!r}. "
            "Expected 'analytic', 'error', or 'zero'."
        )
    return integral_backend_mode, grid_ao_backend_mode, libcint_grad_policy_mode


def build_rks_integral_inputs(
    *,
    atom: Any,
    basis: Any,
    config: RKSConfig | None = None,
    xc_spec: str | None = None,
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 0,
    cart: bool = True,
    grids_level: int = 0,
    max_l: int = 3,
    grid_ao_backend: Literal["jax"] = "jax",
    integral_backend: Literal["jax", "cpu", "gpu", "libcint"] = "cpu",
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy = "analytic",
    precompile_eri: bool = False,
    precompile_eri_chunk_size: int = 512,
    _precompile_eri_kernels: Any = precompile_eri_kernels,
    include_dipole_integrals: bool = True,
    init_guess: Any = "minao",
    chkfile: str | None = None,
    init_guess_sap_basis: Any | None = None,
    init_guess_chkfile_project: bool | None = None,
    verbose: int = 0,
    **mol_kwargs: Any,
) -> RKSIntegralInputs:
    """Build integral/grid inputs for the restricted KS SCF kernel."""

    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)
    if int(spin) != 0:
        raise NotImplementedError("build_rks_integral_inputs only supports closed-shell systems.")
    if not bool(cart):
        raise NotImplementedError("build_rks_integral_inputs currently supports cart=True only.")

    from ..scf.rks import RKSConfig

    cfg, xc_spec_resolved = _resolve_config(config, xc_spec, RKSConfig)
    integral_backend_mode, grid_ao_backend_mode, libcint_grad_policy_mode = _resolve_integral_input_modes(
        integral_backend=integral_backend,
        grid_ao_backend=grid_ao_backend,
        libcint_geometry_grad_policy=libcint_geometry_grad_policy,
    )
    if integral_backend_mode == "cpu":
        return _build_rks_inputs_from_cpu_backbone(
            atom=atom,
            basis=basis,
            cfg=cfg,
            xc_spec_resolved=xc_spec_resolved,
            unit=unit,
            charge=charge,
            spin=spin,
            cart=bool(cart),
            grids_level=grids_level,
            max_l=max_l,
            precompile_eri=precompile_eri,
            include_dipole_integrals=include_dipole_integrals,
            init_guess=init_guess,
            chkfile=chkfile,
            init_guess_sap_basis=init_guess_sap_basis,
            init_guess_chkfile_project=init_guess_chkfile_project,
            verbose=verbose,
            integral_backend_mode=integral_backend_mode,
            grid_ao_backend_mode=grid_ao_backend_mode,
            libcint_grad_policy_mode=libcint_grad_policy_mode,
            mol_kwargs=dict(mol_kwargs),
        )
    inputs = _build_rks_inputs_from_jax_backbone(
        atom=atom,
        basis=basis,
        cfg=cfg,
        xc_spec_resolved=xc_spec_resolved,
        unit=unit,
        charge=charge,
        spin=spin,
        grids_level=grids_level,
        max_l=max_l,
        precompile_eri=precompile_eri,
        precompile_eri_chunk_size=precompile_eri_chunk_size,
        include_dipole_integrals=include_dipole_integrals,
        init_guess=init_guess,
        chkfile=chkfile,
        init_guess_sap_basis=init_guess_sap_basis,
        init_guess_chkfile_project=init_guess_chkfile_project,
        verbose=verbose,
        integral_backend_mode=integral_backend_mode,
        grid_ao_backend_mode=grid_ao_backend_mode,
        _precompile_eri_kernels=_precompile_eri_kernels,
    )
    if integral_backend_mode == "gpu" and cfg.jk_backend == "full":
        inputs = replace(
            inputs,
            eri_pair_matrix=_gpu4pyscf_eri(
                packed=True,
                atom=atom,
                basis=basis,
                unit=unit,
                charge=charge,
                spin=spin,
                cart=bool(cart),
                verbose=verbose,
                mol_kwargs=dict(mol_kwargs),
            ),
        )
    return inputs


def build_uks_integral_inputs(
    *,
    atom: Any,
    basis: Any,
    config: UKSConfig | None = None,
    xc_spec: str | None = None,
    unit: str = "Angstrom",
    charge: int = 0,
    spin: int = 1,
    cart: bool = True,
    grids_level: int = 0,
    max_l: int = 3,
    grid_ao_backend: Literal["jax"] = "jax",
    integral_backend: Literal["jax", "cpu", "gpu", "libcint"] = "cpu",
    libcint_geometry_grad_policy: LibcintGeometryGradPolicy = "error",
    precompile_eri: bool = False,
    precompile_eri_chunk_size: int = 512,
    _precompile_eri_kernels: Any = precompile_eri_kernels,
    init_guess: Any = "minao",
    chkfile: str | None = None,
    init_guess_sap_basis: Any | None = None,
    init_guess_chkfile_project: bool | None = None,
    verbose: int = 0,
    **mol_kwargs: Any,
) -> UKSIntegralInputs:
    """Build integral/grid inputs for the unrestricted KS SCF kernel."""

    if isinstance(atom, MoleculeSpec):
        charge = int(atom.charge)
        spin = int(atom.spin)
    if not bool(cart):
        raise NotImplementedError("build_uks_integral_inputs currently supports cart=True only.")

    from ..scf.uks import UKSConfig

    cfg, xc_spec_resolved = _resolve_config(config, xc_spec, UKSConfig)
    integral_backend_mode, grid_ao_backend_mode, libcint_grad_policy_mode = _resolve_integral_input_modes(
        integral_backend=integral_backend,
        grid_ao_backend=grid_ao_backend,
        libcint_geometry_grad_policy=libcint_geometry_grad_policy,
    )
    if integral_backend_mode in {"cpu", "gpu"}:
        inputs = _build_uks_inputs_from_cpu_backbone(
            atom=atom,
            basis=basis,
            cfg=cfg,
            xc_spec_resolved=xc_spec_resolved,
            unit=unit,
            charge=charge,
            spin=spin,
            cart=bool(cart),
            grids_level=grids_level,
            max_l=max_l,
            precompile_eri=precompile_eri,
            init_guess=init_guess,
            chkfile=chkfile,
            init_guess_sap_basis=init_guess_sap_basis,
            init_guess_chkfile_project=init_guess_chkfile_project,
            verbose=verbose,
            integral_backend_mode=integral_backend_mode,
            grid_ao_backend_mode=grid_ao_backend_mode,
            libcint_grad_policy_mode=libcint_grad_policy_mode,
            mol_kwargs=dict(mol_kwargs),
        )
        if integral_backend_mode == "gpu" and cfg.jk_backend == "full":
            inputs = replace(
                inputs,
                eri=_gpu4pyscf_eri(
                    packed=False,
                    atom=atom,
                    basis=basis,
                    unit=unit,
                    charge=charge,
                    spin=spin,
                    cart=bool(cart),
                    verbose=verbose,
                    mol_kwargs=dict(mol_kwargs),
                ),
            )
        return inputs
    inputs = _build_uks_inputs_from_jax_backbone(
        atom=atom,
        basis=basis,
        cfg=cfg,
        xc_spec_resolved=xc_spec_resolved,
        unit=unit,
        charge=charge,
        spin=spin,
        grids_level=grids_level,
        max_l=max_l,
        precompile_eri=precompile_eri,
        init_guess=init_guess,
        chkfile=chkfile,
        init_guess_sap_basis=init_guess_sap_basis,
        init_guess_chkfile_project=init_guess_chkfile_project,
        precompile_eri_chunk_size=precompile_eri_chunk_size,
        verbose=verbose,
        integral_backend_mode=integral_backend_mode,
        grid_ao_backend_mode=grid_ao_backend_mode,
        _precompile_eri_kernels=_precompile_eri_kernels,
    )
    return inputs


__all__ = ["RKSIntegralInputs", "UKSIntegralInputs", "build_rks_integral_inputs", "build_uks_integral_inputs"]
