"""JAX reference integral construction for restricted and unrestricted inputs."""

from __future__ import annotations

from ..df import eri_to_df_factors_from_basis
from ..xc_backend.jax_libxc import xc_type
from typing import TYPE_CHECKING, Any, Callable
import jax.numpy as jnp
import numpy as np
from ..data.molecule import MoleculeSpec, parse_molecule_spec
from .input_types import RKSIntegralInputs, UKSIntegralInputs
from .input_spin import _unrestricted_spin_electron_counts

if TYPE_CHECKING:
    from ..scf.rks import RKSConfig
    from ..scf.uks import UKSConfig


def _build_rks_inputs_from_jax_backbone(
    *,
    _prepare_basis_grid_context: Callable[..., Any],
    _grid_ao_payload: Callable[..., Any],
    restricted_init_guess_from_pyscf: Callable[..., Any],
    overlap_hcore_matrices: Callable[..., Any],
    dipole_matrix: Callable[..., Any],
    eri_pair_matrix_packed: Callable[..., Any],
    atom: Any,
    basis: Any,
    cfg: RKSConfig,
    xc_spec_resolved: str,
    unit: str,
    charge: int,
    spin: int,
    grids_level: int,
    max_l: int,
    precompile_eri: bool,
    precompile_eri_chunk_size: int,
    include_dipole_integrals: bool,
    init_guess: Any,
    chkfile: str | None,
    init_guess_sap_basis: Any | None,
    init_guess_chkfile_project: bool | None,
    verbose: int,
    integral_backend_mode: str,
    grid_ao_backend_mode: str,
    _precompile_eri_kernels: Any,
) -> RKSIntegralInputs:
    spec = parse_molecule_spec(atom, unit=unit, charge=charge, spin=spin)
    molecule_charge = int(spec.charge)
    needs_ao_laplacian = xc_type(xc_spec_resolved) == "MGGA"
    basis_grid = _prepare_basis_grid_context(
        spec=spec,
        basis=basis,
        max_l=max_l,
        grids_level=grids_level,
        precompute_eri_groups=True,
        needs_ao_laplacian=needs_ao_laplacian,
    )
    geometry_is_traced = basis_grid.geometry_is_traced
    basis_cart = basis_grid.basis
    overlap, hcore = overlap_hcore_matrices(basis_cart, backend="jax")
    if precompile_eri:
        _precompile_eri_kernels(
            basis_cart,
            engine="jit",
            chunk_size=int(precompile_eri_chunk_size),
        )
    eri = None
    eri_pair_matrix = None
    df_factors = None
    if cfg.jk_backend == "df":
        df_factors = eri_to_df_factors_from_basis(
            basis_cart,
            tol=cfg.df_tol,
            max_rank=cfg.df_max_rank,
        )
    elif cfg.jk_backend != "direct":
        eri_pair_matrix = eri_pair_matrix_packed(basis_cart)
    ao, ao_deriv1, ao_laplacian = _grid_ao_payload(
        basis_grid,
        needs_ao_laplacian=needs_ao_laplacian,
    )
    dipole_integrals = dipole_matrix(basis_cart) if include_dipole_integrals else None
    nelectron = int(basis_cart.atom_charges.sum()) - molecule_charge
    initial_guess = restricted_init_guess_from_pyscf(
        atom=spec,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=True,
        verbose=int(verbose),
        xc_spec=xc_spec_resolved,
        init_guess=init_guess,
        sap_basis=init_guess_sap_basis,
        chkfile=chkfile,
        chkfile_project=init_guess_chkfile_project,
        geometry_is_traced=geometry_is_traced,
        dtype=hcore.dtype,
        libcint_mol=None,
    )
    inputs = RKSIntegralInputs(
        basis=basis_cart,
        overlap=overlap,
        hcore=hcore,
        eri=eri,
        eri_pair_matrix=eri_pair_matrix,
        df_factors=df_factors,
        direct_basis=basis_cart if cfg.jk_backend == "direct" else None,
        nelectron=nelectron,
        nuclear_repulsion=spec.nuclear_repulsion,
        coords=basis_grid.coords,
        grid_weights=basis_grid.grid_weights,
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        dipole_integrals=dipole_integrals,
        init_density=initial_guess.density,
        init_mo_coeff=None,
        init_mo_occ=None,
        init_mo_energy=None,
        molecule_charge=molecule_charge,
        geometry_is_traced=geometry_is_traced,
        integral_backend=integral_backend_mode,
        grid_ao_backend=grid_ao_backend_mode,
    )
    return inputs


def _build_uks_inputs_from_jax_backbone(
    *,
    _prepare_basis_grid_context: Callable[..., Any],
    _grid_ao_payload: Callable[..., Any],
    unrestricted_init_guess_from_pyscf: Callable[..., Any],
    overlap_matrix: Callable[..., Any],
    build_hcore: Callable[..., Any],
    dipole_matrix: Callable[..., Any],
    eri_tensor: Callable[..., Any],
    atom: Any,
    basis: Any,
    cfg: UKSConfig,
    xc_spec_resolved: str,
    unit: str,
    charge: int,
    spin: int,
    grids_level: int,
    max_l: int,
    precompile_eri: bool,
    init_guess: Any,
    chkfile: str | None,
    init_guess_sap_basis: Any | None,
    init_guess_chkfile_project: bool | None,
    precompile_eri_chunk_size: int,
    verbose: int,
    integral_backend_mode: str,
    grid_ao_backend_mode: str,
    _precompile_eri_kernels: Any,
) -> UKSIntegralInputs:
    spec = parse_molecule_spec(atom, unit=unit, charge=charge, spin=spin)
    molecule_charge = int(spec.charge)
    basis_grid = _prepare_basis_grid_context(
        spec=spec,
        basis=basis,
        max_l=max_l,
        grids_level=grids_level,
        precompute_eri_groups=True,
        needs_ao_laplacian=True,
    )
    geometry_is_traced = basis_grid.geometry_is_traced
    basis_cart = basis_grid.basis
    overlap = overlap_matrix(basis_cart)
    hcore = build_hcore(basis_cart)
    if precompile_eri:
        _precompile_eri_kernels(
            basis_cart,
            engine="jit",
            chunk_size=int(precompile_eri_chunk_size),
        )
    df_factors = None
    eri = jnp.zeros((0, 0, 0, 0), dtype=hcore.dtype)
    if cfg.jk_backend == "df":
        df_factors = eri_to_df_factors_from_basis(
            basis_cart,
            tol=cfg.df_tol,
            max_rank=cfg.df_max_rank,
        )
    else:
        eri = eri_tensor(basis_cart)
    ao, ao_deriv1, ao_laplacian = _grid_ao_payload(
        basis_grid,
        needs_ao_laplacian=True,
    )
    dipole_integrals = dipole_matrix(basis_cart)
    total_electrons = int(round(float(np.asarray(basis_cart.atom_charges).sum()))) - int(molecule_charge)
    nalpha, nbeta = _unrestricted_spin_electron_counts(total_electrons, int(spin))
    initial_guess = unrestricted_init_guess_from_pyscf(
        atom=spec,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=True,
        verbose=int(verbose),
        xc_spec=xc_spec_resolved,
        init_guess=init_guess,
        sap_basis=init_guess_sap_basis,
        chkfile=chkfile,
        chkfile_project=init_guess_chkfile_project,
        geometry_is_traced=geometry_is_traced,
        dtype=hcore.dtype,
        libcint_mol=None,
    )
    return UKSIntegralInputs(
        basis=basis_cart,
        overlap=overlap,
        hcore=hcore,
        eri=eri,
        df_factors=df_factors,
        nalpha=nalpha,
        nbeta=nbeta,
        nuclear_repulsion=spec.nuclear_repulsion,
        coords=basis_grid.coords,
        grid_weights=basis_grid.grid_weights,
        ao=ao,
        ao_deriv1=ao_deriv1,
        ao_laplacian=ao_laplacian,
        dipole_integrals=dipole_integrals,
        init_density_alpha=initial_guess.density_alpha,
        init_density_beta=initial_guess.density_beta,
        total_electrons=total_electrons,
        molecule_charge=molecule_charge,
        geometry_is_traced=geometry_is_traced,
        integral_backend=integral_backend_mode,
        grid_ao_backend=grid_ao_backend_mode,
    )


