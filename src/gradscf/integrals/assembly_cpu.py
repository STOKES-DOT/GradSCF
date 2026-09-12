"""CPU/libcint input assembly, including traced geometry and density fitting."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import warnings
from .autodiff import libcint_int2e_s4_with_coords, libcint_int2e_full_with_coords
from .backends.pyscf_mol import build_libcint_mol, libcint_intor_name
from .input_libcint import _libcint_one_electron_with_coords
from ..df import eri_pair_matrix_to_df_factors_traceable, true_df_factors_from_libcint_mol
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

_LIBCINT_INPUT_PARALLEL_WORKERS = 2

def _build_rks_inputs_from_cpu_backbone(
    *,
    _prepare_basis_grid_context: Callable[..., Any],
    _grid_ao_payload: Callable[..., Any],
    _cached_libcint_host_integral: Callable[..., Any],
    _libcint_one_electron_from_mol: Callable[..., Any],
    restricted_init_guess_from_pyscf: Callable[..., Any],
    atom: Any,
    basis: Any,
    cfg: RKSConfig,
    xc_spec_resolved: str,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    grids_level: int,
    max_l: int,
    precompile_eri: bool,
    include_dipole_integrals: bool,
    init_guess: Any,
    chkfile: str | None,
    init_guess_sap_basis: Any | None,
    init_guess_chkfile_project: bool | None,
    verbose: int,
    integral_backend_mode: str,
    grid_ao_backend_mode: str,
    libcint_grad_policy_mode: str,
    mol_kwargs: dict[str, Any],
) -> RKSIntegralInputs:
    needs_ao_laplacian = xc_type(xc_spec_resolved) == "MGGA"
    precompute_eri_groups = cfg.jk_backend == "direct"
    executor: ThreadPoolExecutor | None = None
    spec = atom if isinstance(atom, MoleculeSpec) else parse_molecule_spec(
        atom,
        unit=unit,
        charge=charge,
        spin=spin,
    )
    molecule_charge = int(spec.charge)
    basis_grid = _prepare_basis_grid_context(
        spec=spec,
        basis=basis,
        max_l=max_l,
        grids_level=grids_level,
        precompute_eri_groups=precompute_eri_groups,
        needs_ao_laplacian=needs_ao_laplacian,
    )
    geometry_is_traced = basis_grid.geometry_is_traced
    basis_cart = basis_grid.basis

    try:
        if geometry_is_traced:
            if not isinstance(basis, str):
                raise TypeError("Traceable libcint geometry currently supports named basis strings only.")
            coords_bohr = jnp.asarray(spec.coords_bohr)
            overlap, hcore, dipole_integrals = _libcint_one_electron_with_coords(
                coords_bohr=coords_bohr,
                symbols=tuple(spec.symbols),
                basis=str(basis),
                charge=molecule_charge,
                spin=int(spec.spin),
                cart=bool(cart),
                verbose=int(verbose),
                geometry_grad_policy=libcint_grad_policy_mode,
                include_dipole_integrals=include_dipole_integrals,
            )
            if precompile_eri:
                warnings.warn(
                    "precompile_eri is ignored for traceable libcint geometry.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            eri = None
            eri_pair_matrix = None
            df_factors = None
            if cfg.jk_backend == "df":
                eri_pair_matrix_for_df = libcint_int2e_s4_with_coords(
                    coords_bohr,
                    tuple(spec.symbols),
                    str(basis),
                    molecule_charge,
                    int(spec.spin),
                    bool(cart),
                    int(verbose),
                    libcint_grad_policy_mode,
                )
                df_factors = eri_pair_matrix_to_df_factors_traceable(
                    eri_pair_matrix_for_df,
                    nao=basis_cart.nao,
                    tol=cfg.df_tol,
                    max_rank=cfg.df_max_rank,
                )
            elif cfg.jk_backend != "direct":
                eri_pair_matrix = libcint_int2e_s4_with_coords(
                    coords_bohr,
                    tuple(spec.symbols),
                    str(basis),
                    molecule_charge,
                    int(spec.spin),
                    bool(cart),
                    int(verbose),
                    libcint_grad_policy_mode,
                )
            nuclear_repulsion = spec.nuclear_repulsion
            mol = None
        else:
            executor = ThreadPoolExecutor(max_workers=_LIBCINT_INPUT_PARALLEL_WORKERS)
            mol_future = executor.submit(
                build_libcint_mol,
                atom=atom,
                basis=basis,
                unit=unit,
                charge=int(charge),
                spin=int(spin),
                cart=bool(cart),
                verbose=int(verbose),
                **mol_kwargs,
            )
            mol = mol_future.result()
            geometry_anchor = np.asarray(mol.atom_coords(), dtype=float)
            overlap, hcore, dipole_integrals = _libcint_one_electron_from_mol(
                mol=mol,
                geometry_anchor=geometry_anchor,
                geometry_grad_policy=libcint_grad_policy_mode,
                include_dipole_integrals=include_dipole_integrals,
            )
            if precompile_eri:
                warnings.warn(
                    "precompile_eri is ignored when integral_backend='cpu'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            eri_name = libcint_intor_name(mol, "int2e")
            eri = None
            eri_pair_matrix = None
            df_factors = None
            nuclear_repulsion = spec.nuclear_repulsion

        ao, ao_deriv1, ao_laplacian = _grid_ao_payload(
            basis_grid,
            needs_ao_laplacian=needs_ao_laplacian,
        )
        coords = basis_grid.coords
        weights = basis_grid.grid_weights
        nelectron = int(basis_cart.atom_charges.sum()) - molecule_charge
        initial_guess = restricted_init_guess_from_pyscf(
            atom=spec,
            basis=basis,
            unit=unit,
            charge=charge,
            spin=spin,
            cart=bool(cart),
            verbose=int(verbose),
            xc_spec=xc_spec_resolved,
            init_guess=init_guess,
            sap_basis=init_guess_sap_basis,
            chkfile=chkfile,
            chkfile_project=init_guess_chkfile_project,
            geometry_is_traced=geometry_is_traced,
            dtype=hcore.dtype,
            libcint_mol=None if geometry_is_traced else mol,
        )

        direct_basis = basis_cart if cfg.jk_backend == "direct" else None
        if not geometry_is_traced and cfg.jk_backend == "df":
            df_factors = _cached_libcint_host_integral(
                mol=mol,
                integral_name="df_cholesky_eri",
                geometry_anchor=geometry_anchor,
                geometry_grad_policy=libcint_grad_policy_mode,
                loader=lambda: true_df_factors_from_libcint_mol(mol),
            )
        elif not geometry_is_traced and cfg.jk_backend != "direct":
            eri_pair_matrix = _cached_libcint_host_integral(
                mol=mol,
                integral_name=f"{eri_name}_s4",
                geometry_anchor=geometry_anchor,
                geometry_grad_policy=libcint_grad_policy_mode,
                loader=lambda: np.asarray(mol.intor(eri_name, aosym="s4"), dtype=float),
            )

        inputs = RKSIntegralInputs(
            basis=basis_cart,
            overlap=overlap,
            hcore=hcore,
            eri=eri,
            eri_pair_matrix=eri_pair_matrix,
            df_factors=df_factors,
            direct_basis=direct_basis,
            nelectron=nelectron,
            nuclear_repulsion=nuclear_repulsion,
            coords=coords,
            grid_weights=weights,
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
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def _build_uks_inputs_from_cpu_backbone(
    *,
    _prepare_basis_grid_context: Callable[..., Any],
    _grid_ao_payload: Callable[..., Any],
    _cached_libcint_host_integral: Callable[..., Any],
    _libcint_one_electron_from_mol: Callable[..., Any],
    unrestricted_init_guess_from_pyscf: Callable[..., Any],
    atom: Any,
    basis: Any,
    cfg: UKSConfig,
    xc_spec_resolved: str,
    unit: str,
    charge: int,
    spin: int,
    cart: bool,
    grids_level: int,
    max_l: int,
    precompile_eri: bool,
    init_guess: Any,
    chkfile: str | None,
    init_guess_sap_basis: Any | None,
    init_guess_chkfile_project: bool | None,
    verbose: int,
    integral_backend_mode: str,
    grid_ao_backend_mode: str,
    libcint_grad_policy_mode: str,
    mol_kwargs: dict[str, Any],
) -> UKSIntegralInputs:
    use_df = cfg.jk_backend == "df"
    skip_cpu_eri = integral_backend_mode == "gpu" or use_df
    df_factors = None
    mol = None
    spec = atom if isinstance(atom, MoleculeSpec) else parse_molecule_spec(
        atom,
        unit=unit,
        charge=charge,
        spin=spin,
    )
    molecule_charge = int(spec.charge)
    basis_grid = _prepare_basis_grid_context(
        spec=spec,
        basis=basis,
        max_l=max_l,
        grids_level=grids_level,
        precompute_eri_groups=False,
        needs_ao_laplacian=True,
    )
    geometry_is_traced = basis_grid.geometry_is_traced
    basis_cart = basis_grid.basis
    if isinstance(atom, MoleculeSpec):
        if not isinstance(basis, str):
            raise TypeError("Traceable libcint geometry currently supports named basis strings only.")
        coords_bohr = jnp.asarray(spec.coords_bohr)
        overlap, hcore, dipole_integrals = _libcint_one_electron_with_coords(
            coords_bohr=coords_bohr,
            symbols=tuple(spec.symbols),
            basis=str(basis),
            charge=int(spec.charge),
            spin=int(spec.spin),
            cart=bool(cart),
            verbose=int(verbose),
            geometry_grad_policy=libcint_grad_policy_mode,
            include_dipole_integrals=True,
        )
        if precompile_eri:
            warnings.warn(
                "precompile_eri is ignored for libcint UKS input construction.",
                RuntimeWarning,
                stacklevel=2,
            )
        eri = jnp.zeros((0, 0, 0, 0), dtype=hcore.dtype)
        if use_df:
            eri_pair_matrix_for_df = libcint_int2e_s4_with_coords(
                coords_bohr,
                tuple(spec.symbols),
                str(basis),
                int(spec.charge),
                int(spec.spin),
                bool(cart),
                int(verbose),
                libcint_grad_policy_mode,
            )
            df_factors = eri_pair_matrix_to_df_factors_traceable(
                eri_pair_matrix_for_df,
                nao=basis_cart.nao,
                tol=cfg.df_tol,
                max_rank=cfg.df_max_rank,
            )
        elif not skip_cpu_eri:
            eri = libcint_int2e_full_with_coords(
                coords_bohr,
                tuple(spec.symbols),
                str(basis),
                int(spec.charge),
                int(spec.spin),
                bool(cart),
                int(verbose),
                libcint_grad_policy_mode,
            )
        nuclear_repulsion = spec.nuclear_repulsion
    else:
        mol = build_libcint_mol(
            atom=atom,
            basis=basis,
            unit=unit,
            charge=int(charge),
            spin=int(spin),
            cart=bool(cart),
            verbose=int(verbose),
            **mol_kwargs,
        )
        geometry_anchor = jnp.asarray(mol.atom_coords())
        overlap, hcore, dipole_integrals = _libcint_one_electron_from_mol(
            mol=mol,
            geometry_anchor=geometry_anchor,
            geometry_grad_policy=libcint_grad_policy_mode,
            include_dipole_integrals=True,
        )
        if precompile_eri:
            warnings.warn(
                "precompile_eri is ignored for libcint UKS input construction.",
                RuntimeWarning,
                stacklevel=2,
            )
        eri = jnp.zeros((0, 0, 0, 0), dtype=hcore.dtype)
        if not skip_cpu_eri:
            eri_name = libcint_intor_name(mol, "int2e")
            eri = _cached_libcint_host_integral(
                mol=mol,
                integral_name=eri_name,
                geometry_anchor=geometry_anchor,
                geometry_grad_policy=libcint_grad_policy_mode,
                loader=lambda: jnp.asarray(mol.intor(eri_name)),
        )
        elif use_df:
            df_factors = _cached_libcint_host_integral(
                mol=mol,
                integral_name="df_cholesky_eri",
                geometry_anchor=geometry_anchor,
                geometry_grad_policy=libcint_grad_policy_mode,
                loader=lambda: true_df_factors_from_libcint_mol(mol),
            )
        nuclear_repulsion = spec.nuclear_repulsion

    ao, ao_deriv1, ao_laplacian = _grid_ao_payload(
        basis_grid,
        needs_ao_laplacian=True,
    )
    total_electrons = int(round(float(np.asarray(basis_cart.atom_charges).sum()))) - int(molecule_charge)
    nalpha, nbeta = _unrestricted_spin_electron_counts(total_electrons, int(spin))
    initial_guess = unrestricted_init_guess_from_pyscf(
        atom=spec,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        cart=bool(cart),
        verbose=int(verbose),
        xc_spec=xc_spec_resolved,
        init_guess=init_guess,
        sap_basis=init_guess_sap_basis,
        chkfile=chkfile,
        chkfile_project=init_guess_chkfile_project,
        geometry_is_traced=geometry_is_traced,
        dtype=hcore.dtype,
        libcint_mol=None if geometry_is_traced else mol,
    )
    return UKSIntegralInputs(
        basis=basis_cart,
        overlap=overlap,
        hcore=hcore,
        eri=eri,
        df_factors=df_factors,
        nalpha=nalpha,
        nbeta=nbeta,
        nuclear_repulsion=nuclear_repulsion,
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


