"""Unrestricted Hartree-Fock interfaces backed by the shared JAX SCF solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array

from gradscf.integrals.basis import CartesianBasis
from gradscf.integrals import build_hcore, eri_tensor, overlap_matrix
from .facade import UKS
from .rhf import nuclear_repulsion_energy
from .uks import UKSConfig, run_uks_from_integrals


@dataclass(frozen=True)
class UHFConfig:
    """Configuration for full-integral unrestricted Hartree-Fock iterations."""

    max_cycle: int = 80
    conv_tol: float = 1e-10
    conv_tol_density: float = 1e-8
    conv_tol_grad: float = 1e-7
    convergence_metric: Literal["energy_and_residual", "energy"] = "energy_and_residual"
    damping: float = 0.0
    level_shift: float = 0.0
    orthogonalization_eps: float = 1e-10


@dataclass(frozen=True)
class UHFResult:
    """UHF energies (Hartree) and separate alpha/beta AO and MO arrays."""

    converged: bool
    total_energy: float
    electronic_energy: float
    nuclear_repulsion: float
    mo_energy_alpha: Array
    mo_energy_beta: Array
    mo_coeff_alpha: Array
    mo_coeff_beta: Array
    mo_occ_alpha: Array
    mo_occ_beta: Array
    density_matrix_alpha: Array
    density_matrix_beta: Array
    fock_matrix_alpha: Array
    fock_matrix_beta: Array
    overlap_matrix: Array
    hcore_matrix: Array
    cycles: int


def run_uhf_from_integrals(
    *,
    overlap: Array,
    hcore: Array,
    eri: Array,
    nalpha: int,
    nbeta: int,
    nuclear_repulsion: float | Array,
    init_density_alpha: Array | None = None,
    init_density_beta: Array | None = None,
    config: UHFConfig | None = None,
) -> UHFResult:
    """Run UHF from real AO integrals, without constructing a numerical grid.

    ``eri`` has shape ``(nao, nao, nao, nao)`` in chemists' notation.
    Optional spin densities select the initial SCF guess; otherwise the core
    Hamiltonian orbitals are filled with ``nalpha`` and ``nbeta`` electrons.
    """

    cfg = UHFConfig() if config is None else config
    h = jnp.asarray(hcore)
    nao = int(h.shape[0])
    result = run_uks_from_integrals(
        overlap=overlap,
        hcore=h,
        eri=eri,
        nalpha=nalpha,
        nbeta=nbeta,
        nuclear_repulsion=nuclear_repulsion,
        ao=jnp.zeros((0, nao), dtype=h.dtype),
        ao_deriv1=jnp.zeros((4, 0, nao), dtype=h.dtype),
        grid_weights=jnp.zeros((0,), dtype=h.dtype),
        init_density_alpha=init_density_alpha,
        init_density_beta=init_density_beta,
        config=UKSConfig(xc_spec="hf", **asdict(cfg)),
    )
    return UHFResult(
        converged=result.converged,
        total_energy=result.total_energy,
        electronic_energy=result.electronic_energy,
        nuclear_repulsion=result.nuclear_repulsion,
        mo_energy_alpha=result.mo_energy_alpha,
        mo_energy_beta=result.mo_energy_beta,
        mo_coeff_alpha=result.mo_coeff_alpha,
        mo_coeff_beta=result.mo_coeff_beta,
        mo_occ_alpha=result.mo_occ_alpha,
        mo_occ_beta=result.mo_occ_beta,
        density_matrix_alpha=result.density_matrix_alpha,
        density_matrix_beta=result.density_matrix_beta,
        fock_matrix_alpha=result.fock_matrix_alpha,
        fock_matrix_beta=result.fock_matrix_beta,
        overlap_matrix=result.overlap_matrix,
        hcore_matrix=result.hcore_matrix,
        cycles=result.cycles,
    )


def run_uhf(
    *,
    basis: CartesianBasis,
    nalpha: int,
    nbeta: int,
    nuclear_repulsion: float | Array | None = None,
    init_density_alpha: Array | None = None,
    init_density_beta: Array | None = None,
    config: UHFConfig | None = None,
) -> UHFResult:
    """Run UHF using pure-JAX Cartesian AO integrals (coordinates in Bohr)."""

    enuc = (
        nuclear_repulsion_energy(basis.atom_coords, basis.atom_charges)
        if nuclear_repulsion is None
        else nuclear_repulsion
    )
    return run_uhf_from_integrals(
        overlap=overlap_matrix(basis),
        hcore=build_hcore(basis),
        eri=eri_tensor(basis),
        nalpha=nalpha,
        nbeta=nbeta,
        nuclear_repulsion=enuc,
        init_density_alpha=init_density_alpha,
        init_density_beta=init_density_beta,
        config=config,
    )


@dataclass
class UHF(UKS):
    """PySCF-style UHF facade with full exact exchange and no semilocal XC.

    Reuses the UKS molecular-input and reference pipeline. Density fitting,
    direct SCF, and explicit nuclear gradients retain that facade's limitations.
    """

    xc: str = field(default="hf", init=False)

    def _config(self) -> UKSConfig:
        if self.xc != "hf":
            raise ValueError("UHF requires xc='hf'; use UKS for a DFT functional.")
        return super()._config()

    def make_rdm1(self) -> Array:
        """Return the alpha/beta AO density matrices with shape (2, nao, nao)."""

        if self.mo_coeff is None or self.mo_occ is None:
            raise RuntimeError("Run UHF.kernel() or UHF.run() before make_rdm1().")
        coeff = jnp.asarray(self.mo_coeff)
        return jnp.einsum("spi,si,sqi->spq", coeff, self.mo_occ, coeff.conj())
