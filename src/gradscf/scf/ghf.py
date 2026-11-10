"""Generalized Hartree-Fock with complex alpha/beta-mixed spinor orbitals."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from gradscf.integrals.basis import CartesianBasis
from gradscf.integrals import build_hcore, eri_tensor, overlap_matrix
from .gks import GKS, GKSConfig, GKSResult, run_gks_from_integrals
from .rhf import nuclear_repulsion_energy


@dataclass(frozen=True)
class GHFConfig(GKSConfig):
    """Generalized SCF controls with full exact exchange and zero semilocal XC."""

    xc_spec: str = field(default="hf", init=False)


@dataclass(frozen=True)
class GHFResult(GKSResult):
    """Generalized HF spinor orbitals, density, and Fock matrix."""


def run_ghf_from_integrals(
    *, overlap: Array, hcore: Array, eri: Array | None, nelectron: int,
    nuclear_repulsion: float | Array, init_density: Array | None = None,
    config: GHFConfig | None = None, df_factors: Array | None = None,
    direct_basis: Any | None = None,
) -> GHFResult:
    """Run grid-free GHF; hcore may include explicit spin-mixing matrix elements."""

    cfg = GHFConfig() if config is None else config
    if cfg.xc_spec != "hf":
        raise ValueError("GHF requires xc_spec='hf'.")
    s = jnp.asarray(overlap)
    result = run_gks_from_integrals(
        overlap=s, hcore=hcore, eri=eri, nelectron=nelectron,
        nuclear_repulsion=nuclear_repulsion, init_density=init_density,
        ao=jnp.zeros((0, s.shape[0]), dtype=s.dtype),
        ao_deriv1=jnp.zeros((4, 0, s.shape[0]), dtype=s.dtype),
        grid_weights=jnp.zeros((0,), dtype=s.dtype), config=cfg,
        df_factors=df_factors, direct_basis=direct_basis,
    )
    return GHFResult(**{f.name: getattr(result, f.name) for f in fields(result)})


def run_ghf(
    *, basis: CartesianBasis, nelectron: int,
    nuclear_repulsion: float | Array | None = None, init_density: Array | None = None,
    config: GHFConfig | None = None,
) -> GHFResult:
    """Solve complex GHF from Cartesian integrals or a native direct provider."""

    cfg = GHFConfig() if config is None else config
    if cfg.jk_backend == "df":
        raise ValueError("Use GHF(mol).density_fit() or supply explicit factors to run_ghf_from_integrals.")
    direct = None
    if cfg.jk_backend == "direct":
        from gradscf.integrals.libcint import bind_cartesian
        direct = bind_cartesian(basis).direct()
    enuc = (nuclear_repulsion_energy(basis.atom_coords, basis.atom_charges)
            if nuclear_repulsion is None else nuclear_repulsion)
    return run_ghf_from_integrals(
        overlap=overlap_matrix(basis), hcore=build_hcore(basis),
        eri=None if direct is not None else eri_tensor(basis), direct_basis=direct,
        nelectron=nelectron, nuclear_repulsion=enuc, init_density=init_density, config=cfg,
    )


@dataclass
class GHF(GKS):
    """PySCF-style generalized HF facade in an alpha-then-beta AO basis."""

    xc: str = field(default="hf", init=False)

    def _config(self) -> GHFConfig:
        if self.xc != "hf":
            raise ValueError("GHF requires xc='hf'; use GKS for DFT.")
        config = super()._config()
        return GHFConfig(**{item.name: getattr(config, item.name)
                            for item in fields(GHFConfig) if item.init})
