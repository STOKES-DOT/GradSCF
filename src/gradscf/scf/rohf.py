"""Restricted open-shell Hartree-Fock using the shared Roothaan SCF solver."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import jax.numpy as jnp
from jaxtyping import Array

from gradscf.integrals.basis import CartesianBasis
from gradscf.integrals import build_hcore, eri_tensor, overlap_matrix
from .rhf import nuclear_repulsion_energy
from .roks import ROKS, ROKSConfig, ROKSResult, run_roks_from_integrals


@dataclass(frozen=True)
class ROHFConfig(ROKSConfig):
    """Restricted open-shell HF controls with exact exchange fixed to one."""

    xc_spec: str = field(default="hf", init=False)


@dataclass(frozen=True)
class ROHFResult(ROKSResult):
    """ROHF common orbitals and spin densities; semilocal XC energy is zero."""


def run_rohf_from_integrals(
    *, overlap: Array, hcore: Array, eri: Array, nalpha: int, nbeta: int,
    nuclear_repulsion: float | Array, init_density_alpha: Array | None = None,
    init_density_beta: Array | None = None, init_mo_coeff: Array | None = None,
    config: ROHFConfig | None = None,
) -> ROHFResult:
    """Run grid-free ROHF with common orbitals from full, real AO integrals."""

    cfg = ROHFConfig() if config is None else config
    if cfg.xc_spec != "hf":
        raise ValueError("ROHF requires xc_spec='hf'.")
    h = jnp.asarray(hcore)
    result = run_roks_from_integrals(
        overlap=overlap, hcore=h, eri=eri, nalpha=nalpha, nbeta=nbeta,
        nuclear_repulsion=nuclear_repulsion,
        ao=jnp.zeros((0, h.shape[0]), dtype=h.dtype),
        ao_deriv1=jnp.zeros((4, 0, h.shape[0]), dtype=h.dtype),
        grid_weights=jnp.zeros((0,), dtype=h.dtype),
        init_density_alpha=init_density_alpha, init_density_beta=init_density_beta,
        init_mo_coeff=init_mo_coeff, config=cfg,
    )
    return ROHFResult(**{f.name: getattr(result, f.name) for f in fields(result)})


def run_rohf(
    *, basis: CartesianBasis, nalpha: int, nbeta: int,
    nuclear_repulsion: float | Array | None = None, config: ROHFConfig | None = None,
) -> ROHFResult:
    """Build pure-JAX Cartesian AO integrals and run ROHF."""

    enuc = (nuclear_repulsion_energy(basis.atom_coords, basis.atom_charges)
            if nuclear_repulsion is None else nuclear_repulsion)
    return run_rohf_from_integrals(
        overlap=overlap_matrix(basis), hcore=build_hcore(basis), eri=eri_tensor(basis),
        nalpha=nalpha, nbeta=nbeta, nuclear_repulsion=enuc, config=config,
    )


@dataclass
class ROHF(ROKS):
    """Restricted open-shell Hartree-Fock facade with common spatial orbitals."""

    xc: str = field(default="hf", init=False)

    def _config(self) -> ROHFConfig:
        if self.xc != "hf":
            raise ValueError("ROHF requires xc='hf'; use ROKS for a DFT functional.")
        return ROHFConfig(max_cycle=self.max_cycle, conv_tol=self.conv_tol,
                          conv_tol_density=self.conv_tol_density, conv_tol_grad=self.conv_tol_grad,
                          damping=self.damp, level_shift=self.level_shift)
