from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from .._memory import DEFAULT_MAX_MEMORY_MB, validate_memory_mb
from ..tools.features import (
    MoleculeLikeState,
    _spin_density_and_gradient,
    _spin_tau,
    molecule_grid_view,
)
from .core import (
    _build_density_from_occ,
    _contains_jax_tracer,
    _host_float_unless_traced,
    _validate_density_matrix,
)
from .energy import XCContribution, unrestricted_energy, unrestricted_fock
from .rsh import local_xc_spec, validate_rsh_inputs
from .iteration import orthonormal_diis_error, run_unrestricted_scf
from ..solvers.scf import orbital_orthogonalizer
from .rks import (
    _build_jk,
    _df_jk_with_orbital_hint,
    _diagonalize_fock,
    _orthogonalizer,
    _vxc_matrix_from_grid_potential,
)
from ..dft.xc import (
    hybrid_coeff,
    parse_xc,
    xc_type,
)
from ..dft.xc.evaluate import (
    point_energy as _point_unrestricted_xc_energy,
    grid_energy as _point_unrestricted_xc_energy_kernel,
    grid_value_and_grad as _point_unrestricted_xc_value_and_grad_kernel,
    grid_hessian as _point_unrestricted_xc_hessian_kernel,
)


@dataclass(frozen=True)
class UKSConfig:
    """Configuration for unrestricted Kohn-Sham SCF iterations."""

    xc_spec: str = "pbe"
    max_cycle: int = 80
    conv_tol: float = 1e-10
    conv_tol_density: float = 1e-8
    conv_tol_grad: float = 1e-7
    convergence_metric: Literal["energy_and_residual", "energy"] = "energy_and_residual"
    damping: float = 0.0
    level_shift: float = 0.0
    orthogonalization_eps: float = 1e-10
    density_floor: float = 1e-12
    potential_clip: float | None = None
    jk_backend: Literal["full", "df", "direct"] = "full"
    df_tol: float = 1e-10
    df_max_rank: int | None = None
    auxbasis: str | None = None
    direct_scf_tol: float = 0.0
    direct_scf_workers: int = 1
    grid_block_size: int = 0
    grid_max_memory: float | None = None
    max_memory: float = field(default=DEFAULT_MAX_MEMORY_MB, kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, "max_memory", validate_memory_mb(self.max_memory))
        if self.grid_max_memory is not None:
            object.__setattr__(self, "grid_max_memory",
                               validate_memory_mb(self.grid_max_memory, name="grid_max_memory"))


@dataclass(frozen=True)
class UKSResult:
    """Unrestricted Kohn-Sham result object."""

    converged: bool
    total_energy: float
    electronic_energy: float
    nuclear_repulsion: float
    xc_energy: float
    exact_exchange_fraction: float
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
    range_separated_coefficients: tuple[float, float, float] | None = None
    dispersion_energy: Array | float | None = None

def _default_spin_mo_occ(nao: int, nelec: int, dtype) -> Array:
    return jnp.zeros((nao,), dtype=dtype).at[:nelec].set(1.0)


def _validate_spin_occ(mo_occ: Array, *, nao: int, nelec: int, label: str) -> Array:
    occ = jnp.asarray(mo_occ)
    if occ.ndim != 1 or int(occ.shape[0]) != nao:
        raise ValueError(f"{label} occupations must be a 1D vector with length nao.")
    if not _contains_jax_tracer(occ):
        if float(jnp.min(occ)) < -1e-8 or float(jnp.max(occ)) > 1.0 + 1e-8:
            raise ValueError(f"{label} occupations must lie in [0, 1].")
        if abs(float(jnp.sum(occ)) - float(nelec)) > 1e-6:
            raise ValueError(f"{label} occupations must sum to the requested electron count.")
    return occ


def _unrestricted_xc_energy_and_potential_on_grid(
    *,
    ao: Array,
    ao_deriv1: Array | None,
    weights: Array,
    density_a: Array,
    density_b: Array,
    xc_spec: str,
    density_floor: float,
    potential_clip: float | None,
    xc_kind: str,
    compute_potential: bool = True,
    return_tau: bool = False,
) -> tuple[Array | None, ...]:
    if xc_kind == "MGGA" and not return_tau:
        raise ValueError("MGGA grid potentials require return_tau=True to retain both kinetic-density potentials.")
    if xc_kind in {"HF", "LDA"}:
        rho_a = jnp.einsum("rp,pq,rq->r", ao, density_a, ao, precision=Precision.HIGHEST)
        rho_b = jnp.einsum("rp,pq,rq->r", ao, density_b, ao, precision=Precision.HIGHEST)
        grad_a = grad_b = jnp.zeros((ao.shape[0], 3), dtype=rho_a.dtype)
    else:
        rho_a, grad_a = _spin_density_and_gradient(ao, ao_deriv1, density_a)
        rho_b, grad_b = _spin_density_and_gradient(ao, ao_deriv1, density_b)
    rho_total = rho_a + rho_b
    if xc_kind == "HF":
        if not compute_potential:
            result = (jnp.asarray(0.0, dtype=rho_total.dtype), None, None, None, None)
            return result + (None, None) if return_tau else result
        zeros = jnp.zeros_like(rho_total)
        zero_grads = jnp.zeros((rho_total.shape[0], 3), dtype=rho_total.dtype)
        result = (jnp.asarray(0.0, dtype=rho_total.dtype), zeros, zeros, zero_grads, zero_grads)
        return result + (zeros, zeros) if return_tau else result

    if xc_kind == "LDA":
        response_variables = jnp.stack([rho_a, rho_b], axis=-1)
    elif xc_kind in {"GGA", "MGGA"}:
        response_variables = jnp.concatenate(
            [
                rho_a[..., None],
                rho_b[..., None],
                grad_a,
                grad_b,
            ],
            axis=-1,
        )
        if xc_kind == "MGGA":
            response_variables = jnp.concatenate([response_variables,
                _spin_tau(ao_deriv1, density_a)[:, None],
                _spin_tau(ao_deriv1, density_b)[:, None]], axis=-1)
    else:
        raise ValueError(f"Unsupported XC kind={xc_kind!r}.")

    mask = rho_total > density_floor
    response_variables = jnp.where(mask[:, None], response_variables,
                                   jnp.ones_like(response_variables))
    if compute_potential:
        point_exc, point_grad = _point_unrestricted_xc_value_and_grad_kernel(
            xc_spec, xc_kind)(response_variables)
    else:
        point_exc = _point_unrestricted_xc_energy_kernel(xc_spec, xc_kind)(response_variables)
    point_exc = jnp.where(mask, point_exc, 0.)
    exc = jnp.tensordot(weights, point_exc, axes=(0, 0))
    if not compute_potential:
        result = (exc, None, None, None, None)
        return result + (None, None) if return_tau else result

    vxc_rho_a = jnp.where(mask, point_grad[:, 0], 0.0)
    vxc_rho_b = jnp.where(mask, point_grad[:, 1], 0.0)
    if xc_kind in {"GGA", "MGGA"}:
        vxc_grad_a = jnp.where(mask[:, None], point_grad[:, 2:5], 0.0)
        vxc_grad_b = jnp.where(mask[:, None], point_grad[:, 5:8], 0.0)
    else:
        vxc_grad_a = jnp.zeros((rho_total.shape[0], 3), dtype=rho_total.dtype)
        vxc_grad_b = jnp.zeros((rho_total.shape[0], 3), dtype=rho_total.dtype)
    vxc_tau_a = jnp.where(mask, point_grad[:, 8], 0.) if xc_kind == "MGGA" else jnp.zeros_like(rho_total)
    vxc_tau_b = jnp.where(mask, point_grad[:, 9], 0.) if xc_kind == "MGGA" else jnp.zeros_like(rho_total)

    if potential_clip is not None:
        clip = jnp.asarray(potential_clip, dtype=rho_total.dtype)
        vxc_rho_a = jnp.clip(vxc_rho_a, -clip, clip)
        vxc_rho_b = jnp.clip(vxc_rho_b, -clip, clip)
        vxc_grad_a = jnp.clip(vxc_grad_a, -clip, clip)
        vxc_grad_b = jnp.clip(vxc_grad_b, -clip, clip)
        vxc_tau_a = jnp.clip(vxc_tau_a, -clip, clip)
        vxc_tau_b = jnp.clip(vxc_tau_b, -clip, clip)
    result = (exc, vxc_rho_a, vxc_rho_b, vxc_grad_a, vxc_grad_b)
    return result + (vxc_tau_a, vxc_tau_b) if return_tau else result


def _spin_jk_matrices(
    eri: Array,
    density_a: Array,
    density_b: Array,
    df_factors: Array | None = None,
    mo_coeff_a: Array | None = None,
    mo_coeff_b: Array | None = None,
    mo_occ_a: Array | None = None,
    mo_occ_b: Array | None = None,
    with_k: bool = True,
    direct_basis: Any | None = None,
    direct_scf_tol: float = 0.0,
    direct_scf_workers: int = 1,
) -> tuple[Array, Array, Array, Array]:
    density_tot = density_a + density_b
    if df_factors is not None:
        factors = jnp.asarray(df_factors)
        rho_aux = jnp.einsum("Qpq,pq->Q", factors, density_tot, precision=Precision.HIGHEST)
        j_tot = jnp.einsum("Q,Qpq->pq", rho_aux, factors, precision=Precision.HIGHEST)
        j_tot = 0.5 * (j_tot + j_tot.T)
        if not bool(with_k):
            zero_k = jnp.zeros_like(j_tot)
            return density_tot, j_tot, zero_k, zero_k

        def _k_from_density_or_orbitals(density: Array, coeff: Array | None, occ: Array | None) -> Array:
            if coeff is None or occ is None:
                projected = jnp.matmul(factors, density, precision=Precision.HIGHEST)
                k_mat = jnp.einsum("Qps,Qqs->pq", projected, factors, precision=Precision.HIGHEST)
            else:
                _, k_mat = _df_jk_with_orbital_hint(
                    factors, density, jnp.asarray(coeff, dtype=factors.dtype),
                    jnp.asarray(occ, dtype=factors.dtype), int(coeff.shape[-1]),
                )
            return 0.5 * (k_mat + k_mat.T)

        k_a = _k_from_density_or_orbitals(density_a, mo_coeff_a, mo_occ_a)
        k_b = _k_from_density_or_orbitals(density_b, mo_coeff_b, mo_occ_b)
        return density_tot, j_tot, k_a, k_b
    if direct_basis is not None:
        from ..integrals.backends.native.jk import _selected_jk

        density = jnp.stack((density_a, density_b)) if with_k else density_tot
        coulomb, exchange = _selected_jk(
            direct_basis, density, with_k=with_k,
            screening_threshold=direct_scf_tol, workers=direct_scf_workers)
        if with_k:
            return density_tot, coulomb.sum(axis=0), exchange[0], exchange[1]
        zero = jnp.zeros_like(coulomb)
        return density_tot, coulomb, zero, zero
    j_tot, _ = _build_jk(eri, density_tot, with_k=False)
    if not bool(with_k):
        zero_k = jnp.zeros_like(j_tot)
        return density_tot, j_tot, zero_k, zero_k
    _, exchange = _build_jk(eri, jnp.stack([density_a, density_b]), with_j=False)
    return density_tot, j_tot, exchange[0], exchange[1]


def _unrestricted_vxc_matrices(
    *,
    ao: Array,
    ao_deriv1: Array | None,
    ao_laplacian: Array | None,
    weights: Array,
    vxc_rho_a: Array,
    vxc_rho_b: Array,
    vxc_grad_a: Array,
    vxc_grad_b: Array,
    xc_kind: str,
    vxc_tau_a: Array | None = None,
    vxc_tau_b: Array | None = None,
) -> tuple[Array, Array]:
    if xc_kind in {"MGGA", "MGGA_LAPL"} and (vxc_tau_a is None or vxc_tau_b is None):
        raise ValueError("MGGA AO potentials require both spin kinetic-density potentials.")
    zeros_a = jnp.zeros_like(vxc_rho_a)
    zeros_b = jnp.zeros_like(vxc_rho_b)
    return (
        _vxc_matrix_from_grid_potential(
            ao=ao,
            ao_deriv1=ao_deriv1,
            ao_laplacian=ao_laplacian,
            weights=weights,
            vxc_rho=vxc_rho_a,
            vxc_grad=vxc_grad_a,
            vxc_tau=zeros_a if vxc_tau_a is None else vxc_tau_a,
            vxc_lapl=zeros_a,
            xc_kind=xc_kind,
        ),
        _vxc_matrix_from_grid_potential(
            ao=ao,
            ao_deriv1=ao_deriv1,
            ao_laplacian=ao_laplacian,
            weights=weights,
            vxc_rho=vxc_rho_b,
            vxc_grad=vxc_grad_b,
            vxc_tau=zeros_b if vxc_tau_b is None else vxc_tau_b,
            vxc_lapl=zeros_b,
            xc_kind=xc_kind,
        ),
    )


def _unrestricted_xc_energy_and_matrices(
    *,
    density_a: Array,
    density_b: Array,
    ao: Array,
    ao_deriv1: Array | None,
    weights: Array,
    cfg: UKSConfig,
    xc_kind: str,
    grid_ao: Any | None = None,
    compute_potential: bool = True,
) -> tuple[Array, Array]:
    """Reduce unrestricted built-in XC to an energy and two AO potentials."""
    dtype = jnp.result_type(density_a, density_b, ao, weights)
    initial = (jnp.asarray(0., dtype=dtype),
               jnp.zeros((2,) + density_a.shape, dtype=dtype))
    if xc_kind == "HF":
        return initial

    def contribution(carry, block_ao, derivative, laplacian, coords, block_weights):
        del laplacian, coords
        energy, v_ra, v_rb, v_ga, v_gb, v_ta, v_tb = _unrestricted_xc_energy_and_potential_on_grid(
            ao=block_ao, ao_deriv1=derivative, weights=block_weights,
            density_a=density_a, density_b=density_b, xc_spec=cfg.xc_spec,
            density_floor=cfg.density_floor, potential_clip=cfg.potential_clip,
            xc_kind=xc_kind, compute_potential=compute_potential, return_tau=True)
        if not compute_potential:
            return carry[0] + energy, carry[1]
        va, vb = _unrestricted_vxc_matrices(
            ao=block_ao, ao_deriv1=derivative, ao_laplacian=None, weights=block_weights,
            vxc_rho_a=v_ra, vxc_rho_b=v_rb, vxc_grad_a=v_ga, vxc_grad_b=v_gb,
            vxc_tau_a=v_ta, vxc_tau_b=v_tb, xc_kind=xc_kind)
        return carry[0] + energy, carry[1] + jnp.stack([va, vb])

    if grid_ao is None:
        return contribution(initial, ao, ao_deriv1, None, None, weights)
    return grid_ao.reduce(contribution, initial, deriv=0 if xc_kind == "LDA" else 1)


def _raw_fock_and_energy_for_state(
    *,
    density_a: Array,
    density_b: Array,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    h: Array,
    eri: Array,
    df_factors: Array | None,
    enuc: Array,
    alpha: Array | None,
    cfg: UKSConfig,
    xc_kind: str,
    mo_coeff_a: Array | None = None,
    mo_coeff_b: Array | None = None,
    mo_occ_a: Array | None = None,
    mo_occ_b: Array | None = None,
    mo_energy_a: Array | None = None,
    mo_energy_b: Array | None = None,
    overlap: Array | None = None,
    bound_xc: Any | None = None,
    molecule_template: Any | None = None,
    eri_lr: Array | None = None,
    dispersion_energy: Array | None = None,
    compute_potential: bool = True,
    direct_basis: Any | None = None,
    grid_ao: Any | None = None,
) -> tuple[Array, Array, Array | None, Array | None]:
    if not compute_potential and bound_xc is not None:
        raise NotImplementedError("Energy-only evaluation is limited to the classic orbital objective.")
    rsh = validate_rsh_inputs(cfg.xc_spec, eri, eri_lr, dispersion_energy,
                              jk_backend=cfg.jk_backend, direct_basis=direct_basis)
    if rsh is not None and bound_xc is not None:
        raise NotImplementedError("Range-separated XC does not support bound_xc overrides.")
    if rsh is not None and cfg.jk_backend == "direct":
        from .rsh import _direct_rsh_jk
        density_tot = density_a + density_b
        coefficients = rsh.to_range_separated_hybrid_coefficients()
        controls = dict(screening_threshold=cfg.direct_scf_tol, workers=cfg.direct_scf_workers)
        j_tot, _ = _direct_rsh_jk(direct_basis, density_tot, coefficients, with_k=False, **controls)
        _, exchange = _direct_rsh_jk(direct_basis, jnp.stack((density_a, density_b)),
                                     coefficients, with_j=False, **controls)
        k_a, k_b = exchange[0], exchange[1]
    else:
        density_tot, j_tot, k_a, k_b = _spin_jk_matrices(
            eri,
            density_a,
            density_b,
            df_factors=df_factors,
            mo_coeff_a=mo_coeff_a,
            mo_coeff_b=mo_coeff_b,
            mo_occ_a=mo_occ_a,
            mo_occ_b=mo_occ_b,
            with_k=(bound_xc is not None or rsh is not None
                    or abs(float(hybrid_coeff(cfg.xc_spec))) > 1e-14),
            direct_basis=direct_basis,
            direct_scf_tol=cfg.direct_scf_tol,
            direct_scf_workers=cfg.direct_scf_workers,
        )
        if rsh is not None:
            sr, delta = rsh.default_sr_hf_fraction, rsh.default_lr_hf_fraction-rsh.default_sr_hf_fraction
            _, long_range = _build_jk(eri_lr, jnp.stack([density_a, density_b]), with_j=False)
            k_a = sr*k_a + delta*long_range[0]
            k_b = sr*k_b + delta*long_range[1]
    extra_fock_a = extra_fock_b = jnp.zeros_like(h)
    include_hf_energy = bound_xc is None
    if bound_xc is None:
        alpha_eff = jnp.asarray(1.0 if rsh is not None else alpha, dtype=h.dtype)
        xc_energy, potential = _unrestricted_xc_energy_and_matrices(
            density_a=density_a, density_b=density_b, ao=ao, ao_deriv1=ao_deriv1,
            weights=weights, cfg=cfg, xc_kind=xc_kind, grid_ao=grid_ao,
            compute_potential=compute_potential)
    else:
        molecule_state = _molecule_like_state_for_bound_xc(
            density_a=density_a,
            density_b=density_b,
            mo_coeff_a=mo_coeff_a,
            mo_coeff_b=mo_coeff_b,
            mo_occ_a=mo_occ_a,
            mo_occ_b=mo_occ_b,
            mo_energy_a=mo_energy_a,
            mo_energy_b=mo_energy_b,
            ao=ao,
            ao_deriv1=ao_deriv1,
            weights=weights,
            h=h,
            eri=eri,
            overlap=overlap,
            molecule_template=molecule_template,
            grid_ao=grid_ao,
        )
        (
            vxc_rho_a,
            vxc_rho_b,
            vxc_grad_a,
            vxc_grad_b,
            xc_kind,
            alpha_eff,
            extra_fock_a,
            extra_fock_b,
        ) = bound_xc.unrestricted_scf_components(molecule_state)
        xc_energy = jnp.asarray(bound_xc.energy_from_molecule(molecule_state), dtype=h.dtype)
        # This legacy callback has no kinetic-density potential channels.
        if xc_kind in {"MGGA", "MGGA_LAPL"}:
            raise ValueError("MGGA AO potentials require both spin kinetic-density potentials.")
        from .differentiable import _build_vxc_matrix_from_components

        potential = jnp.stack([
            _build_vxc_matrix_from_components(
                molecule=molecule_state, weights=weights, v_rho=v_rho, v_grad=v_grad,
                v_tau=jnp.zeros_like(v_rho), v_lapl=jnp.zeros_like(v_rho), xc_kind=xc_kind)
            for v_rho, v_grad in ((vxc_rho_a, vxc_grad_a), (vxc_rho_b, vxc_grad_b))
        ])
    xc=XCContribution(xc_energy,potential,alpha_eff,
        jnp.stack([extra_fock_a,extra_fock_b]),energy_includes_exact_exchange=not include_hf_energy)
    exchange=jnp.stack([k_a,k_b])
    total=unrestricted_energy(jnp.stack([density_a,density_b]),h,j_tot,exchange,xc,nuclear_repulsion=enuc)
    if dispersion_energy is not None:
        total = total + dispersion_energy
    if not compute_potential:
        return total, xc_energy, None, None
    fock=unrestricted_fock(h,j_tot,exchange,xc)
    return total,xc_energy,fock[0],fock[1]


def _molecule_like_state_for_bound_xc(
    *,
    density_a: Array,
    density_b: Array,
    mo_coeff_a: Array,
    mo_coeff_b: Array,
    mo_occ_a: Array,
    mo_occ_b: Array,
    mo_energy_a: Array,
    mo_energy_b: Array,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    h: Array,
    eri: Array,
    overlap: Array,
    molecule_template: Any | None,
    grid_ao: Any | None = None,
) -> Any:
    return MoleculeLikeState(
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid=molecule_grid_view(weights, template=getattr(molecule_template, "grid", None)),
        rdm1=jnp.stack([density_a, density_b], axis=0),
        mo_coeff=jnp.stack([mo_coeff_a, mo_coeff_b], axis=0),
        mo_occ=jnp.stack([mo_occ_a, mo_occ_b], axis=0),
        mo_energy=jnp.stack([mo_energy_a, mo_energy_b], axis=0),
        rep_tensor=eri,
        h1e=h,
        overlap_matrix=overlap,
        ao_laplacian=getattr(molecule_template, "ao_laplacian", None),
        atom_coords=getattr(molecule_template, "atom_coords", None),
        atom_charges=getattr(molecule_template, "atom_charges", None),
        hfx_omega_values=getattr(molecule_template, "hfx_omega_values", None),
        hfx_local=getattr(molecule_template, "hfx_local", None),
        hfx_fxx=getattr(molecule_template, "hfx_fxx", None),
        hfx_nu=getattr(molecule_template, "hfx_nu", None),
        pt2_local=getattr(molecule_template, "pt2_local", None),
        pt2_fock_response=getattr(molecule_template, "pt2_fock_response", None),
        grid_ao=grid_ao if grid_ao is not None else getattr(molecule_template, "grid_ao", None),
    )


def _spin_diis_error(fock_spin: Array, density_spin: Array, overlap: Array, corth: Array) -> Array:
    errors = jax.vmap(
        lambda fock, density: orthonormal_diis_error(fock, density, overlap, corth)
    )(fock_spin, density_spin)
    return errors.reshape(-1)


def run_uks_from_integrals(
    *,
    overlap: Array,
    hcore: Array,
    eri: Array,
    nalpha: int,
    nbeta: int,
    nuclear_repulsion: float | Array,
    ao: Array,
    ao_deriv1: Array,
    grid_weights: Array,
    grid_ao: Any | None = None,
    init_density_alpha: Array | None = None,
    init_density_beta: Array | None = None,
    init_mo_coeff_alpha: Array | None = None,
    init_mo_coeff_beta: Array | None = None,
    init_mo_occ_alpha: Array | None = None,
    init_mo_occ_beta: Array | None = None,
    init_mo_energy_alpha: Array | None = None,
    init_mo_energy_beta: Array | None = None,
    df_factors: Array | None = None,
    config: UKSConfig | None = None,
    bound_xc: Any | None = None,
    molecule_template: Any | None = None,
    eri_lr: Array | None = None,
    dispersion_energy: Array | None = None,
    direct_basis: Any | None = None,
) -> UKSResult:
    """Run unrestricted Kohn-Sham SCF from AO integrals and numerical grid data."""

    cfg = UKSConfig() if config is None else config
    rsh = validate_rsh_inputs(cfg.xc_spec, eri, eri_lr, dispersion_energy,
                              jk_backend=cfg.jk_backend, direct_basis=direct_basis)
    xc_kind = None
    if bound_xc is None:
        parse_xc(local_xc_spec(cfg.xc_spec))
        xc_kind = xc_type(local_xc_spec(cfg.xc_spec))
    s = jnp.asarray(overlap)
    h = jnp.asarray(hcore)
    eri = jnp.asarray(eri)
    df_factors_arr = None if df_factors is None else jnp.asarray(df_factors)
    ao = jnp.asarray(ao)
    ao_deriv1 = jnp.asarray(ao_deriv1)
    weights = jnp.asarray(grid_weights)
    enuc = jnp.asarray(nuclear_repulsion)
    if cfg.jk_backend == "df" and df_factors_arr is None:
        raise ValueError("UKS jk_backend='df' requires df_factors.")
    if cfg.jk_backend == "direct" and direct_basis is None:
        raise ValueError("UKS jk_backend='direct' requires an AO J/K provider.")
    traceable_inputs = _contains_jax_tracer((s, h, eri, eri_lr, dispersion_energy, df_factors_arr, ao, ao_deriv1, weights, enuc, grid_ao))
    nao = int(s.shape[0])
    if nalpha < 0 or nalpha > nao or nbeta < 0 or nbeta > nao or (nalpha + nbeta) <= 0:
        raise ValueError("Invalid occupation counts for UKS.")
    if xc_kind in {"GGA", "MGGA"} and ao_deriv1.shape[0] < 4:
        raise ValueError("GGA/MGGA UKS requires ao_deriv1 to include AO values plus first derivatives.")

    x = (_orthogonalizer(s, cfg.orthogonalization_eps) if bound_xc is not None
         else orbital_orthogonalizer(s, cfg.orthogonalization_eps))
    core_mo_energy, core_mo_coeff = _diagonalize_fock(h, x)
    mo_coeff_a = core_mo_coeff if init_mo_coeff_alpha is None else jnp.asarray(init_mo_coeff_alpha)
    mo_coeff_b = core_mo_coeff if init_mo_coeff_beta is None else jnp.asarray(init_mo_coeff_beta)
    mo_energy_a = core_mo_energy if init_mo_energy_alpha is None else jnp.asarray(init_mo_energy_alpha)
    mo_energy_b = core_mo_energy if init_mo_energy_beta is None else jnp.asarray(init_mo_energy_beta)

    mo_occ_a = _default_spin_mo_occ(nao, nalpha, h.dtype) if init_mo_occ_alpha is None else init_mo_occ_alpha
    mo_occ_b = _default_spin_mo_occ(nao, nbeta, h.dtype) if init_mo_occ_beta is None else init_mo_occ_beta
    mo_occ_a = _validate_spin_occ(mo_occ_a, nao=nao, nelec=nalpha, label="alpha")
    mo_occ_b = _validate_spin_occ(mo_occ_b, nao=nao, nelec=nbeta, label="beta")

    density_a = _build_density_from_occ(mo_coeff_a, mo_occ_a)
    density_b = _build_density_from_occ(mo_coeff_b, mo_occ_b)
    init_density_a = _validate_density_matrix(
        init_density_alpha,
        nao=nao,
        dtype=h.dtype,
        label="init_density_alpha",
        method="UKS",
    )
    init_density_b = _validate_density_matrix(
        init_density_beta,
        nao=nao,
        dtype=h.dtype,
        label="init_density_beta",
        method="UKS",
    )
    if init_density_a is not None:
        density_a = init_density_a
    if init_density_b is not None:
        density_b = init_density_b

    if bound_xc is None:
        alpha_scalar = rsh.default_sr_hf_fraction if rsh is not None else float(hybrid_coeff(cfg.xc_spec))
        alpha = jnp.asarray(alpha_scalar, dtype=h.dtype)
    else:
        alpha = None
        alpha_scalar = float(jnp.asarray(getattr(bound_xc, "exact_exchange_fraction", 0.0)))

    def eval_state(
        density_a_arg: Array,
        density_b_arg: Array,
        mo_coeff_a_arg: Array,
        mo_coeff_b_arg: Array,
        mo_energy_a_arg: Array,
        mo_energy_b_arg: Array,
    ) -> tuple[Array, Array, Array, Array]:
        return _raw_fock_and_energy_for_state(
            density_a=density_a_arg,
            density_b=density_b_arg,
            mo_coeff_a=mo_coeff_a_arg,
            mo_coeff_b=mo_coeff_b_arg,
            mo_occ_a=mo_occ_a,
            mo_occ_b=mo_occ_b,
            mo_energy_a=mo_energy_a_arg,
            mo_energy_b=mo_energy_b_arg,
            ao=ao,
            ao_deriv1=ao_deriv1,
            weights=weights,
            h=h,
            eri=eri,
            df_factors=df_factors_arr if cfg.jk_backend == "df" else None,
            enuc=enuc,
            alpha=alpha,
            cfg=cfg,
            xc_kind=xc_kind,
            overlap=s,
            bound_xc=bound_xc,
            molecule_template=molecule_template,
            eri_lr=eri_lr,
            dispersion_energy=dispersion_energy,
            direct_basis=direct_basis if cfg.jk_backend == "direct" else None,
            grid_ao=grid_ao,
        )

    def fock_builder(density_spin: Array, mo_coeff_spin: Array, mo_energy_spin: Array) -> tuple[Array, Array, Array]:
        energy, _, raw_fock_a, raw_fock_b = eval_state(
            density_spin[0],
            density_spin[1],
            mo_coeff_spin[0],
            mo_coeff_spin[1],
            mo_energy_spin[0],
            mo_energy_spin[1],
        )
        fock_eff_a = raw_fock_a
        fock_eff_b = raw_fock_b
        if cfg.level_shift != 0.0:
            shift = jnp.asarray(cfg.level_shift, dtype=h.dtype)
            fock_eff_a = fock_eff_a + shift * s
            fock_eff_b = fock_eff_b + shift * s
        return (
            jnp.stack([fock_eff_a, fock_eff_b], axis=0),
            jnp.stack([raw_fock_a, raw_fock_b], axis=0),
            energy,
        )

    (
        density_spin,
        mo_coeff_spin,
        mo_energy_spin,
        raw_fock_spin,
        converged_arr,
        cycles_arr,
        _rms_history,
        _selected_cycle,
        _best_cycle,
        _selected_rms,
        _best_rms,
    ) = run_unrestricted_scf(
        fock_builder=fock_builder,
        density_spin=jnp.stack([density_a, density_b], axis=0),
        mo_coeff_spin=jnp.stack([mo_coeff_a, mo_coeff_b], axis=0),
        mo_occ_spin=jnp.stack([mo_occ_a, mo_occ_b], axis=0),
        mo_energy_spin=jnp.stack([mo_energy_a, mo_energy_b], axis=0),
        overlap=s,
        max_cycle=int(cfg.max_cycle),
        damping=float(cfg.damping),
        conv_tol=float(cfg.conv_tol),
        conv_tol_density=float(cfg.conv_tol_density),
        conv_tol_grad=float(cfg.conv_tol_grad),
        orthogonalization_eps=float(cfg.orthogonalization_eps),
        convergence_metric=str(cfg.convergence_metric),
        level_shift=float(cfg.level_shift),
        iterate_selection="best_rms",
        orthogonalizer=x,
    )
    density_a, density_b = density_spin[0], density_spin[1]
    mo_coeff_a, mo_coeff_b = mo_coeff_spin[0], mo_coeff_spin[1]
    mo_energy_a, mo_energy_b = mo_energy_spin[0], mo_energy_spin[1]
    fock_a, fock_b = raw_fock_spin[0], raw_fock_spin[1]
    if cfg.level_shift != 0.0:
        # Remove the shift from reported energies without reoccupying a different
        # unshifted root. Convergence establishes stationarity, not stability.
        mo_energy_a = jnp.diag(mo_coeff_a.T @ fock_a @ mo_coeff_a)
        mo_energy_b = jnp.diag(mo_coeff_b.T @ fock_b @ mo_coeff_b)
    energy, xc_energy, fock_a, fock_b = eval_state(
        density_a,
        density_b,
        mo_coeff_a,
        mo_coeff_b,
        mo_energy_a,
        mo_energy_b,
    )
    converged = bool(converged_arr) if not traceable_inputs else converged_arr
    cycles = int(cycles_arr) if not traceable_inputs else cycles_arr

    return UKSResult(
        converged=converged,
        total_energy=_host_float_unless_traced(energy),
        electronic_energy=_host_float_unless_traced(energy - enuc - (0. if dispersion_energy is None else dispersion_energy)),
        nuclear_repulsion=_host_float_unless_traced(enuc),
        xc_energy=_host_float_unless_traced(xc_energy),
        exact_exchange_fraction=float(alpha_scalar),
        mo_energy_alpha=mo_energy_a,
        mo_energy_beta=mo_energy_b,
        mo_coeff_alpha=mo_coeff_a,
        mo_coeff_beta=mo_coeff_b,
        mo_occ_alpha=mo_occ_a,
        mo_occ_beta=mo_occ_b,
        density_matrix_alpha=density_a,
        density_matrix_beta=density_b,
        fock_matrix_alpha=fock_a,
        fock_matrix_beta=fock_b,
        overlap_matrix=s,
        hcore_matrix=h,
        cycles=cycles,
        range_separated_coefficients=None if rsh is None else rsh.to_range_separated_hybrid_coefficients(),
        dispersion_energy=dispersion_energy,
    )
