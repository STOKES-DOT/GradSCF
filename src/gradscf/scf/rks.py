from __future__ import annotations

from functools import lru_cache, partial
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from .._memory import DEFAULT_MAX_MEMORY_MB, validate_memory_mb
from gradscf.integrals.basis import CartesianBasis
from gradscf.integrals import (
    build_j_from_eri_pair_matrix,
    build_jk_from_eri_pair_matrix,
)
from gradscf.integrals.molecular.jk import build_j_from_df, build_jk_from_df, build_jk_from_df_orbitals
from gradscf.integrals.molecular.factorization import eri_to_df_factors
from ..tools.features import (
    MoleculeLikeState,
    _spin_density_and_gradient,
    _spin_density_fields,
    _spin_tau,
    molecule_grid_view,
    restricted_grid_features_with_gradients,
)
from ..dft.xc import (
    hybrid_coeff,
    parse_xc,
    xc_type,
)
from ..dft.xc.evaluate import grid_value_and_grad
from .core import (
    _build_density_from_occ,
    _diagonalize_fock,
    _orthogonalizer,
    _validate_density_matrix,
)
from ._pytree import pytree_dataclass
from .rsh import local_xc_spec, validate_rsh_inputs
from .energy import XCContribution, restricted_energy, restricted_fock
from .iteration import run_restricted_scf
from ..solvers.scf import orbital_gradient_norm, orbital_orthogonalizer

JKBuilder = Callable[
    [Array, Array | None, Array | None, Array | None, Array | None, Array | None],
    tuple[Array, Array],
]


@dataclass(frozen=True)
class RKSConfig:
    """Configuration for restricted Kohn-Sham SCF iterations."""

    xc_spec: str = "pbe"
    max_cycle: int = 80
    conv_tol: float = 1e-10
    conv_tol_density: float = 1e-8
    conv_tol_grad: float = 1e-7
    damping: float = 0.0
    level_shift: float = 0.0
    orthogonalization_eps: float = 1e-10
    density_floor: float = 1e-12
    potential_clip: float | None = None
    jk_backend: Literal["full", "df", "direct"] = "full"
    df_tol: float = 1e-10
    df_max_rank: int | None = None
    auxbasis: str | None = None
    only_dfj: bool = False
    direct_scf_tol: float = 0.0
    direct_scf_workers: int = 1
    grid_block_size: int = 0
    grid_max_memory: float | None = None
    grid_screen_tol: float = 0.
    max_memory: float = field(default=DEFAULT_MAX_MEMORY_MB, kw_only=True)
    convergence_metric: Literal["energy_and_residual", "energy"] = "energy_and_residual"

    def __post_init__(self):
        if not 0 <= self.grid_screen_tol < float('inf'):
            raise ValueError("grid_screen_tol must be finite and nonnegative.")
        object.__setattr__(self, "max_memory", validate_memory_mb(self.max_memory))
        if self.grid_max_memory is not None:
            object.__setattr__(self, "grid_max_memory",
                               validate_memory_mb(self.grid_max_memory, name="grid_max_memory"))


@pytree_dataclass
@dataclass(frozen=True)
class RKSResult:
    """Canonical RKS PyTree; only the eager entry converts scalars to Python."""

    converged: Array | bool
    total_energy: Array | float
    electronic_energy: Array | float
    nuclear_repulsion: Array | float
    xc_energy: Array | float
    exact_exchange_fraction: Array | float
    mo_energy: Array
    mo_coeff: Array
    mo_occ: Array
    density_matrix: Array
    fock_matrix: Array
    overlap_matrix: Array
    hcore_matrix: Array
    cycles: Array | int
    # For RSH, exact_exchange_fraction is the short-range value. The tuple
    # identifies (omega, long-range fraction, short-range fraction).
    range_separated_coefficients: tuple[float, float, float] | None = field(default=None, kw_only=True)
    dispersion_energy: Array | float | None = field(default=None, kw_only=True)


# Preserve the historical import/constructor while sharing the result contract.
TraceableRKSResult = RKSResult


def _closed_shell_mo_occ(nao: int, nocc: int, dtype: Array) -> Array:
    return jnp.zeros((nao,), dtype=dtype).at[:nocc].set(2.0)


def _build_jk(eri: Array, density: Array, *, with_j=True, with_k=True) -> tuple[Array, Array]:
    eri_arr = jnp.asarray(eri)
    if eri_arr.ndim in (1,2):
        from gradscf.integrals.molecular.jk import build_jk_from_packed
        return build_jk_from_packed(eri_arr, density, with_j=with_j, with_k=with_k)
    if int(eri_arr.size) == 0:
        raise ValueError("JK build requires full AO ERI or packed AO-pair ERI data.")
    j_mat = (jnp.einsum("pqrs,rs->pq", eri_arr, density, precision=Precision.HIGHEST)
             if with_j else jnp.zeros_like(density, dtype=jnp.result_type(eri_arr, density)))
    from gradscf.integrals.molecular.jk import exchange_matrix
    k_mat = exchange_matrix(eri_arr, density) if with_k else jnp.zeros_like(j_mat)
    return j_mat, k_mat


def _mo_residual_norm(fock: Array, mo_coeff: Array, mo_occ: Array) -> Array:
    return orbital_gradient_norm(fock, mo_coeff, mo_occ)


@partial(jax.custom_jvp, nondiff_argnums=(4,))
def _df_jk_with_orbital_hint(factors, density, mo_coeff, mo_occ, nocc):
    """Orbitals accelerate a consistent primal; density defines its response."""
    orbital_density = _build_density_from_occ(mo_coeff, mo_occ)
    tolerance = 64*jnp.finfo(density.dtype).eps*jnp.maximum(1., jnp.max(jnp.abs(density)))
    consistent = jnp.max(jnp.abs(density-orbital_density)) <= tolerance
    return jax.lax.cond(
        consistent,
        lambda _: build_jk_from_df_orbitals(factors, density, mo_coeff, mo_occ, nocc=nocc),
        lambda _: build_jk_from_df(factors, density), None)


@_df_jk_with_orbital_hint.defjvp
def _df_jk_with_orbital_hint_jvp(nocc, primals, tangents):
    factors, density, coeff, occ = primals
    dfactors, ddensity, _, _ = tangents
    value = _df_jk_with_orbital_hint(factors, density, coeff, occ, nocc)
    # A numerically coincident guess/damped density need not share its tangent
    # with the orbital hint. Differentiate the physical J/K operands instead.
    _, tangent = jax.jvp(build_jk_from_df, (factors, density), (dfactors, ddensity))
    return value, tangent


def _make_jk_builder(
    eri: Array | None,
    cfg: RKSConfig,
    *,
    eri_pair_matrix: Array | None = None,
    df_factors: Array | None = None,
    direct_basis: CartesianBasis | None = None,
    with_k: bool = True,
    nocc: int | None = None,
    eri_lr: Array | None = None,
) -> JKBuilder:
    from .rsh import rsh_preset
    rsh = rsh_preset(cfg.xc_spec)
    if rsh is not None:
        if cfg.jk_backend == "direct" or (cfg.jk_backend == "df" and cfg.only_dfj):
            from .rsh import _require_direct_rsh_provider
            _require_direct_rsh_provider(direct_basis)
        elif cfg.jk_backend == "full":
            if eri_lr is None or eri is None or jnp.ndim(eri) != 4:
                raise ValueError("Range-separated XC requires full eri and eri_lr.")
            exchange_eri = (rsh.default_sr_hf_fraction * jnp.asarray(eri)
                            + (rsh.default_lr_hf_fraction-rsh.default_sr_hf_fraction) * jnp.asarray(eri_lr))
        else:
            raise NotImplementedError("Range-separated XC requires full ERIs or canonical native direct J/K.")
    def _direct_builder(*, with_j=True) -> JKBuilder:
        if direct_basis is None:
            raise ValueError("jk_backend='direct' requires direct_basis.")

        threshold = float(cfg.direct_scf_tol)
        from gradscf.integrals.backends.native.jk import (
            NativeDirectBasis, ProjectedNativeDirectBasis, _prepare_direct_basis)
        if isinstance(direct_basis,(NativeDirectBasis,ProjectedNativeDirectBasis)):
            provider = _prepare_direct_basis(direct_basis)
            def _native_direct(density,mo_coeff=None,mo_occ=None,density_last=None,j_last=None,k_last=None):
                del mo_coeff,mo_occ
                delta=density if density_last is None else density-density_last
                controls = dict(screening_threshold=threshold, workers=cfg.direct_scf_workers,
                                with_j=with_j, with_k=with_k)
                if type(provider) is NativeDirectBasis and threshold > 0 and density_last is not None:
                    # Dynamic S selects a discrete quartet subset; its indicator
                    # response is handled by the native derivative rule.
                    controls["screen_density"] = delta
                if rsh is not None:
                    from .rsh import _direct_rsh_jk
                    j,k=_direct_rsh_jk(provider,delta,rsh.to_range_separated_hybrid_coefficients(),
                                      **controls)
                else:
                    j,k=provider.get_jk(delta, **controls)
                if density_last is not None:
                    if with_j:j=j+j_last
                    if with_k:k=k+k_last
                return j,k if with_k else jnp.zeros_like(density)
            return _native_direct
        raise TypeError('Direct SCF requires a libcint AO provider; use the molecular direct() interface.')

    if cfg.jk_backend == "full":
        pair_arr = None if eri_pair_matrix is None else jnp.asarray(eri_pair_matrix)
        eri_arr = None if eri is None else jnp.asarray(eri)
        if eri_arr is None and pair_arr is None:
            if direct_basis is not None:
                return _direct_builder()
            raise ValueError("jk_backend='full' requires full AO ERI, packed AO-pair ERI, or direct_basis.")
        if not with_k:
            def _full_j(
                density: Array,
                mo_coeff: Array | None = None,
                mo_occ: Array | None = None,
                density_last: Array | None = None,
                j_last: Array | None = None,
                k_last: Array | None = None,
            ):
                del mo_coeff, mo_occ, density_last, j_last, k_last
                if pair_arr is not None:
                    return build_j_from_eri_pair_matrix(pair_arr, density), jnp.zeros_like(density)
                if eri_arr is None:
                    raise ValueError("jk_backend='full' requires full AO ERI or packed AO-pair ERI.")
                if eri_arr.ndim in (1,2):
                    return _build_jk(eri_arr, density, with_k=False)
                return (
                    jnp.einsum("pqrs,rs->pq", eri_arr, density, precision=Precision.HIGHEST),
                    jnp.zeros_like(density),
                )

            return _full_j

        def _full_jk(
            density: Array,
            mo_coeff: Array | None = None,
            mo_occ: Array | None = None,
            density_last: Array | None = None,
            j_last: Array | None = None,
            k_last: Array | None = None,
        ):
            del mo_coeff, mo_occ, density_last, j_last, k_last
            if rsh is not None:
                j_mat = jnp.einsum("pqrs,rs->pq", eri_arr, density, precision=Precision.HIGHEST)
                k_mat = jnp.einsum("prqs,rs->pq", exchange_eri, density, precision=Precision.HIGHEST)
                return j_mat, k_mat
            if pair_arr is not None:
                return build_jk_from_eri_pair_matrix(pair_arr, density)
            if eri_arr is None:
                raise ValueError("jk_backend='full' requires full AO ERI or packed AO-pair ERI.")
            return _build_jk(eri_arr, density)

        return _full_jk
    if cfg.jk_backend == "df":
        if df_factors is None:
            if eri is None:
                raise ValueError("jk_backend='df' requires either eri or df_factors.")
            df_factors = eri_to_df_factors(
                eri,
                tol=cfg.df_tol,
                max_rank=cfg.df_max_rank,
            )
        factors = jnp.asarray(df_factors)
        if cfg.only_dfj and with_k:
            exact_exchange = _direct_builder(with_j=False)

            def _df_j_direct_k(density, mo_coeff=None, mo_occ=None,
                               density_last=None, j_last=None, k_last=None):
                _, exchange = exact_exchange(density, mo_coeff, mo_occ,
                    density_last, j_last, k_last)
                return build_j_from_df(factors, density), exchange

            return _df_j_direct_k
        if not with_k:
            def _df_j(
                density: Array,
                mo_coeff: Array | None = None,
                mo_occ: Array | None = None,
                density_last: Array | None = None,
                j_last: Array | None = None,
                k_last: Array | None = None,
            ):
                del mo_coeff, mo_occ, density_last, j_last, k_last
                return build_j_from_df(factors, density), jnp.zeros_like(density)

            return _df_j

        use_orbital_k = nocc is not None and jnp.finfo(factors.dtype).bits >= 64

        def _df_jk(
            density: Array,
            mo_coeff: Array | None = None,
            mo_occ: Array | None = None,
            density_last: Array | None = None,
            j_last: Array | None = None,
            k_last: Array | None = None,
        ):
            del density_last, j_last, k_last
            if use_orbital_k and mo_coeff is not None and mo_occ is not None:
                # Density-only guesses and density damping need K[D], not
                # K from orbitals that describe a different density.
                return _df_jk_with_orbital_hint(factors, density, mo_coeff, mo_occ, int(nocc))
            return build_jk_from_df(factors, density)

        return _df_jk
    if cfg.jk_backend == "direct":
        return _direct_builder()
    raise ValueError(
        f"Unsupported jk_backend={cfg.jk_backend!r}. Choose one of {{'full', 'df', 'direct'}}."
    )


def _restricted_spin_view(
    *,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    mo_energy: Array,
):
    density_spin = jnp.stack([0.5 * density, 0.5 * density], axis=0)
    coeff_spin = jnp.stack([mo_coeff, mo_coeff], axis=0)
    occ_spin = jnp.stack([0.5 * mo_occ, 0.5 * mo_occ], axis=0)
    energy_spin = jnp.stack([mo_energy, mo_energy], axis=0)
    return MoleculeLikeState(
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid=molecule_grid_view(weights),
        rdm1=density_spin,
        mo_coeff=coeff_spin,
        mo_occ=occ_spin,
        mo_energy=energy_spin,
    )


@lru_cache(maxsize=64)
def _array_xc_value_and_grad_kernel(
    xc_spec: str,
    xc_kind: str,
    density_floor: float,
) -> Callable[[Array], tuple[Array, Array]]:
    def kernel(variables: Array) -> tuple[Array, Array]:
        rho = jnp.maximum(variables[:, :1], density_floor)
        blocks = [rho/2, rho/2]
        if xc_kind in {"GGA", "MGGA"}:
            blocks.extend([variables[:, 1:4]/2, variables[:, 1:4]/2])
        if xc_kind == "MGGA":
            tau = jnp.maximum(variables[:, 4:5], 0.)
            blocks.extend([tau/2, tau/2])
        values, spin_grad = grid_value_and_grad(xc_spec, xc_kind)(jnp.concatenate(blocks, axis=-1))
        gradients = [(spin_grad[:, :1]+spin_grad[:, 1:2])/2]
        if xc_kind in {"GGA", "MGGA"}:
            gradients.append((spin_grad[:, 2:5]+spin_grad[:, 5:8])/2)
        if xc_kind == "MGGA":
            gradients.append((spin_grad[:, 8:9]+spin_grad[:, 9:10])/2)
        return values, jnp.concatenate(gradients, axis=-1)

    return jax.jit(kernel)


def _xc_energy_and_potential_on_grid(
    *,
    rho: Array,
    grad: Array,
    tau: Array,
    weights: Array,
    xc_spec: str,
    density_floor: float,
    potential_clip: float | None,
    xc_kind: str,
    return_tau: bool = False,
) -> tuple[Array, ...]:
    if xc_kind == "MGGA" and not return_tau:
        raise ValueError("MGGA grid potentials require return_tau=True to retain the kinetic-density potential.")
    rho = jnp.maximum(jnp.asarray(rho), density_floor)
    grad = jnp.asarray(grad)
    tau = jnp.maximum(jnp.asarray(tau), 0.0)
    weights = jnp.asarray(weights)

    if xc_kind == "HF":
        zeros = jnp.zeros_like(rho)
        result = (jnp.asarray(0.0, dtype=rho.dtype), zeros, jnp.zeros((rho.shape[0], 3), dtype=rho.dtype))
        return result + (zeros,) if return_tau else result

    if xc_kind == "LDA":
        response_variables = rho[..., None]
    elif xc_kind == "GGA":
        response_variables = jnp.concatenate([rho[..., None], grad], axis=-1)
    elif xc_kind == "MGGA":
        response_variables = jnp.concatenate([rho[..., None], grad, tau[..., None]], axis=-1)
    else:
        raise ValueError(f"Unsupported XC kind={xc_kind!r}.")

    mask = rho > density_floor
    # The quadrature cutoff applies to energy and all potential channels.
    # Evaluate inactive points at a regular dummy input before masking, so
    # invalid tail branches cannot contaminate higher derivatives.
    response_variables = jnp.where(mask[:, None], response_variables,
                                   jnp.ones_like(response_variables))
    point_exc, point_grad = _array_xc_value_and_grad_kernel(
        xc_spec,
        xc_kind,
        density_floor,
    )(response_variables)
    point_exc = jnp.where(mask, point_exc, 0.)
    exc = jnp.tensordot(weights, point_exc, axes=(0, 0))

    vxc_rho = jnp.where(mask, point_grad[:, 0], 0.0)
    if xc_kind in {"GGA", "MGGA"}:
        vxc_grad = jnp.where(mask[:, None], point_grad[:, 1:4], 0.0)
    else:
        vxc_grad = jnp.zeros((rho.shape[0], 3), dtype=rho.dtype)
    vxc_tau = jnp.where(mask, point_grad[:, 4], 0.) if xc_kind == "MGGA" else jnp.zeros_like(rho)

    if potential_clip is not None:
        clip = jnp.asarray(potential_clip, dtype=rho.dtype)
        vxc_rho = jnp.clip(vxc_rho, -clip, clip)
        vxc_grad = jnp.clip(vxc_grad, -clip, clip)
        vxc_tau = jnp.clip(vxc_tau, -clip, clip)
    result = (exc, vxc_rho, vxc_grad)
    return result + (vxc_tau,) if return_tau else result


def _vxc_matrix_from_grid_potential(
    *,
    ao: Array,
    ao_deriv1: Array | None,
    ao_laplacian: Array | None,
    weights: Array,
    vxc_rho: Array,
    vxc_grad: Array,
    vxc_tau: Array,
    vxc_lapl: Array,
    xc_kind: str,
) -> Array:
    if xc_kind in {"GGA", "MGGA", "MGGA_LAPL"}:
        weighted_ao = .5 * (weights * vxc_rho)[:, None] * ao
        weighted_ao = weighted_ao + jnp.einsum(
            "rx,xrp->rp", weights[:, None] * vxc_grad, ao_deriv1[1:4],
            precision=Precision.HIGHEST,
        )
        gram = jnp.einsum("rp,rq->pq", weighted_ao, ao, precision=Precision.HIGHEST)
        vxc_matrix = gram + gram.T
    else:
        vxc_matrix = jnp.einsum(
            "r,rp,rq->pq", weights * vxc_rho, ao, ao,
            precision=Precision.HIGHEST,
        )
    if xc_kind in {"MGGA", "MGGA_LAPL"}:
        tau_term = 0.5 * jnp.einsum(
            "r,xrp,xrq->pq",
            weights * vxc_tau,
            ao_deriv1[1:4],
            ao_deriv1[1:4],
            precision=Precision.HIGHEST,
        )
        vxc_matrix = vxc_matrix + tau_term
    if xc_kind == "MGGA_LAPL":
        lapl_left = jnp.einsum(
            "r,rp,rq->pq",
            weights * vxc_lapl,
            ao_laplacian,
            ao,
            precision=Precision.HIGHEST,
        )
        lapl_grad = 2.0 * jnp.einsum(
            "r,xrp,xrq->pq",
            weights * vxc_lapl,
            ao_deriv1[1:4],
            ao_deriv1[1:4],
            precision=Precision.HIGHEST,
        )
        vxc_matrix = vxc_matrix + lapl_left + lapl_left.T + lapl_grad
    return 0.5 * (vxc_matrix + vxc_matrix.T)


def _restricted_xc_energy_and_matrix(
    *,
    density: Array,
    ao: Array,
    ao_deriv1: Array | None,
    weights: Array,
    cfg: RKSConfig,
    xc_kind: str,
    grid_ao: Any | None = None,
    mo_coeff: Array | None = None,
    mo_occ: Array | None = None,
    occupied_columns: int | None = None,
) -> tuple[Array, Array]:
    """Integrate built-in XC, reducing AO blocks before advancing the grid."""
    dtype = jnp.result_type(density, ao, weights)
    initial = (jnp.asarray(0., dtype=dtype), jnp.zeros(density.shape, dtype=dtype))
    if xc_kind == "HF":
        return initial

    from gradscf.integrals.grids.blocks import _ao_submatrix, _add_ao_submatrix

    def contribution(carry, block_ao, derivative, laplacian, coords, block_weights, indices=None):
        del laplacian, coords
        block_density = _ao_submatrix(density, indices)
        # Only measured MGGA block sizes enable automatic orbital hints.
        use_orbitals = (xc_kind == "MGGA" and block_ao.shape[1] >= 128
                        and block_ao.shape[0] >= 256 and mo_coeff is not None
                        and mo_occ is not None and occupied_columns is not None
                        and 0 < occupied_columns < block_ao.shape[1])
        if use_orbitals:
            coeff = mo_coeff if indices is None else mo_coeff[indices]
            rho, grad, tau = _spin_density_fields(
                block_ao, derivative, block_density, xc_kind=xc_kind,
                mo_coeff=coeff, mo_occ=mo_occ, occupied_columns=occupied_columns)
        else:
            if xc_kind == "LDA":
                rho = jnp.einsum("rp,pq,rq->r", block_ao, block_density, block_ao,
                                 precision=Precision.HIGHEST)
                grad = jnp.zeros((block_ao.shape[0], 3), dtype=rho.dtype)
            else:
                rho, grad = _spin_density_and_gradient(block_ao, derivative, block_density)
            tau = _spin_tau(derivative, block_density) if xc_kind == "MGGA" else jnp.zeros_like(rho)
        energy, v_rho, v_grad, v_tau = _xc_energy_and_potential_on_grid(
            rho=rho, grad=grad, tau=tau, weights=block_weights,
            xc_spec=cfg.xc_spec, density_floor=cfg.density_floor,
            potential_clip=cfg.potential_clip, xc_kind=xc_kind, return_tau=True)
        potential = _vxc_matrix_from_grid_potential(
            ao=block_ao, ao_deriv1=derivative, ao_laplacian=None, weights=block_weights,
            vxc_rho=v_rho, vxc_grad=v_grad, vxc_tau=v_tau,
            vxc_lapl=jnp.zeros_like(rho), xc_kind=xc_kind)
        return carry[0] + energy, _add_ao_submatrix(carry[1], potential, indices)

    if grid_ao is None:
        return contribution(initial, ao, ao_deriv1, None, None, weights)
    reduce = getattr(grid_ao, 'reduce_local', grid_ao.reduce)
    return reduce(contribution, initial, deriv=0 if xc_kind == "LDA" else 1)


def _fock_components_for_density(
    *,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    mo_energy: Array,
    jk_builder: JKBuilder,
    density_last: Array | None = None,
    j_last: Array | None = None,
    k_last: Array | None = None,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    h: Array,
    alpha: Array,
    cfg: RKSConfig,
    xc_kind: str,
    grid_ao: Any | None = None,
    occupied_columns: int | None = None,
) -> tuple[Array, Array, Array, Array]:
    j_mat, k_mat = jk_builder(density, mo_coeff, mo_occ, density_last, j_last, k_last)
    del mo_energy
    xc_energy, vxc_matrix = _restricted_xc_energy_and_matrix(
        density=density, ao=ao, ao_deriv1=ao_deriv1, weights=weights,
        cfg=cfg, xc_kind=xc_kind, grid_ao=grid_ao,
        mo_coeff=mo_coeff, mo_occ=mo_occ, occupied_columns=occupied_columns)
    xc = XCContribution(xc_energy,vxc_matrix,alpha)
    return j_mat,k_mat,xc,restricted_fock(h,j_mat,k_mat,xc)


def _energy_and_raw_fock_for_density(
    *,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    mo_energy: Array,
    jk_builder: JKBuilder,
    density_last: Array | None = None,
    j_last: Array | None = None,
    k_last: Array | None = None,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    h: Array,
    enuc: Array,
    alpha: Array,
    cfg: RKSConfig,
    xc_kind: str,
    grid_ao: Any | None = None,
    occupied_columns: int | None = None,
) -> tuple[Array, Array, Array, Array, Array]:
    j_mat, k_mat, xc, fock = _fock_components_for_density(
        density=density,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        jk_builder=jk_builder,
        density_last=density_last,
        j_last=j_last,
        k_last=k_last,
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        h=h,
        alpha=alpha,
        cfg=cfg,
        xc_kind=xc_kind,
        grid_ao=grid_ao,
        occupied_columns=occupied_columns,
    )

    total=restricted_energy(density,h,j_mat,k_mat,xc,nuclear_repulsion=enuc)
    return total,xc.energy,fock,j_mat,k_mat


def _maybe_run_extra_final_cycle(
    *,
    density: Array,
    mo_coeff: Array,
    mo_energy: Array,
    energy: Array,
    xc_energy: Array,
    raw_fock: Array,
    converged,
    x: Array,
    jk_builder: JKBuilder,
    ao: Array,
    ao_deriv1: Array,
    weights: Array,
    h: Array,
    s: Array,
    enuc: Array,
    cfg: RKSConfig,
    alpha: Array,
    xc_kind: str,
    mo_occ_fixed: Array,
    j_mat: Array,
    k_mat: Array,
    traceable: bool,
) -> tuple[Array, Array, Array, Array, Array, Array, Any]:
    # A final Aufbau refill can leave the selected stationary branch. Keep its
    # density/orbitals and report unshifted raw-Fock expectations instead.
    if cfg.level_shift != 0.0:
        mo_energy = jnp.real(jnp.diag(mo_coeff.conj().T @ raw_fock @ mo_coeff))
    return density, mo_coeff, mo_energy, energy, xc_energy, raw_fock, converged


@partial(jax.jit, static_argnames=(
    "cfg", "xc_kind", "nocc", "with_k", "skip_first_fock_damping",
))
def _run_rks_iterations(
    *, h, s, x, eri, eri_pair_matrix, df_factors, direct_basis,
    ao, ao_deriv1, weights, grid_ao, enuc, alpha, cfg, xc_kind, nocc,
    with_k, skip_first_fock_damping, density, mo_coeff, mo_occ, mo_energy,
    raw_fock, j_mat, k_mat, eri_lr,
):
    """Reuse the SCF executable with all physical inputs as dynamic leaves.

    Build the local callbacks inside this stable JIT boundary. Creating a new
    integral/grid/provider object must not create another compiled scan body.
    """
    jk_builder = _make_jk_builder(
        eri, cfg, eri_pair_matrix=eri_pair_matrix, df_factors=df_factors,
        direct_basis=direct_basis, with_k=with_k, nocc=nocc, eri_lr=eri_lr,
    )

    def energy_and_fock_builder(density_i, mo_coeff_i, mo_occ_i, mo_energy_i,
                                density_last, j_last, k_last):
        return _energy_and_raw_fock_for_density(
            density=density_i, mo_coeff=mo_coeff_i, mo_occ=mo_occ_i,
            mo_energy=mo_energy_i, jk_builder=jk_builder,
            density_last=density_last, j_last=j_last, k_last=k_last,
            ao=ao, ao_deriv1=ao_deriv1, weights=weights, h=h, enuc=enuc,
            alpha=alpha, cfg=cfg, xc_kind=xc_kind, grid_ao=grid_ao, occupied_columns=nocc,
        )

    return run_restricted_scf(
        h=h, s=s, x=x, energy_and_fock_builder=energy_and_fock_builder,
        cfg=cfg, mo_occ_fixed=mo_occ, diis_basis=mo_coeff,
        skip_first_fock_damping=skip_first_fock_damping,
        density=density, mo_coeff=mo_coeff, mo_occ=mo_occ,
        mo_energy=mo_energy, raw_fock=raw_fock, j_mat=j_mat, k_mat=k_mat,
        refresh_on_convergence=(direct_basis is not None and eri is None
                                and eri_pair_matrix is None),
        store_iterates=False,
    )


def _run_rks_from_integrals_shared(
    *,
    overlap: Array,
    hcore: Array,
    eri: Array | None,
    eri_pair_matrix: Array | None = None,
    nelectron: int,
    nuclear_repulsion: float | Array,
    ao: Array,
    ao_deriv1: Array,
    grid_weights: Array,
    grid_ao: Any | None = None,
    df_factors: Array | None = None,
    direct_basis: CartesianBasis | None = None,
    init_density: Array | None = None,
    init_mo_coeff: Array | None = None,
    init_mo_occ: Array | None = None,
    init_mo_energy: Array | None = None,
    config: RKSConfig | None = None,
    eri_lr: Array | None = None,
    dispersion_energy: Array | None = None,
    traceable: bool,
) -> RKSResult:
    cfg = RKSConfig() if config is None else config
    rsh = validate_rsh_inputs(cfg.xc_spec, eri, eri_lr, dispersion_energy,
        jk_backend="direct" if cfg.only_dfj and cfg.jk_backend == "df" else cfg.jk_backend,
        direct_basis=direct_basis)
    parse_xc(local_xc_spec(cfg.xc_spec))
    xc_kind = xc_type(local_xc_spec(cfg.xc_spec))

    s = jnp.asarray(overlap)
    h = jnp.asarray(hcore)
    eri_arr = None if eri is None else jnp.asarray(eri)
    eri_pair_arr = None if eri_pair_matrix is None else jnp.asarray(eri_pair_matrix)
    ao = jnp.asarray(ao)
    ao_deriv1 = jnp.asarray(ao_deriv1)
    weights = jnp.asarray(grid_weights)
    enuc = jnp.asarray(nuclear_repulsion, dtype=h.dtype)
    nao = int(s.shape[0])

    x = orbital_orthogonalizer(s, cfg.orthogonalization_eps)
    if init_mo_coeff is None:
        mo_energy, mo_coeff = _diagonalize_fock(h, x)
    else:
        mo_coeff = jnp.asarray(init_mo_coeff)
        if mo_coeff.ndim == 3:
            mo_coeff = mo_coeff[0]
        if init_mo_energy is None:
            mo_energy, _ = _diagonalize_fock(h, x)
        else:
            mo_energy = jnp.asarray(init_mo_energy)
            if mo_energy.ndim == 2:
                mo_energy = mo_energy[0]
    if init_mo_occ is None:
        if nelectron % 2 != 0:
            raise ValueError("RKS requires an even number of electrons when init_mo_occ is not provided.")
        nocc = nelectron // 2
        if nocc <= 0 or nocc > nao:
            raise ValueError("Invalid occupation count for RKS.")
        mo_occ_fixed = _closed_shell_mo_occ(nao, nocc, h.dtype)
    else:
        mo_occ_fixed = jnp.asarray(init_mo_occ, dtype=h.dtype)
        if mo_occ_fixed.ndim == 2:
            mo_occ_fixed = mo_occ_fixed[0]
        if mo_occ_fixed.ndim != 1:
            raise ValueError("init_mo_occ must be a 1D occupation vector for RKS.")
        if float(jnp.max(mo_occ_fixed)) <= 1.0 + 1e-6:
            mo_occ_fixed = mo_occ_fixed * 2.0
        if int(mo_occ_fixed.shape[0]) != nao:
            raise ValueError("init_mo_occ must match the AO/MO dimension for RKS.")
        nocc = int(jnp.count_nonzero(mo_occ_fixed > jnp.asarray(1e-12, dtype=h.dtype)))
    density = _build_density_from_occ(mo_coeff, mo_occ_fixed)

    init_density_matrix = _validate_density_matrix(
        init_density,
        nao=nao,
        dtype=h.dtype,
        label="init_density",
        method="RKS",
    )
    if init_density_matrix is not None:
        density = init_density_matrix

    skip_first_fock_damping = (init_mo_coeff is not None) or (init_density_matrix is not None)
    alpha_scalar = 1.0 if rsh is not None else float(hybrid_coeff(cfg.xc_spec))
    alpha = jnp.asarray(alpha_scalar, dtype=h.dtype)
    if cfg.jk_backend == "df" and df_factors is None and eri_arr is not None:
        # The existing rank-selecting spectral compression is a host step.
        # Pass its factors to both Fock builders, outside the compiled loop.
        df_factors = eri_to_df_factors(eri_arr, tol=cfg.df_tol, max_rank=cfg.df_max_rank)
    if grid_ao is not None and xc_kind in {"LDA", "GGA", "MGGA"}:
        from gradscf._memory import tree_storage_bytes
        from .inputs.grid import prepare_scf_grid, _scf_operator_reservation

        resident = tree_storage_bytes(s, h, eri_arr, eri_pair_arr, eri_lr,
                                      df_factors, ao, ao_deriv1, density, mo_coeff)
        # Built-in forward SCF keeps DIIS/state matrices, not max_cycle copies.
        resident += 32 * s.size * max(8, h.dtype.itemsize)
        operator = _scf_operator_reservation(df_factors, direct_basis,
            with_k=abs(alpha_scalar) > 1e-14, only_dfj=cfg.only_dfj,
            density_dtype=density.dtype, workers=cfg.direct_scf_workers, cutoff=cfg.direct_scf_tol)
        if operator is not None:
            device_work, host_work = operator
            grid_ao = prepare_scf_grid(grid_ao, max_memory=cfg.max_memory,
                resident_bytes=resident + device_work, host_workspace_bytes=host_work,
                deriv=0 if xc_kind == "LDA" else 1, screen_tol=cfg.grid_screen_tol)
    jk_builder = _make_jk_builder(
        eri_arr,
        cfg,
        eri_pair_matrix=eri_pair_arr,
        df_factors=df_factors,
        direct_basis=direct_basis,
        with_k=bool(abs(alpha_scalar) > 1e-14),
        nocc=nocc,
        eri_lr=eri_lr,
    )
    j_mat, k_mat, _, fock = _fock_components_for_density(
        density=density,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ_fixed,
        mo_energy=mo_energy,
        jk_builder=jk_builder,
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        h=h,
        alpha=alpha,
        cfg=cfg,
        xc_kind=xc_kind,
        grid_ao=grid_ao,
        occupied_columns=nocc,
    )

    (
        converged,
        cycles,
        density,
        mo_coeff,
        mo_energy,
        energy,
        xc_energy,
        fock,
        j_mat,
        k_mat,
        _density_history,
        _mo_coeff_history,
        _mo_energy_history,
        _density_rms_history,
    ) = _run_rks_iterations(
        h=h, s=s, x=x, eri=eri_arr, eri_pair_matrix=eri_pair_arr,
        df_factors=df_factors,
        direct_basis=(direct_basis if cfg.jk_backend == "direct" or cfg.only_dfj or
                      (cfg.jk_backend == "full" and eri_arr is None and eri_pair_arr is None)
                      else None),
        ao=ao, ao_deriv1=ao_deriv1, weights=weights, grid_ao=grid_ao,
        enuc=enuc, alpha=alpha, cfg=cfg, xc_kind=xc_kind, nocc=nocc,
        with_k=bool(abs(alpha_scalar) > 1e-14),
        skip_first_fock_damping=skip_first_fock_damping,
        density=density, mo_coeff=mo_coeff, mo_occ=mo_occ_fixed,
        mo_energy=mo_energy, raw_fock=fock, j_mat=j_mat, k_mat=k_mat,
        eri_lr=eri_lr,
    )
    density, mo_coeff, mo_energy, energy, xc_energy, fock, converged = _maybe_run_extra_final_cycle(
        density=density,
        mo_coeff=mo_coeff,
        mo_energy=mo_energy,
        energy=energy,
        xc_energy=xc_energy,
        raw_fock=fock,
        converged=converged,
        x=x,
        jk_builder=jk_builder,
        ao=ao,
        ao_deriv1=ao_deriv1,
        weights=weights,
        h=h,
        s=s,
        enuc=enuc,
        cfg=cfg,
        alpha=alpha,
        xc_kind=xc_kind,
        mo_occ_fixed=mo_occ_fixed,
        j_mat=j_mat,
        k_mat=k_mat,
        traceable=traceable,
    )
    total_energy = energy if dispersion_energy is None else energy + dispersion_energy
    return RKSResult(
        converged=jnp.asarray(converged) & jnp.isfinite(total_energy),
        total_energy=total_energy,
        electronic_energy=energy - enuc,
        nuclear_repulsion=enuc,
        xc_energy=xc_energy,
        exact_exchange_fraction=alpha if rsh is None else jnp.asarray(rsh.default_sr_hf_fraction),
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ_fixed,
        density_matrix=density,
        fock_matrix=fock,
        overlap_matrix=s,
        hcore_matrix=h,
        cycles=jnp.asarray(cycles),
        range_separated_coefficients=None if rsh is None else rsh.to_range_separated_hybrid_coefficients(),
        dispersion_energy=dispersion_energy,
    )


def run_rks_from_integrals(
    *,
    overlap: Array,
    hcore: Array,
    eri: Array | None,
    eri_pair_matrix: Array | None = None,
    nelectron: int,
    nuclear_repulsion: float | Array,
    ao: Array,
    ao_deriv1: Array,
    grid_weights: Array,
    grid_ao: Any | None = None,
    df_factors: Array | None = None,
    direct_basis: CartesianBasis | None = None,
    init_density: Array | None = None,
    init_mo_coeff: Array | None = None,
    init_mo_occ: Array | None = None,
    init_mo_energy: Array | None = None,
    config: RKSConfig | None = None,
    eri_lr: Array | None = None,
    dispersion_energy: Array | None = None,
) -> RKSResult:
    """Run restricted Kohn-Sham SCF from AO integrals and numerical grid data."""
    result = _run_rks_from_integrals_shared(
        overlap=overlap,
        hcore=hcore,
        eri=eri,
        eri_pair_matrix=eri_pair_matrix,
        nelectron=nelectron,
        nuclear_repulsion=nuclear_repulsion,
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=grid_weights,
        grid_ao=grid_ao,
        df_factors=df_factors,
        direct_basis=direct_basis,
        init_density=init_density,
        init_mo_coeff=init_mo_coeff,
        init_mo_occ=init_mo_occ,
        init_mo_energy=init_mo_energy,
        config=config,
        eri_lr=eri_lr,
        dispersion_energy=dispersion_energy,
        traceable=False,
    )

    return replace(
        result,
        converged=bool(result.converged),
        total_energy=float(result.total_energy),
        electronic_energy=float(result.electronic_energy),
        nuclear_repulsion=float(result.nuclear_repulsion),
        xc_energy=float(result.xc_energy),
        exact_exchange_fraction=float(result.exact_exchange_fraction),
        cycles=int(result.cycles),
    )


def run_rks_from_integrals_traceable(
    *,
    overlap: Array,
    hcore: Array,
    eri: Array | None,
    eri_pair_matrix: Array | None = None,
    nelectron: int,
    nuclear_repulsion: float | Array,
    ao: Array,
    ao_deriv1: Array,
    grid_weights: Array,
    grid_ao: Any | None = None,
    df_factors: Array | None = None,
    direct_basis: CartesianBasis | None = None,
    init_density: Array | None = None,
    init_mo_coeff: Array | None = None,
    init_mo_occ: Array | None = None,
    init_mo_energy: Array | None = None,
    config: RKSConfig | None = None,
    eri_lr: Array | None = None,
    dispersion_energy: Array | None = None,
) -> TraceableRKSResult:
    """Traceable RKS SCF from AO integrals (array-valued outputs for autodiff)."""
    return _run_rks_from_integrals_shared(
        overlap=overlap,
        hcore=hcore,
        eri=eri,
        eri_pair_matrix=eri_pair_matrix,
        nelectron=nelectron,
        nuclear_repulsion=nuclear_repulsion,
        ao=ao,
        ao_deriv1=ao_deriv1,
        grid_weights=grid_weights,
        grid_ao=grid_ao,
        df_factors=df_factors,
        direct_basis=direct_basis,
        init_density=init_density,
        init_mo_coeff=init_mo_coeff,
        init_mo_occ=init_mo_occ,
        init_mo_energy=init_mo_energy,
        config=config,
        eri_lr=eri_lr,
        dispersion_energy=dispersion_energy,
        traceable=True,
    )
