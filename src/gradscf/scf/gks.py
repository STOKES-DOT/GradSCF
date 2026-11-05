"""Complex generalized Kohn-Sham SCF in an alpha-then-beta AO basis."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ..dft.xc import grid_energy, grid_value_and_grad, hybrid_coeff, parse_recipe, xc_type
from .core import _contains_jax_tracer, _host_float_unless_traced
from .facade import _BaseKS
from gradscf.scf.inputs.assembly import build_uks_integral_inputs
from ..solvers.scf import SCFConfig, SCFState, orbital_gradient_norm, orbital_orthogonalizer, solve_scf
from .rks import RKSResult
from .uks import UKSConfig
from .uks import _unrestricted_vxc_matrices, _unrestricted_xc_energy_and_potential_on_grid


@dataclass(frozen=True)
class GKSConfig(UKSConfig):
    """Generalized SCF controls: collinear LDA/GGA or noncollinear LDA XC."""

    collinear: str = "col"
    conv_tol_grad: float = 1e-7


@dataclass(frozen=True)
class GKSResult(RKSResult):
    """Spinor orbitals/density/Fock of size 2*nao and single occupations."""

    orbital_gradient_norm: float


def _hermitian(a: Array) -> Array:
    return (a + a.conj().T) * 0.5


def _spin_diagonal(a: Array) -> Array:
    zero = jnp.zeros_like(a)
    return jnp.block([[a, zero], [zero, a]])


def _validate_hermitian(a: Array, shape: tuple[int, int], name: str) -> None:
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not _contains_jax_tracer(a):
        # Native integral roundoff scales with the operator, e.g. core levels
        # of heavier atoms. Keep rejecting finite, materially asymmetric data.
        scale = jnp.maximum(1., jnp.max(jnp.abs(a)))
        tolerance = 100 * jnp.finfo(a.real.dtype).eps * scale
        if not bool(jnp.all(jnp.isfinite(a))) or not bool(jnp.allclose(a, a.conj().T, atol=tolerance, rtol=0)):
            raise ValueError(f"{name} must be finite and Hermitian.")


def _validate_mode(config: GKSConfig) -> str:
    recipe = parse_recipe(config.xc_spec)
    recipe.require_complete_support()
    if config.collinear not in {"col", "ncol"}:
        raise NotImplementedError("GKS supports col and ncol; mcol is not implemented.")
    kind = xc_type(config.xc_spec)
    if kind not in {"HF", "LDA", "GGA"}:
        raise NotImplementedError("GKS supports LDA/GGA/global hybrids; meta-GGA is not implemented.")
    if config.collinear == "ncol" and kind not in {"HF", "LDA"}:
        raise NotImplementedError("Noncollinear GKS currently supports LDA (and HF) only.")
    if config.jk_backend not in {"full", "df", "direct"}:
        raise ValueError("Generalized SCF requires full, df, or direct J/K.")
    if config.only_dfj and config.jk_backend != "df":
        raise ValueError("only_dfj requires jk_backend='df'.")
    if recipe.exact_exchange.omega is not None and config.jk_backend == "df" and not config.only_dfj:
        raise NotImplementedError("Generalized RSH supports RI-J/direct-K, not RI exchange.")
    return kind


def generalized_jk(eri: Array | None, density: Array, *, with_k=True,
                   df_factors: Array | None = None, direct_basis: Any | None = None,
                   only_dfj: bool = False, direct_scf_tol: float = 0.,
                   direct_scf_workers: int = 1, eri_lr: Array | None = None,
                   range_separated_coefficients=None) -> tuple[Array, Array]:
    """Spin-diagonal J and four complex K blocks from one spatial operator.

    DF consumes explicit spatial factors. ``only_dfj`` uses them only for J,
    retaining native direct K. RSH K includes its complete SR/LR coefficients.
    No individual spin block is symmetrized or replaced by a collinear density.
    """
    from gradscf.integrals.molecular.jk import build_jk_from_packed, exchange_matrix
    from gradscf.integrals.backends.native.jk import _selected_jk

    density = jnp.asarray(density)
    if density.ndim != 2 or density.shape[0] != density.shape[1] or density.shape[0] % 2:
        raise ValueError("Generalized density must be a square even-dimensional spinor matrix.")
    n = density.shape[0]//2
    total = density[:n, :n] + density[n:, n:]
    blocks = density.reshape(2, n, 2, n).transpose(0, 2, 1, 3)
    source = None if eri is None else jnp.asarray(eri)
    factors = None if df_factors is None else jnp.asarray(df_factors)
    controls = dict(screening_threshold=direct_scf_tol, workers=direct_scf_workers)
    if factors is not None:
        if factors.ndim != 3 or factors.shape[1:] != (n, n) or jnp.iscomplexobj(factors):
            raise ValueError("Generalized RI requires real factors of shape (naux, nao, nao).")
        auxiliary_density = jnp.einsum("Qpq,qp->Q", factors, total, precision=Precision.HIGHEST)
        coulomb = jnp.einsum("Q,Qpq->pq", auxiliary_density, factors, precision=Precision.HIGHEST)
    elif direct_basis is not None:
        coulomb, _ = _selected_jk(direct_basis, total, with_j=True, with_k=False, **controls)
    elif source is not None and source.ndim in (1, 2):
        coulomb, _ = build_jk_from_packed(source, total.T, with_k=False)
    elif source is not None and source.shape == (n, n, n, n):
        coulomb = jnp.einsum("pqrs,sr->pq", source, total, precision=Precision.HIGHEST)
    else:
        raise ValueError("Generalized J/K requires full/packed ERIs, RI factors, or a direct provider.")
    j = _hermitian(_spin_diagonal(coulomb))
    if not with_k:
        return j, jnp.zeros_like(density)
    if only_dfj and (factors is None or direct_basis is None):
        raise ValueError("RI-J/direct-K requires both df_factors and direct_basis.")
    if range_separated_coefficients is not None:
        if factors is not None and not only_dfj:
            raise NotImplementedError("Generalized RSH does not implement RI exchange.")
        if direct_basis is not None:
            from .rsh import _direct_rsh_jk
            _, exchange = _direct_rsh_jk(direct_basis, blocks, range_separated_coefficients,
                                        with_j=False, with_k=True, **controls)
        else:
            if source is None or source.ndim != 4 or eri_lr is None or jnp.shape(eri_lr) != source.shape:
                raise ValueError("Full generalized RSH requires matching four-index eri and eri_lr.")
            _, long_range, short_range = range_separated_coefficients
            exchange = short_range*exchange_matrix(source, blocks)
            exchange += (long_range-short_range)*exchange_matrix(jnp.asarray(eri_lr), blocks)
    elif direct_basis is not None and (factors is None or only_dfj):
        _, exchange = _selected_jk(direct_basis, blocks, with_j=False, with_k=True, **controls)
    elif factors is not None:
        def exchange_block(block):
            projected = jnp.matmul(factors, block, precision=Precision.HIGHEST)
            return jnp.einsum("Qps,Qqs->pq", projected, factors, precision=Precision.HIGHEST)
        exchange = jax.lax.map(jax.checkpoint(exchange_block), blocks.reshape(4, n, n)).reshape(2, 2, n, n)
    elif source.ndim in (1, 2):
        _, exchange = build_jk_from_packed(source, blocks, with_j=False)
    else:
        exchange = exchange_matrix(source, blocks)
    exchange = exchange.transpose(0, 2, 1, 3).reshape(2*n, 2*n)
    return j, _hermitian(exchange)


def _scf_direct_spinor_jk(provider, density, *, cfg, df_factors=None,
                           coefficients=None, incremental=False):
    """One native batch per spatial operator, local to the SCF consumer.

    The generic generalized_jk/response contract is unchanged. Four complex
    spin blocks share the ordinary traversal; Coulomb uses their aa+bb sum.
    """
    from ..integrals.backends.native.jk import NativeDirectBasis, _selected_jk

    n = density.shape[0]//2
    blocks = density.reshape(2, n, 2, n).transpose(0, 2, 1, 3)
    with_k = coefficients is not None or abs(float(hybrid_coeff(cfg.xc_spec))) > 1e-14
    if cfg.only_dfj:
        coulomb, _ = generalized_jk(None, density, df_factors=df_factors, with_k=False)
        if not with_k:
            return coulomb, jnp.zeros_like(density)
    payload = blocks if with_k else density[:n, :n]+density[n:, n:]
    controls = dict(with_j=not cfg.only_dfj, with_k=with_k,
        screening_threshold=cfg.direct_scf_tol, workers=cfg.direct_scf_workers)
    if incremental and type(provider) is NativeDirectBasis and cfg.direct_scf_tol > 0:
        # Dynamic S selects the subset; native rules handle its indicator response.
        controls["screen_density"] = payload
    if coefficients is None:
        j_blocks, k_blocks = _selected_jk(provider, payload, **controls)
    else:
        from .rsh import _direct_rsh_jk
        j_blocks, k_blocks = _direct_rsh_jk(provider, payload, coefficients, **controls)
    if not cfg.only_dfj:
        spatial_j = j_blocks[0, 0]+j_blocks[1, 1] if with_k else j_blocks
        coulomb = _hermitian(_spin_diagonal(spatial_j))
    exchange = (k_blocks.transpose(0, 2, 1, 3).reshape(2*n, 2*n)
                if with_k else jnp.zeros_like(density))
    return coulomb, _hermitian(exchange)


def _generalized_xc(density, ao, deriv, weights, cfg, kind, *, compute_potential=True, grid_ao=None):
    if grid_ao is not None and kind != "HF":
        initial = (jnp.asarray(0., dtype=density.real.dtype), jnp.zeros_like(density))
        def contribution(carry, block_ao, derivative, laplacian, coords, block_weights):
            del laplacian, coords
            energy, potential = _generalized_xc(density, block_ao, derivative,
                block_weights, cfg, kind, compute_potential=compute_potential)
            return carry[0]+energy, carry[1] if potential is None else carry[1]+potential
        energy, potential = grid_ao.reduce(contribution, initial, deriv=0 if kind == "LDA" else 1)
        return energy, potential if compute_potential else None
    n = ao.shape[1]
    if kind == "HF":
        return jnp.asarray(0., dtype=density.real.dtype), jnp.zeros_like(density)
    if cfg.collinear == "col":
        exc, va, vb, ga, gb = _unrestricted_xc_energy_and_potential_on_grid(
            ao=ao, ao_deriv1=deriv, weights=weights,
            density_a=density[:n, :n].real, density_b=density[n:, n:].real,
            xc_spec=cfg.xc_spec, density_floor=cfg.density_floor,
            potential_clip=cfg.potential_clip, xc_kind=kind,
            compute_potential=compute_potential,
        )
        if not compute_potential:
            return exc, None
        a, b = _unrestricted_vxc_matrices(
            ao=ao, ao_deriv1=deriv, ao_laplacian=jnp.zeros_like(ao), weights=weights,
            vxc_rho_a=va, vxc_rho_b=vb, vxc_grad_a=ga, vxc_grad_b=gb, xc_kind=kind,
        )
        zero = jnp.zeros_like(a)
        return exc, jnp.block([[a, zero], [zero, b]])

    # Local spin-density eigenvalues (rho +/- |m|)/2 define noncollinear LDA.
    local = jnp.einsum("gp,apbq,gq->gab", ao, density.reshape(2, n, 2, n), ao,
                       precision=Precision.HIGHEST)
    total = (local[:, 0, 0] + local[:, 1, 1]).real
    magnetization = jnp.stack([2 * local[:, 0, 1].real, -2 * local[:, 0, 1].imag,
                              (local[:, 0, 0] - local[:, 1, 1]).real], axis=1)
    norm2 = jnp.sum(magnetization**2, axis=1)
    # Safe zero-magnetization limit, matching PySCF's 1e-20 direction cutoff.
    cutoff = max(1e-20, float(jnp.finfo(density.real.dtype).tiny)**.5)
    safe_norm = jnp.sqrt(jnp.maximum(norm2, cutoff**2))
    magnitude = jnp.where(norm2 >= cutoff**2, safe_norm, 0.)
    rho = jnp.stack([(total + magnitude) * .5, (total - magnitude) * .5], axis=1)
    if not compute_potential:
        return jnp.dot(weights, grid_energy(cfg.xc_spec, "LDA")(rho)), None
    values, potentials = grid_value_and_grad(cfg.xc_spec, "LDA")(rho)
    potentials = jnp.where((total > cfg.density_floor)[:, None], potentials, 0.)
    if cfg.potential_clip is not None:
        potentials = jnp.clip(potentials, -cfg.potential_clip, cfg.potential_clip)
    scalar = (potentials[:, 0] + potentials[:, 1]) * .5
    spin = (potentials[:, 0] - potentials[:, 1]) * .5
    direction = jnp.where((norm2 >= cutoff**2)[:, None], magnetization / safe_norm[:, None], 0.)
    vx, vy, vz = (spin[:, None] * direction).T

    def integrate(value):
        return jnp.einsum("g,gp,gq->pq", weights * value, ao, ao, precision=Precision.HIGHEST)

    ab = integrate(vx - 1j * vy)
    vxc = jnp.block([[integrate(scalar + vz), ab], [ab.conj().T, integrate(scalar - vz)]])
    return jnp.dot(weights, values), vxc


def generalized_energy_and_fock(
    *, density: Array, hcore: Array, eri: Array | None, nuclear_repulsion: float | Array,
    ao: Array, ao_deriv1: Array, grid_weights: Array, config: GKSConfig,
    compute_potential: bool = True, df_factors: Array | None = None,
    direct_basis: Any | None = None, grid_ao: Any | None = None,
    eri_lr: Array | None = None, dispersion_energy: Array | float | None = None,
    jk_matrices: tuple[Array, Array] | None = None,
) -> tuple[Array, Array, Array | None]:
    """Return energy/XC/Fock; energy-only requests return None for the Fock."""

    kind = _validate_mode(config)
    density, h = jnp.asarray(density), jnp.asarray(hcore)
    if h.shape[0] == density.shape[0] // 2:
        h = _spin_diagonal(h)
    from .rsh import validate_rsh_inputs
    preset = validate_rsh_inputs(config.xc_spec, eri=eri, eri_lr=eri_lr,
        dispersion_energy=dispersion_energy,
        jk_backend="direct" if config.only_dfj else config.jk_backend,
        direct_basis=direct_basis)
    coefficients = None if preset is None else preset.to_range_separated_hybrid_coefficients()
    alpha = hybrid_coeff(config.xc_spec) if preset is None else 1.
    if jk_matrices is None:
        j, k = generalized_jk(eri, density, with_k=alpha != 0., df_factors=df_factors,
            direct_basis=direct_basis, only_dfj=config.only_dfj,
            direct_scf_tol=config.direct_scf_tol, direct_scf_workers=config.direct_scf_workers,
            eri_lr=eri_lr, range_separated_coefficients=coefficients)
    else:
        j, k = jk_matrices
    exc, vxc = _generalized_xc(density, jnp.asarray(ao), jnp.asarray(ao_deriv1),
                             jnp.asarray(grid_weights), config, kind,
                             compute_potential=compute_potential, grid_ao=grid_ao)
    potential = j - alpha * k
    energy = (jnp.einsum("pq,qp->", density, h + .5 * potential,
                        precision=Precision.HIGHEST).real + exc + nuclear_repulsion
              + (0. if dispersion_energy is None else dispersion_energy))
    if not compute_potential:
        return energy, exc, None
    return energy, exc, _hermitian(h + potential + vxc)


def run_gks_from_integrals(
    *, overlap: Array, hcore: Array, eri: Array | None, nelectron: int,
    nuclear_repulsion: float | Array, ao: Array, ao_deriv1: Array, grid_weights: Array,
    init_density: Array | None = None, config: GKSConfig | None = None,
    df_factors: Array | None = None, direct_basis: Any | None = None,
    grid_ao: Any | None = None, eri_lr: Array | None = None,
    dispersion_energy: Array | float | None = None,
) -> GKSResult:
    """Generalized SCF: spatial overlap/ERIs, spatial or 2*nao spinor hcore.

    Only total electron number is constrained. Spin mixing is preserved in
    complex densities, exchange blocks, eigensolutions, and DIIS residuals.
    """

    cfg = GKSConfig() if config is None else config
    _validate_mode(cfg)
    s, h = map(jnp.asarray, (overlap, hcore))
    eri = None if eri is None else jnp.asarray(eri)
    df_factors = None if df_factors is None else jnp.asarray(df_factors)
    n = s.shape[0]
    from gradscf.integrals.molecular.eri import packed_eri_shape
    if s.shape != (n, n):
        raise ValueError("Generalized SCF requires square spatial overlap.")
    if cfg.jk_backend == "full" and (eri is None or eri.shape not in {
            (n,n,n,n), packed_eri_shape(n,1), packed_eri_shape(n,2)}):
        raise ValueError("Generalized full SCF requires full/s4/s8 ERIs.")
    if cfg.jk_backend == "df" and (df_factors is None or df_factors.size == 0):
        raise ValueError("Generalized RI requires nonempty df_factors.")
    if (cfg.jk_backend == "direct" or cfg.only_dfj) and direct_basis is None:
        raise ValueError("Generalized direct exchange requires direct_basis.")
    if any(jnp.iscomplexobj(a) for a in (s, eri, df_factors, eri_lr, ao, ao_deriv1) if a is not None):
        raise ValueError("Generalized SCF currently requires real spatial AOs and integrals.")
    if not isinstance(nelectron, Integral) or not 0 < nelectron <= 2*n:
        raise ValueError("Invalid total electron count for generalized SCF.")
    if cfg.max_cycle < 1 or cfg.conv_tol < 0 or min(cfg.conv_tol_density, cfg.conv_tol_grad) <= 0:
        raise ValueError("Generalized SCF requires positive iterations, density/gradient tolerances, and nonnegative conv_tol.")
    if cfg.convergence_metric not in ("energy_and_residual", "energy"):
        raise ValueError("Generalized SCF requires energy_and_residual or energy convergence.")
    ao, deriv, weights = map(jnp.asarray, (ao, ao_deriv1, grid_weights))
    real_dtype = jnp.result_type(s, h.real, ao, deriv, weights, nuclear_repulsion,
        *(a for a in (eri, df_factors, eri_lr) if a is not None), jnp.float32)
    if init_density is not None:
        real_dtype = jnp.result_type(real_dtype, jnp.asarray(init_density).real)
    complex_dtype = jnp.result_type(real_dtype, jnp.complex64)
    s, h = s.astype(real_dtype), h.astype(complex_dtype)
    eri = None if eri is None else eri.astype(real_dtype)
    df_factors = None if df_factors is None else df_factors.astype(real_dtype)
    eri_lr = None if eri_lr is None else jnp.asarray(eri_lr, dtype=real_dtype)
    ao, deriv, weights = (a.astype(real_dtype) for a in (ao, deriv, weights))
    _validate_hermitian(s, (n, n), "overlap")
    if h.shape == (n, n):
        h = _spin_diagonal(h)
    _validate_hermitian(h, (2*n, 2*n), "hcore")
    expected_grid = weights.size if grid_ao is None else 0
    if ao.shape != (expected_grid, n) or deriv.shape != (4, expected_grid, n):
        raise ValueError("AO values/first derivatives must match the grid and spatial AO dimensions.")
    eig = jnp.linalg.eigvalsh(s)
    if not _contains_jax_tracer(eig) and bool(jnp.any(eig <= 0)):
        raise ValueError("overlap must be positive definite.")
    x = _spin_diagonal(orbital_orthogonalizer(s, cfg.orthogonalization_eps))
    s2 = _spin_diagonal(s)
    occ = jnp.zeros((2*n,), dtype=real_dtype).at[:nelectron].set(1.)
    if grid_ao is not None and _validate_mode(cfg) in {"LDA", "GGA"}:
        from gradscf._memory import tree_storage_bytes
        from .inputs.grid import prepare_scf_grid, scf_operator_workspace_bytes

        resident = tree_storage_bytes(s, h, eri, eri_lr, df_factors, ao, deriv, init_density)
        resident += 32 * h.size * h.dtype.itemsize
        exchange = parse_recipe(cfg.xc_spec).exact_exchange
        operator = scf_operator_workspace_bytes(df_factors, direct_basis,
            with_k=exchange.omega is not None or float(hybrid_coeff(cfg.xc_spec)) != 0.,
            only_dfj=cfg.only_dfj, density_dtype=h.dtype,
            workers=cfg.direct_scf_workers, cutoff=cfg.direct_scf_tol)
        if operator is not None:
            grid_ao = prepare_scf_grid(grid_ao, max_memory=cfg.max_memory,
                resident_bytes=resident + operator, deriv=0 if xc_type(cfg.xc_spec) == "LDA" else 1)

    def diagonalize(fock):
        values, c = jnp.linalg.eigh(_hermitian(x.conj().T @ fock @ x))
        return values, x @ c

    def density_from(coeff):
        return _hermitian((coeff * occ) @ coeff.conj().T)

    from .rsh import rsh_preset
    preset = rsh_preset(cfg.xc_spec)
    coefficients = None if preset is None else preset.to_range_separated_hybrid_coefficients()
    incremental = direct_basis is not None and (cfg.jk_backend == "direct" or cfg.only_dfj)

    def evaluate(dm, coeff, energies, previous):
        matrices = None
        if incremental:
            delta = dm if previous is None else dm-previous.density
            matrices = _scf_direct_spinor_jk(direct_basis, delta, cfg=cfg,
                df_factors=df_factors, coefficients=coefficients, incremental=previous is not None)
            if previous is not None:
                matrices = tuple(current+last for current, last in zip(matrices, previous.aux[1]))
        energy, exc, fock = generalized_energy_and_fock(
            density=dm, hcore=h, eri=eri, nuclear_repulsion=nuclear_repulsion,
            ao=ao, ao_deriv1=deriv, grid_weights=weights, config=cfg,
            df_factors=df_factors, direct_basis=direct_basis, grid_ao=grid_ao,
            eri_lr=eri_lr, dispersion_energy=dispersion_energy, jk_matrices=matrices,
        )
        return energy, fock, (exc, () if matrices is None else matrices, jnp.asarray(previous is None or not incremental))

    eps0, c0 = diagonalize(h)
    dm0 = density_from(c0) if init_density is None else jnp.asarray(init_density, dtype=complex_dtype)
    _validate_hermitian(dm0, (2*n, 2*n), "init_density")
    e0, f0, aux0 = evaluate(dm0, c0, eps0, None)
    result = solve_scf(SCFState(dm0, c0, eps0, e0, f0, aux0),
        evaluate=evaluate, diagonalize=diagonalize, density_from_coeff=density_from,
        diis_error=lambda f, state: x.conj().T @ (
            f @ state.density @ s2 - s2 @ state.density @ f) @ x,
        orbital_gradient=lambda state: orbital_gradient_norm(state.fock, state.mo_coeff, occ),
        shift_fock=lambda f, state: _hermitian(
            f+cfg.level_shift*(s2-s2 @ state.density @ s2)),
        config=SCFConfig(max_cycle=cfg.max_cycle, conv_tol=cfg.conv_tol,
            conv_tol_density=cfg.conv_tol_density, conv_tol_grad=cfg.conv_tol_grad,
            damping=cfg.damping, fock_damping_cycles=2,
            energy_only=cfg.convergence_metric == "energy", refresh_on_convergence=incremental))
    final = result.state
    traced = _contains_jax_tracer(result)
    return GKSResult(
        converged=result.converged if traced else bool(result.converged),
        cycles=result.cycles if traced else int(result.cycles),
        total_energy=_host_float_unless_traced(final.energy),
        electronic_energy=_host_float_unless_traced(final.energy - nuclear_repulsion),
        nuclear_repulsion=_host_float_unless_traced(nuclear_repulsion),
        xc_energy=_host_float_unless_traced(final.aux[0]),
        exact_exchange_fraction=float(hybrid_coeff(cfg.xc_spec)),
        mo_energy=jnp.diag(final.mo_coeff.conj().T @ final.fock @ final.mo_coeff).real,
        mo_coeff=final.mo_coeff, mo_occ=occ, density_matrix=final.density,
        fock_matrix=final.fock, overlap_matrix=s2, hcore_matrix=h,
        orbital_gradient_norm=_host_float_unless_traced(result.metrics.gradient),
        range_separated_coefficients=parse_recipe(cfg.xc_spec).to_range_separated_hybrid_coefficients(),
        dispersion_energy=dispersion_energy,
    )


@dataclass
class GKS(_BaseKS):
    """Generalized ground-state facade; mol.spin controls only the initial guess."""

    collinear: str = "col"
    conv_tol_grad: float = 1e-7
    jk_backend: Literal["full", "df", "direct"] = "full"
    auxbasis: str | None = None
    only_dfj: bool = False
    df_tol: float = 1e-10
    df_max_rank: int | None = None
    direct_scf_tol: float = 0.
    direct_scf_workers: int = 1
    scf_result: GKSResult | None = None

    def _config(self) -> GKSConfig:
        return GKSConfig(xc_spec=self.xc, collinear=self.collinear, max_cycle=self.max_cycle,
                         conv_tol=self.conv_tol, conv_tol_density=self.conv_tol_density,
                         conv_tol_grad=self.conv_tol_grad, damping=self.damp, level_shift=self.level_shift,
                         max_memory=self.max_memory, jk_backend=self.jk_backend,
                         auxbasis=self.auxbasis, only_dfj=self.only_dfj, df_tol=self.df_tol,
                         df_max_rank=self.df_max_rank, direct_scf_tol=self.direct_scf_tol,
                         direct_scf_workers=self.direct_scf_workers,
                         grid_block_size=self.grid_block_size, grid_max_memory=self.grid_max_memory)

    def kernel(self, dm0: Array | None = None) -> Any:
        cfg = self._config()
        _validate_mode(cfg)
        self._configure_jax_cache()
        supplied = dm0 if dm0 is not None else (None if isinstance(self.init_guess, str) else self.init_guess)
        inputs = build_uks_integral_inputs(
            atom=self._spec(), basis=self.mol.basis, config=cfg,
            xc_spec=self.xc, unit=self.mol.unit, charge=self.mol.charge, spin=self.mol.spin,
            cart=self.mol.cart, grids_level=self.grids_level, max_l=self.max_l,
            integral_backend=self.integral_backend, grid_ao_backend=self.grid_ao_backend,
            libcint_geometry_grad_policy=self.geometry_grad_policy,
            init_guess=self.init_guess if supplied is None else "hcore", chkfile=self.chkfile,
            init_guess_sap_basis=self.sap_basis,
            init_guess_chkfile_project=self.init_guess_chkfile_project, verbose=self.mol.verbose,
            scf_history_matrices=48,
        )
        if supplied is None and inputs.init_density_alpha is not None:
            a, b = inputs.init_density_alpha, inputs.init_density_beta
            zero = jnp.zeros_like(a)
            supplied = jnp.block([[a, zero], [zero, b]])
        kwargs = dict(overlap=inputs.overlap, hcore=inputs.hcore, eri=inputs.eri,
                      nuclear_repulsion=inputs.nuclear_repulsion, ao=inputs.ao,
                      ao_deriv1=inputs.ao_deriv1, grid_weights=inputs.grid_weights, init_density=supplied,
                      df_factors=inputs.df_factors, direct_basis=inputs.direct_basis,
                      grid_ao=inputs.grid_ao, eri_lr=inputs.eri_lr, dispersion_energy=inputs.dispersion_energy)
        if self.execution_device != "auto":
            from ..tools.device import resolve_execution_device
            device = resolve_execution_device(self.execution_device)
            kwargs = jax.tree_util.tree_map(lambda a: jax.device_put(a, device), kwargs)
        result = run_gks_from_integrals(**kwargs, nelectron=self.mol.nelectron, config=cfg)
        self._scf_inputs = inputs
        self.scf_result = result
        self._sync_from_scf_result(result)
        self._cached_scf_key = self._scf_signature()
        return self.e_tot

    def run(self, dm0: Array | None = None) -> "GKS":
        self.kernel(dm0=dm0)
        return self

    def make_rdm1(self) -> Array:
        if self.mo_coeff is None or self.mo_occ is None:
            raise RuntimeError("Run generalized SCF before make_rdm1().")
        return (self.mo_coeff * self.mo_occ) @ self.mo_coeff.conj().T

    def _ensure_reference(self) -> Any:
        raise NotImplementedError("Generalized spinor response is not implemented.")

    def TDA(self, **kwargs: Any) -> Any:
        return self._ensure_reference()

    def TDDFT(self, **kwargs: Any) -> Any:
        return self._ensure_reference()

    def density_fit(self, auxbasis: str | None = None, *, only_dfj: bool = False) -> "GKS":
        self.jk_backend = "df"
        self.auxbasis = auxbasis or "def2-universal-jkfit"
        self.only_dfj = bool(only_dfj)
        return self

    def direct_scf(self) -> "GKS":
        self.jk_backend = "direct"
        self.only_dfj = False
        return self
