"""Basis/grid construction and device-aware caching of AO values."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from gradscf._memory import (
    DEFAULT_MAX_MEMORY_MB, array_storage_bytes, validate_memory_mb, workspace_bytes,
    _requested_workspace_bytes, validate_reserved_bytes,
)
from gradscf.integrals.basis import CartesianBasis
from gradscf.data.molecule import MoleculeSpec
from gradscf.integrals.grids.blocks import (
    AOGrid, ao_block_size, ao_point_bytes, _materialize_scf_grid,
)
from gradscf.scf.core import _cache_signature

_GRID_AO_INPUT_CACHE_MAXSIZE = 32
_GRID_AO_INPUT_CACHE_MAXBYTES = 256_000_000
_GRID_AO_INPUT_CACHE: dict[tuple[Any, ...], Any] = {}

@dataclass(frozen=True)
class _GridAOInputBundle:
    basis: CartesianBasis
    coords: Array
    grid_weights: Array
    ao: Array
    ao_deriv1: Array
    ao_laplacian: Array | None
    grid_ao: AOGrid | None = None
    execution_key: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class _BasisGridContext:
    basis: CartesianBasis
    coords: Array
    grid_weights: Array
    geometry_is_traced: bool
    grid_ao_bundle: _GridAOInputBundle | None
    grid_ao: AOGrid | None = None


def _grid_device(array):
    """Inspect only an existing array or the caller's selected device context.

    A tracer need not expose a device. Leaving it unknown avoids initializing
    unrelated accelerators merely to estimate working-array storage.
    """
    device = getattr(array, "device", None)
    if callable(device):
        device = device()
    return device if device is not None else getattr(jax.config, "jax_default_device", None)


def _active_device_cache_key() -> tuple[str, int]:
    device = _grid_device(jnp.asarray(0.0))
    return (str(getattr(device, "platform", "unknown")), int(getattr(device, "id", -1)))


def _grid_ao_input_cache_key(
    *,
    spec: MoleculeSpec,
    basis: Any,
    max_l: int,
    grids_level: int,
    needs_ao_laplacian: bool,
    grid_block_size: int = 0,
    grid_max_memory: float | None = None,
    max_memory: float = DEFAULT_MAX_MEMORY_MB,
    scf_max_cycle: int = 80,
    scf_history_matrices: int = 2,
    fixed_workspace_bytes: int = 0,
    force_streamed: bool = False,
) -> tuple[Any, ...]:
    coords_bohr = np.asarray(jax.device_get(spec.coords_bohr), dtype=np.float64)
    physical = (
        tuple(spec.symbols), _cache_signature(getattr(spec, "charges", None)),
        _cache_signature(basis), int(max_l), int(grids_level),
        bool(needs_ao_laplacian), bool(jax.config.x64_enabled),
        coords_bohr.shape, coords_bohr.dtype.str, coords_bohr.tobytes(), _active_device_cache_key(),
    )
    execution = (
        int(grid_block_size), grid_max_memory, float(max_memory), int(scf_max_cycle),
        int(scf_history_matrices), int(fixed_workspace_bytes), bool(force_streamed),
    )
    return physical, execution


def _grid_resident_bytes(basis, coords, weights, *arrays, scf_max_cycle,
                         scf_history_matrices=2, fixed_workspace_bytes=0):
    # The eager scan result can retain both density and orbital histories.
    # Reserve those as well as SCF/DIIS matrices before allocating AO work.
    itemsize = max(8, np.dtype(coords.dtype).itemsize)
    matrices = (24 + int(scf_history_matrices) * int(scf_max_cycle)) * basis.nao**2 * itemsize
    if (isinstance(fixed_workspace_bytes, bool)
            or not isinstance(fixed_workspace_bytes, Integral) or fixed_workspace_bytes < 0):
        raise ValueError("fixed_workspace_bytes must be a nonnegative integer.")
    cached_arrays = (array for bundle in _GRID_AO_INPUT_CACHE.values() for array in (
        bundle.coords, bundle.grid_weights, bundle.ao,
        bundle.ao_deriv1, bundle.ao_laplacian,
        bundle.basis.ao_centers, bundle.basis.ao_exponents_padded,
        bundle.basis.ao_coefficients_padded,
    ))
    return matrices + fixed_workspace_bytes + array_storage_bytes(
        coords, weights, basis.ao_centers,
        basis.ao_exponents_padded, basis.ao_coefficients_padded,
        *arrays, *cached_arrays,
    )


def _streamed_grid(basis, coords, weights, *, needs_ao_laplacian,
                   grid_block_size, grid_max_memory,
                   max_memory=DEFAULT_MAX_MEMORY_MB, scf_max_cycle=80,
                   scf_history_matrices=2, fixed_workspace_bytes=0,
                   _keep_streamed=False, _resident_arrays=()):
    if grid_block_size < -1:
        raise ValueError("grid_block_size must be -1 (dense), 0 (auto), or positive.")
    validate_memory_mb(max_memory)
    if grid_max_memory is not None:
        validate_memory_mb(grid_max_memory, name="grid_max_memory")
    if not coords.shape[0]:
        return None
    deriv = 2 if needs_ao_laplacian else 1
    device = _grid_device(coords)
    available = workspace_bytes(max_memory, device=device, resident_bytes=
        _grid_resident_bytes(basis, coords, weights, *_resident_arrays, scf_max_cycle=scf_max_cycle,
            scf_history_matrices=scf_history_matrices,
            fixed_workspace_bytes=fixed_workspace_bytes))
    full_workspace = coords.shape[0] * ao_point_bytes(basis, deriv=deriv)
    if grid_block_size == -1:
        if full_workspace > available:
            raise ValueError("Dense AO quadrature exceeds max_memory; use grid_block_size=0.")
        return None
    if grid_max_memory is not None:
        available = min(available, int(grid_max_memory * 1e6))
    if grid_block_size == 0 and full_workspace <= available and not _keep_streamed:
        return None
    block = ao_block_size(basis, deriv=deriv, max_memory=available / 1e6,
                          requested=max(0, grid_block_size), device=device)
    if grid_block_size == 0:
        block = min(block, int(coords.shape[0]))
    return AOGrid.from_basis(basis, coords, weights, block_size=block,
        adaptive=grid_block_size == 0 and grid_max_memory is None)


def _scf_operator_reservation(df_factors=None, direct_basis=None, with_k=False,
                              only_dfj=False, density_dtype=None, workers=1, cutoff=0):
    """Return (device, host) working bytes; None disables optional AO reuse.

    RI products follow their JAX consumer's device. Libcint work remains on
    the host, including the GPU callback path and three-call allowance.
    Opaque/projected providers retain their original streamed execution.
    """
    reserve = 0
    if df_factors is not None:
        try:
            shape = tuple(int(n) for n in df_factors.shape)
            item = np.dtype(np.result_type(df_factors.dtype,
                df_factors.dtype if density_dtype is None else density_dtype)).itemsize
        except (AttributeError, TypeError, ValueError):
            return None
        if len(shape) != 3 or shape[1] != shape[2] or min(shape) < 0:
            return None
        reserve = 2 * shape[0] * item  # Auxiliary contraction/solve vectors.
        if with_k and not only_dfj:
            return int(reserve + shape[0] * shape[1] * shape[2] * item), 0
        if not with_k or not only_dfj:
            return int(reserve), 0
    elif only_dfj:
        return None
    if direct_basis is None:
        return None if only_dfj and with_k else (int(reserve), 0)
    from gradscf.integrals.backends.native.jk import NativeDirectBasis, native_jk_workspace_bytes
    if type(direct_basis) is not NativeDirectBasis:
        return None
    try:
        estimate = native_jk_workspace_bytes(direct_basis, workers, cutoff)
    except (AttributeError, TypeError, ValueError):
        return None
    return None if estimate is None else (int(reserve), int(3 * estimate))



def scf_operator_workspace_bytes(df_factors=None, direct_basis=None, with_k=False,
                                 only_dfj=False, density_dtype=None, workers=1, cutoff=0):
    """Total method reservation, retaining the existing scalar estimate API."""
    reservation = _scf_operator_reservation(df_factors, direct_basis, with_k,
        only_dfj, density_dtype, workers, cutoff)
    return None if reservation is None else sum(reservation)


def prepare_scf_grid(raw, *, max_memory, resident_bytes=0, deriv=1,
                     host_workspace_bytes=0, expected_passes=None, screen_tol=0.):
    """Return call-local AO reuse when outputs, one copy and scratch fit.

    The raw provider stays unchanged. Caller reservations cover other SCF
    arrays; the provider's own leaves are additionally charged conservatively.
    Host-only native work consumes the method budget before the device cap;
    on CPU all storage shares one pool. An explicit zero/single-consumption
    hint streams; unknown/default reuse keeps the existing admission policy.
    No SCF iteration limit is interpreted as a count of Fock evaluations.
    Unknown traced devices and explicit streaming retain the original path.
    """
    if not np.isfinite(screen_tol) or screen_tol < 0:
        raise ValueError("grid_screen_tol must be finite and nonnegative.")
    if not isinstance(raw, AOGrid) or not raw.ngrids or not raw.nao:
        return raw
    if not raw.adaptive and screen_tol == 0:
        return raw
    if deriv not in (0, 1, 2):
        raise ValueError("SCF AO preparation requires deriv in (0, 1, 2).")
    host = validate_reserved_bytes(host_workspace_bytes, name="host_workspace_bytes")
    if expected_passes is not None:
        if isinstance(expected_passes, bool) or not isinstance(expected_passes, Integral) or expected_passes < 0:
            raise ValueError("expected_passes must be a nonnegative integer or None.")
        if expected_passes <= 1:
            return raw
    device = _grid_device(raw.coords)
    if device is None and screen_tol == 0:
        return raw
    owned = array_storage_bytes(*jax.tree_util.tree_leaves(raw))
    try:
        if getattr(device, "platform", None) == "cpu" or device == "cpu":
            available = workspace_bytes(max_memory,
                resident_bytes=resident_bytes + owned + host, device=device)
        else:
            # Keep the shared 20% reserve for both method and device limits.
            # Host scratch is not a VRAM allocation, but cannot disappear
            # from the total method allowance when the device cap is lower.
            available = min(workspace_bytes(max_memory, device=device),
                _requested_workspace_bytes(max_memory) - host) - resident_bytes - owned
    except ValueError:
        if screen_tol:
            raise ValueError("Local AO screening workspace exceeds max_memory.") from None
        return raw  # An optional cache must not displace an already streamed calculation.
    itemsize = max(8, np.dtype(jnp.result_type(raw.coords, raw.exponents, raw.coefficients)).itemsize)
    if screen_tol:
        from gradscf.integrals.grids.blocks import _LocalAOGrid
        # Include sorting/support metadata and the dense overflow branch.
        metadata = 192 * (raw.ngrids + raw.nao * raw.exponents.shape[1])
        required = raw.block_size * ao_point_bytes(raw, deriv=deriv) + metadata + 2*raw.nao**2*itemsize
        if required > available:
            raise ValueError("Local AO screening requires its dense fallback workspace within max_memory.")
        capacity = min(raw.nao, 1 << max(5, (raw.nao-1).bit_length()-1))
        return _LocalAOGrid.from_grid(raw, cutoff=float(screen_tol), capacity=capacity)
    point = raw.nao * (1, 4, 5)[deriv] * itemsize
    scratch = ao_point_bytes(raw, deriv=deriv) * itemsize // max(8, np.dtype(raw.exponents.dtype).itemsize)
    # scan's padded output and the trimmed result may coexist. Reserving a
    # full extra tile for both makes the bound independent of tail divisibility.
    block = min(raw.block_size, raw.ngrids,
                (available - 2 * raw.ngrids * point) // (scratch + 2 * point))
    if block < 1:
        return raw
    return _materialize_scf_grid(raw, block_size=int(block), deriv=deriv)


def _empty_ao_payload(nao, dtype):
    return jnp.zeros((0, nao), dtype=dtype), jnp.zeros((4, 0, nao), dtype=dtype), None


def _dense_ao_payload(basis, coords, *, needs_ao_laplacian, evaluate):
    ao, derivatives = evaluate(basis, coords, deriv=2 if needs_ao_laplacian else 1)
    if needs_ao_laplacian:
        return ao, derivatives[:4], derivatives[4]
    return ao, derivatives, None


def _build_grid_ao_input_bundle(
    *,
    basis_from_molecule_spec: Callable[..., Any],
    build_molecular_grid_from_spec: Callable[..., Any],
    evaluate_cartesian_ao_with_derivatives: Callable[..., Any],
    spec: MoleculeSpec,
    basis: Any,
    max_l: int,
    grids_level: int,
    needs_ao_laplacian: bool,
    grid_block_size: int = 0,
    grid_max_memory: float | None = None,
    max_memory: float = DEFAULT_MAX_MEMORY_MB,
    scf_max_cycle: int = 80,
    scf_history_matrices: int = 2,
    fixed_workspace_bytes: int = 0,
    force_streamed: bool = False,
) -> _GridAOInputBundle:
    basis_cart = basis_from_molecule_spec(spec, basis=basis, max_l=max_l)
    coords, weights = build_molecular_grid_from_spec(spec, level=grids_level)
    provider = _streamed_grid(basis_cart, coords, weights,
        needs_ao_laplacian=needs_ao_laplacian, grid_block_size=grid_block_size,
        grid_max_memory=grid_max_memory, max_memory=max_memory,
        scf_max_cycle=scf_max_cycle, scf_history_matrices=scf_history_matrices,
        fixed_workspace_bytes=fixed_workspace_bytes, _keep_streamed=force_streamed)
    if provider is not None:
        return _GridAOInputBundle(basis_cart, coords, weights,
            *_empty_ao_payload(basis_cart.nao, coords.dtype), provider)
    return _GridAOInputBundle(basis_cart, coords, weights,
        *_dense_ao_payload(basis_cart, coords, needs_ao_laplacian=needs_ao_laplacian,
                           evaluate=evaluate_cartesian_ao_with_derivatives))


def _cached_grid_ao_input_bundle(
    *,
    basis_from_molecule_spec: Callable[..., Any],
    build_molecular_grid_from_spec: Callable[..., Any],
    evaluate_cartesian_ao_with_derivatives: Callable[..., Any],
    spec: MoleculeSpec,
    basis: Any,
    max_l: int,
    grids_level: int,
    needs_ao_laplacian: bool,
    grid_block_size: int = 0,
    grid_max_memory: float | None = None,
    max_memory: float = DEFAULT_MAX_MEMORY_MB,
    scf_max_cycle: int = 80,
    scf_history_matrices: int = 2,
    fixed_workspace_bytes: int = 0,
    force_streamed: bool = False,
) -> _GridAOInputBundle:
    grid_options = dict(needs_ao_laplacian=needs_ao_laplacian, grid_block_size=grid_block_size,
                        grid_max_memory=grid_max_memory, max_memory=max_memory,
                        scf_max_cycle=scf_max_cycle, scf_history_matrices=scf_history_matrices,
                        fixed_workspace_bytes=fixed_workspace_bytes)
    key, execution_key = _grid_ao_input_cache_key(spec=spec, basis=basis, max_l=max_l,
        grids_level=grids_level, force_streamed=force_streamed, **grid_options)
    cached = _GRID_AO_INPUT_CACHE.get(key)
    if cached is None:
        bundle = _build_grid_ao_input_bundle(
            basis_from_molecule_spec=basis_from_molecule_spec,
            build_molecular_grid_from_spec=build_molecular_grid_from_spec,
            evaluate_cartesian_ao_with_derivatives=evaluate_cartesian_ao_with_derivatives,
            spec=spec, basis=basis, max_l=max_l, grids_level=grids_level,
            force_streamed=force_streamed, **grid_options,
        )
    else:
        # Budget changes replan execution while reusing the spatial quadrature.
        old_provider = cached.grid_ao
        same_execution = cached.execution_key == execution_key
        keep_streamed = force_streamed or (old_provider is not None and same_execution)
        provider = _streamed_grid(cached.basis, cached.coords, cached.grid_weights,
                                  _keep_streamed=keep_streamed, **grid_options)
        if provider is None:
            if old_provider is None:
                return cached
            ao, derivative, laplacian = _dense_ao_payload(cached.basis, cached.coords,
                needs_ao_laplacian=needs_ao_laplacian, evaluate=evaluate_cartesian_ao_with_derivatives)
            bundle = replace(cached, ao=ao, ao_deriv1=derivative, ao_laplacian=laplacian, grid_ao=None)
        elif old_provider is None:
            ao, derivative, laplacian = _empty_ao_payload(cached.basis.nao, cached.coords.dtype)
            bundle = replace(cached, ao=ao, ao_deriv1=derivative, ao_laplacian=laplacian, grid_ao=provider)
        else:
            if (provider.block_size == old_provider.block_size
                    and provider.adaptive == old_provider.adaptive):
                if same_execution:
                    return cached
                provider = old_provider
            else:
                provider = replace(old_provider, block_size=provider.block_size, adaptive=provider.adaptive)
            bundle = replace(cached, grid_ao=provider)
    bundle = replace(bundle, execution_key=execution_key)
    _cache_bounded(_GRID_AO_INPUT_CACHE, key, bundle, _GRID_AO_INPUT_CACHE_MAXSIZE)
    return bundle


def _prepare_basis_grid_context(
    *,
    basis_from_molecule_spec: Callable[..., Any],
    build_molecular_grid_from_spec: Callable[..., Any],
    evaluate_cartesian_ao_with_derivatives: Callable[..., Any],
    spec: MoleculeSpec,
    basis: Any,
    max_l: int,
    grids_level: int,
    needs_ao_laplacian: bool,
    needs_grid: bool = True,
    grid_block_size: int = 0,
    grid_max_memory: float | None = None,
    max_memory: float = DEFAULT_MAX_MEMORY_MB,
    scf_max_cycle: int = 80,
    scf_history_matrices: int = 2,
    fixed_workspace_bytes: int = 0,
    force_streamed: bool = False,
) -> _BasisGridContext:
    geometry_is_traced = any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(spec.coords_bohr))
    if not needs_grid:
        # Pure HF ghost energies need the basis but no XC quadrature. Build
        # it from this spec instead of reusing charge-insensitive grid data.
        basis_cart = basis_from_molecule_spec(
            spec, basis=basis, max_l=max_l,
        )
        dtype = jnp.asarray(spec.coords_bohr).dtype
        bundle = _GridAOInputBundle(
            basis_cart, jnp.zeros((0, 3), dtype=dtype), jnp.zeros((0,), dtype=dtype),
            *_empty_ao_payload(basis_cart.nao, dtype),
        )
        return _BasisGridContext(
            basis=basis_cart, coords=bundle.coords,
            grid_weights=bundle.grid_weights,
            geometry_is_traced=geometry_is_traced,
            grid_ao_bundle=bundle,
        )
    if geometry_is_traced:
        coords, weights = build_molecular_grid_from_spec(spec, level=grids_level)
        basis_cart = basis_from_molecule_spec(
            spec,
            basis=basis,
            max_l=max_l,
        )
        return _BasisGridContext(
            basis=basis_cart,
            coords=coords,
            grid_weights=weights,
            geometry_is_traced=True,
            grid_ao_bundle=None,
            grid_ao=_streamed_grid(basis_cart, coords, weights,
                needs_ao_laplacian=needs_ao_laplacian, grid_block_size=grid_block_size,
                grid_max_memory=grid_max_memory, max_memory=max_memory,
                scf_max_cycle=scf_max_cycle, scf_history_matrices=scf_history_matrices,
                fixed_workspace_bytes=fixed_workspace_bytes, _keep_streamed=force_streamed),
        )
    # Fixed coordinates may still be closed over by a JIT of a field or model
    # parameter. Build/cache their constant grid eagerly: device inspection
    # cannot consume a tracer, and cached AO arrays must not escape a trace.
    with jax.ensure_compile_time_eval():
        grid_ao_bundle = _cached_grid_ao_input_bundle(
            basis_from_molecule_spec=basis_from_molecule_spec,
            build_molecular_grid_from_spec=build_molecular_grid_from_spec,
            evaluate_cartesian_ao_with_derivatives=evaluate_cartesian_ao_with_derivatives,
            spec=spec,
            basis=basis,
            max_l=max_l,
            grids_level=grids_level,
            needs_ao_laplacian=needs_ao_laplacian,
            grid_block_size=grid_block_size, grid_max_memory=grid_max_memory,
            max_memory=max_memory, scf_max_cycle=scf_max_cycle,
            scf_history_matrices=scf_history_matrices,
            fixed_workspace_bytes=fixed_workspace_bytes, force_streamed=force_streamed,
        )
    return _BasisGridContext(
        basis=grid_ao_bundle.basis,
        coords=grid_ao_bundle.coords,
        grid_weights=grid_ao_bundle.grid_weights,
        geometry_is_traced=False,
        grid_ao_bundle=grid_ao_bundle,
        grid_ao=grid_ao_bundle.grid_ao,
    )


def _grid_ao_payload(
    context: _BasisGridContext,
    *,
    evaluate_cartesian_ao_with_derivatives: Callable[..., Any],
    needs_ao_laplacian: bool,
) -> tuple[Array, Array, Array | None]:
    bundle = context.grid_ao_bundle
    if bundle is not None:
        return bundle.ao, bundle.ao_deriv1, bundle.ao_laplacian
    if context.grid_ao is not None:
        return _empty_ao_payload(context.basis.nao, context.coords.dtype)
    if context.geometry_is_traced:
        return _dense_ao_payload(context.basis, context.coords,
            needs_ao_laplacian=needs_ao_laplacian, evaluate=evaluate_cartesian_ao_with_derivatives)
    raise ValueError("Non-traced basis/grid context is missing cached AO data.")


def _cache_bounded(
    cache: dict[tuple[Any, ...], Any],
    key: tuple[Any, ...],
    value: Any,
    maxsize: int,
) -> None:
    def nbytes(bundle):
        return sum(array.nbytes for array in (
            bundle.coords, bundle.grid_weights, bundle.ao,
            bundle.ao_deriv1, bundle.ao_laplacian,
        ) if array is not None)

    size = nbytes(value)
    if size > _GRID_AO_INPUT_CACHE_MAXBYTES:
        return
    cache.pop(key, None)
    resident = sum(nbytes(bundle) for bundle in cache.values())
    while cache and (len(cache) >= maxsize or resident + size > _GRID_AO_INPUT_CACHE_MAXBYTES):
        resident -= nbytes(cache.pop(next(iter(cache))))
    cache[key] = value
