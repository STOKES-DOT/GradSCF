"""Differentiable AO quadrature blocks without a full-grid AO cache."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .ao import _evaluate_cartesian_ao_kernel, _as_device_array
from .locality import (
    ao_block_bounds, evaluate_local_ao, select_local_aos, spatial_grid_permutation,
)


def grid_slice(array, start, size: int, *, axis: int = 0):
    """Slice a fixed-size grid block, padding points beyond the grid with zero."""
    array = jnp.asarray(array)
    padding = [(0, 0)] * array.ndim
    padding[axis] = (0, size)
    return jax.lax.dynamic_slice_in_dim(
        jnp.pad(array, padding), start, size, axis=axis,
    )


def _ao_submatrix(matrix, indices):
    """Gather spatial or spin-block AO matrices without changing their symmetry."""
    return matrix if indices is None else matrix[..., indices[:, None], indices]


def _add_ao_submatrix(total, value, indices):
    """Accumulate local products; padded duplicate rows carry zero AO values."""
    return total + value if indices is None else total.at[..., indices[:, None], indices].add(value)


def ao_point_bytes(basis_or_provider, *, deriv: int) -> int:
    """Estimate one grid point's AO/primitive workspace from static metadata."""
    if deriv not in (0, 1, 2):
        raise ValueError("AO workspace requires deriv in (0,1,2).")
    exponents = getattr(basis_or_provider, "ao_exponents_padded", None)
    if exponents is None:
        exponents = basis_or_provider.exponents
    nprimitive = int(exponents.shape[1])
    components = (1, 4, 5)[deriv]
    # Coordinates can promote a float32 model basis to float64 AO values.
    itemsize = max(8, np.dtype(exponents.dtype).itemsize)
    return max(1, basis_or_provider.nao * itemsize * (12 * nprimitive + 2 * components + 8))


def ao_block_size(basis, *, deriv: int, max_memory: float, requested: int = 0,
                  device=None) -> int:
    """Plan an AO block within its allocated workspace in decimal MB.

    The caller charges resident arrays against the total calculation budget
    before supplying this workspace. Large blocks amortize AO/XC launches;
    CPU blocks use a practical 4096-point cap; larger GPU blocks can amortize
    launches. Explicit requested sizes may exceed either automatic cap.
    """
    if not np.isfinite(max_memory) or max_memory <= 0:
        raise ValueError("grid_max_memory must be a finite positive number of MB.")
    if requested < 0:
        raise ValueError("An AO block size must be nonnegative.")
    per_point = ao_point_bytes(basis, deriv=deriv)
    available = int(max_memory * 1e6)
    if available < per_point:
        raise ValueError("grid_max_memory cannot hold one AO quadrature point's workspace.")
    limit = available // per_point
    if limit >= 64:
        limit = limit // 64 * 64
    automatic_cap = 16384 if getattr(device, "platform", None) == "gpu" else 4096
    return min(requested or automatic_cap, limit)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AOGrid:
    """AO supplier with dynamic geometry/basis leaves and bounded grid blocks.

    ``reduce`` checkpoints AO production together with its consumer. ``map``
    is for full-grid *small* density features; it must not be used to cache AO
    arrays in production SCF/response. All coordinates and quadrature weights
    remain differentiable, including a non-divisible final block.
    """

    coords: Any
    weights: Any
    centers: Any
    exponents: Any
    coefficients: Any
    nprims: tuple[int, ...]
    angulars: tuple[tuple[int, int, int], ...]
    block_size: int
    adaptive: bool = False

    def tree_flatten(self):
        return ((self.coords, self.weights, self.centers, self.exponents, self.coefficients),
                (self.nprims, self.angulars, self.block_size, self.adaptive))

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children, *metadata)

    @classmethod
    def from_basis(cls, basis, coords, weights, *, block_size: int = 4096,
                   adaptive: bool = False):
        if int(block_size) <= 0:
            raise ValueError("block_size must be positive.")
        coords, weights = _as_device_array(coords), _as_device_array(weights)
        if coords.ndim != 2 or coords.shape[1] != 3 or weights.shape != coords.shape[:1]:
            raise ValueError("AO grid requires coordinates (ngrids,3) and weights (ngrids,).")
        return cls(coords, weights, _as_device_array(basis.ao_centers),
                   _as_device_array(basis.ao_exponents_padded),
                   _as_device_array(basis.ao_coefficients_padded),
                   tuple(basis.ao_nprims_tuple), tuple(basis.ao_angulars), int(block_size),
                   bool(adaptive))

    @property
    def nao(self) -> int:
        return int(self.centers.shape[0])

    @property
    def ngrids(self) -> int:
        return int(self.coords.shape[0])

    def block(self, start, size: int, *, deriv: int = 0):
        """Return AO, first derivatives, Laplacian, coordinates and weights.

        Absent derivative channels are ``None``. Response consumers may request
        a smaller block than the SCF block; this does not change the quadrature.
        """
        if size <= 0 or deriv not in (0, 1, 2):
            raise ValueError("AO blocks need size>0 and deriv in (0,1,2).")
        coords = grid_slice(self.coords, start, size)
        weights = grid_slice(self.weights, start, size)
        values = _evaluate_cartesian_ao_kernel(
            coords, self.centers, self.exponents, self.coefficients,
            jnp.asarray(self.nprims, dtype=jnp.int32),
            jnp.asarray(self.angulars, dtype=jnp.int32).reshape(self.nao, 3),
            deriv=deriv,
        )
        if deriv == 0:
            return values, None, None, coords, weights
        return values[0], values[:4], values[4] if deriv == 2 else None, coords, weights

    def reduce(self, function: Callable, initial, *, deriv: int = 0):
        """Reduce blocks; recompute block intermediates during reverse AD."""
        if self.ngrids == 0:
            return initial

        def body(carry, index):
            block = self.block(index * self.block_size, self.block_size, deriv=deriv)
            return function(carry, *block), None

        count = (self.ngrids + self.block_size - 1) // self.block_size
        return jax.lax.scan(jax.checkpoint(body), initial, jnp.arange(count))[0]

    def map(self, function: Callable, *, deriv: int = 0):
        """Collect grid-leading small feature arrays, trimming padded points."""
        def body(_, index):
            block = self.block(index * self.block_size, self.block_size, deriv=deriv)
            features = function(*block)
            for leaf in jax.tree_util.tree_leaves(features):
                if leaf.ndim == 0 or leaf.shape[0] != self.block_size:
                    raise ValueError("AOGrid.map outputs must have a leading grid-block axis.")
            return None, features

        count = (self.ngrids + self.block_size - 1) // self.block_size
        _, features = jax.lax.scan(jax.checkpoint(body), None, jnp.arange(count))
        return jax.tree_util.tree_map(
            lambda value: value.reshape((-1,) + value.shape[2:])[:self.ngrids], features,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _LocalAOGrid:
    """Private compact reduction with dynamic support and exact overflow fallback.

    Only ``reduce_local`` uses spatial order and AO screening. Generic methods
    preserve the raw grid order for global/neural consumers. A positive cutoff
    is a piecewise-smooth approximation shared by primal and AD; no AO support
    or derived AO arrays are cached. Workspace must cover the full-AO fallback.
    """

    raw: AOGrid
    order: Any
    cutoff: float
    capacity: int
    adaptive = False

    def tree_flatten(self):
        return (self.raw, self.order), (self.cutoff, self.capacity)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children, *metadata)

    @classmethod
    def from_grid(cls, raw: AOGrid, *, cutoff: float, capacity: int,
                  cell_size: float = 4.0):
        if (not np.isfinite(cutoff) or cutoff < 0 or not np.isfinite(capacity)
                or capacity < 0 or int(capacity) != capacity):
            raise ValueError("AO locality requires a finite nonnegative cutoff and integer capacity.")
        order, _ = spatial_grid_permutation(raw.coords, cell_size=cell_size)
        return cls(raw, order, float(cutoff), int(capacity))

    @property
    def coords(self):
        return self.raw.coords

    @property
    def weights(self):
        return self.raw.weights

    @property
    def centers(self):
        return self.raw.centers

    @property
    def exponents(self):
        return self.raw.exponents

    @property
    def coefficients(self):
        return self.raw.coefficients

    @property
    def nprims(self):
        return self.raw.nprims

    @property
    def angulars(self):
        return self.raw.angulars

    @property
    def nao(self):
        return self.raw.nao

    @property
    def ngrids(self):
        return self.raw.ngrids

    @property
    def block_size(self):
        return self.raw.block_size

    def block(self, start, size, *, deriv=0):
        return self.raw.block(start, size, deriv=deriv)

    def map(self, function, *, deriv=0):
        return self.raw.map(function, deriv=deriv)

    def reduce(self, function, initial, *, deriv=0):
        return self.raw.reduce(function, initial, deriv=deriv)

    def reduce_local(self, function: Callable, initial, *, deriv: int = 0):
        """Reduce local AO columns, passing their global indices to the consumer.

        The consumer must gather its density/MO rows and scatter matrix updates
        into the same full-size carry in both branches. Unused compact columns
        are zero, including derivatives. Overflow never truncates active AOs.
        """
        if deriv not in (0, 1, 2):
            raise ValueError("AO locality requires deriv in (0,1,2).")
        if self.ngrids == 0:
            return initial
        grid = replace(self.raw, coords=self.raw.coords[self.order],
                       weights=self.raw.weights[self.order])
        full_indices = jnp.arange(self.nao, dtype=jnp.int32)

        def body(carry, index):
            start = index*self.block_size

            def dense(value):
                return function(value, *grid.block(start, self.block_size, deriv=deriv),
                                full_indices)

            if self.cutoff == 0:
                return dense(carry), None
            coords = grid_slice(grid.coords, start, self.block_size)
            weights = grid_slice(grid.weights, start, self.block_size)
            point_mask = start + jnp.arange(self.block_size) < self.ngrids
            bounds = ao_block_bounds(self.raw, coords, weights, deriv=deriv,
                                     coefficient_response=True, point_mask=point_mask)
            indices, valid, overflow = select_local_aos(
                bounds, cutoff=self.cutoff, capacity=self.capacity)

            def compact(value, width=self.capacity):
                selected = indices[:width]
                block = evaluate_local_ao(self.raw, coords, weights, selected, valid[:width],
                                          deriv=deriv)
                return function(value, *block, selected)

            def bucketed(value):
                if self.capacity < 256:
                    return compact(value)
                # A false slot at halfway means the smaller bucket holds every
                # active AO; this changes padding only, never the support.
                return jax.lax.cond(valid[self.capacity//2], compact,
                                    lambda v: compact(v, self.capacity//2), value)

            return jax.lax.cond(overflow, dense, bucketed, carry), None

        count = (self.ngrids+self.block_size-1)//self.block_size
        return jax.lax.scan(jax.checkpoint(body), initial, jnp.arange(count))[0]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _CachedAOGrid:
    """Call-local derived AO data; never attached to a raw provider or result."""
    coords: Any
    weights: Any
    values: Any  # (grid, value/derivative component, AO), no duplicate AO plane
    block_size: int
    adaptive = False

    def tree_flatten(self):
        return (self.coords, self.weights, self.values), (self.block_size,)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children, *metadata)

    @property
    def nao(self):
        return self.values.shape[-1]

    @property
    def ngrids(self):
        return self.values.shape[0]

    def block(self, start, size, *, deriv=0):
        components = (1, 4, 5)
        if size <= 0 or deriv not in (0, 1, 2) or components[deriv] > self.values.shape[1]:
            raise ValueError("Cached AO block request exceeds its prepared derivative order.")
        indices = start + jnp.arange(size)
        valid = (indices >= 0) & (indices < self.ngrids)
        indices = jnp.clip(indices, 0, self.ngrids - 1)
        # Gather only this tile; padding the full cached grid would duplicate GBs
        # of data on every SCF block and defeat the preparation memory bound.
        values = jnp.where(valid[:, None, None], jnp.take(self.values, indices, axis=0), 0.)
        coords = jnp.where(valid[:, None], jnp.take(self.coords, indices, axis=0), 0.)
        weights = jnp.where(valid, jnp.take(self.weights, indices), 0.)
        derivative = values[:, :4].transpose(1, 0, 2) if deriv else None
        laplacian = values[:, 4] if deriv == 2 else None
        return values[:, 0], derivative, laplacian, coords, weights

    reduce = AOGrid.reduce
    map = AOGrid.map


@partial(jax.jit, static_argnames=("block_size", "deriv"))
def _materialize_scf_grid(raw, *, block_size, deriv):
    """Fill an exact-size derived AO buffer with bounded, differentiable tiles."""
    components = (1, 4, 5)[deriv]
    values = jnp.zeros((raw.ngrids, components, raw.nao),
                       dtype=jnp.result_type(raw.coords, raw.exponents, raw.coefficients))

    @partial(jax.checkpoint, static_argnums=(2,))
    def produce(index, output, width):
        start = index * block_size
        ao, derivative, laplacian, _, _ = raw.block(start, width, deriv=deriv)
        tile = ao[:, None, :] if deriv == 0 else derivative.transpose(1, 0, 2)
        if deriv == 2:
            tile = jnp.concatenate([tile, laplacian[:, None, :]], axis=1)
        return jax.lax.dynamic_update_slice(output, tile, (start, 0, 0))

    full, tail = divmod(raw.ngrids, block_size)
    values = jax.lax.fori_loop(0, full, lambda index, output: produce(index, output, block_size), values)
    if tail:
        values = produce(full, values, tail)
    return _CachedAOGrid(raw.coords, raw.weights, values, block_size)
