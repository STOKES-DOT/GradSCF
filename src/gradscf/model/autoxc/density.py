"""Blocked real-space quadrature for candidate-defined two-point kernels."""
from math import isqrt

import jax
import jax.numpy as jnp


def pair_block_size(ngrids, block_size, max_memory, dtype):
    # Budget for the pair distances, kernel values and reduction work arrays.
    elements = int(max_memory * 1e6 / (16 * jnp.dtype(dtype).itemsize))
    if elements < 1:
        raise ValueError('Memory budget is too small for pair quadrature')
    return min(ngrids, block_size, 256, isqrt(elements))


def pair_quadrature(coords, weights, block_size, dtype):
    """Build a pair reduction with bounded tiles in the forward and reverse passes."""
    ngrids = coords.shape[0]
    padding = -ngrids % block_size
    coords = jnp.pad(coords.astype(dtype), ((0, padding), (0, 0)), mode='edge')
    coords = coords.reshape((-1, block_size, coords.shape[-1]))
    weights = jnp.pad(weights.astype(dtype), (0, padding)).reshape((-1, block_size))
    blocks = weights.shape[0]

    def tile(field, index):
        field = jnp.asarray(field, dtype=dtype)
        if field.ndim == 0:
            return jnp.broadcast_to(field, (block_size,))
        start = index * block_size
        indices = jnp.minimum(start + jnp.arange(block_size), ngrids - 1)
        return field[indices]

    def evaluate(operator, field, kernel, rank):
        normalize = operator == 'pair_average'

        def target_block(target_index):
            target_coords = coords[target_index]
            numerator_zero = jnp.zeros((block_size,) + (3,)*rank, dtype=dtype)
            denominator_zero = jnp.zeros(block_size, dtype=dtype)

            def accumulate(source_index, totals):
                numerator, denominator = totals
                delta = coords[source_index][None, :, :] - target_coords[:, None, :]
                distance2 = jnp.sum(delta * delta, axis=-1)
                values = kernel(distance2, lambda x: tile(x, target_index)[:, None],
                                lambda x: tile(x, source_index)[None, :])
                weighted = jnp.where(weights[source_index] != 0, values * weights[source_index], 0.)
                weighted = jnp.broadcast_to(weighted, (block_size, block_size))
                source_field = tile(field, source_index)
                if rank:
                    tensor = delta
                    if rank == 2:
                        tensor = (delta[..., :, None]*delta[..., None, :]
                                  - distance2[..., None, None]*jnp.eye(3, dtype=dtype)/3)
                    numerator = numerator + jnp.einsum('ij,ij...->i...', weighted*source_field, tensor)
                else:
                    numerator = numerator + weighted @ source_field
                if normalize:
                    denominator = denominator + jnp.sum(weighted, axis=1)
                return numerator, denominator

            numerator, denominator = jax.lax.fori_loop(
                0, blocks, jax.checkpoint(accumulate), (numerator_zero, denominator_zero)
            )
            if rank:
                return jnp.sum(numerator.reshape(block_size, -1)**2, axis=1)
            if normalize:
                return jnp.where(denominator != 0, numerator / denominator, jnp.nan)
            return numerator

        values = jax.lax.map(jax.checkpoint(target_block), jnp.arange(blocks))
        return values.reshape(-1)[:ngrids]

    return evaluate
