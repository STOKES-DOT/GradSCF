"""Shared batching for native derivative product primitives."""
import jax
from jax.interpreters import batching


def batch_product(primitive, args, axes):
    size = next(a.shape[axis] for a, axis in zip(args, axes) if axis is not None)
    arrays = tuple(batching.bdim_at_front(a, axis, size) for a, axis in zip(args, axes))
    return jax.lax.map(lambda xs: primitive.bind(*xs), arrays), 0
