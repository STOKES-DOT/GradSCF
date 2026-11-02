"""Uncontract the trainable pool only; preserve core basis and all electrons."""
from dataclasses import replace

import jax.numpy as jnp

from gradscf.integrals.basis.contraction import contraction_matrix


def active_pool_basis(layout):
    counts = tuple(n if slot >= 0 else c for slot, n, c in zip(
        layout.slots, layout.topology.primitive_counts, layout.topology.contraction_counts))
    coefficients = tuple(jnp.eye(n, dtype=c.dtype) if slot >= 0 else c
        for slot, n, c in zip(layout.slots, layout.topology.primitive_counts,
                              layout.parameters.coefficients))
    return (replace(layout.topology, contraction_counts=counts),
            replace(layout.parameters, coefficients=coefficients))


def pool_projection(layout, outputs):
    """Normalized map from the mixed pool to the compact contracted basis.

    Active radial coefficients are shared across every m component. Fixed
    core blocks use identity because the pool already contains their original
    contractions; they must not be uncontracted or normalized a second time.
    """
    top, _ = active_pool_basis(layout)
    full = contraction_matrix(layout.topology, layout.bind(outputs))
    transform = jnp.zeros((top.nao, layout.topology.nao), dtype=full.dtype)
    primitive_offset = pool_offset = contracted_offset = 0
    for l, n, nc, slot in zip(layout.topology.angular_momenta,
            layout.topology.primitive_counts, layout.topology.contraction_counts, layout.slots):
        angular = (l+1)*(l+2)//2 if layout.topology.cart else 2*l+1
        primitive_width, contracted_width = n*angular, nc*angular
        if slot >= 0:
            block = full[primitive_offset:primitive_offset+primitive_width,
                         contracted_offset:contracted_offset+contracted_width]
        else:
            block = jnp.eye(contracted_width, dtype=full.dtype)
        transform = transform.at[pool_offset:pool_offset+block.shape[0],
            contracted_offset:contracted_offset+contracted_width].set(block)
        primitive_offset += primitive_width
        pool_offset += block.shape[0]
        contracted_offset += contracted_width
    return transform
