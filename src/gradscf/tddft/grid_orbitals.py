"""Budgeted MO fields from streamed AO blocks; never cache AO or OV tensors."""
from dataclasses import replace

import jax.numpy as jnp
from jax.lax import Precision


def streamed_orbital_fields(provider, occupied, virtual, *, feature_kind,
                            dtype, block_size):
    """Return (component, grid, orbital) fields with differentiable AO input.

    The caller charges retained fields plus scan padding against its action
    budget before requesting this preparation. Gaussian AO values live only
    in one block and are recomputed by the provider's existing AD mechanism.
    """
    provider = replace(provider, block_size=min(provider.block_size, int(block_size)))
    deriv = 0 if feature_kind == 'LDA' else 1

    def project(ao, dao, laplacian, coords, weights):
        fields = ao[:, None, :] if deriv == 0 else jnp.moveaxis(dao[:4], 0, 1)
        fields = jnp.asarray(fields, dtype=dtype)
        return (jnp.einsum('gxp,pi->gxi', fields, occupied, precision=Precision.HIGHEST),
                jnp.einsum('gxp,pa->gxa', fields, virtual, precision=Precision.HIGHEST))

    occupied_fields, virtual_fields = provider.map(project, deriv=deriv)
    return jnp.moveaxis(occupied_fields, 1, 0), jnp.moveaxis(virtual_fields, 1, 0)
