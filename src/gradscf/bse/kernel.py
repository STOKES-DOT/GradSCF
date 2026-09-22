"""Factorized static TDA-BSE kernel; no electron-hole matrix or private solver."""

import jax.numpy as jnp
from ..solvers import LinearOperator
from ..solvers.diagnostics import require_converged_derivative
from ..gw.screened import apply_static_screening


def build_tda_operator(
    qp_energy, mo_factors, space, screening, *, singlet=True, block_size=16
):
    qp, l = jnp.asarray(qp_energy), jnp.asarray(mo_factors)
    if qp.shape != (space.nmo,) or l.ndim != 3 or l.shape[1:] != (space.nmo, space.nmo):
        raise ValueError("BSE energy/factor shapes do not match the orbital space")
    if jnp.iscomplexobj(qp) or jnp.iscomplexobj(l):
        raise NotImplementedError("Molecular TDA-BSE currently requires real inputs")
    if screening.dielectric.shape != (l.shape[0], l.shape[0]):
        raise ValueError("Screening and factors have different auxiliary dimensions")
    if not space.size or block_size < 1:
        raise ValueError("BSE requires nonempty transitions and a positive block_size")
    oi, va = jnp.asarray(space.occupied), jnp.asarray(space.virtual)
    no, nv = len(space.occupied), len(space.virtual)
    loo = l[:, oi[:, None], oi[None, :]]
    lov = l[:, oi[:, None], va[None, :]]
    lvv = l[:, va[:, None], va[None, :]]
    # Screen one virtual slab at a time: never batch dielectric factorizations
    # over all RHS columns, and never construct W_ij,ab or the BSE matrix.
    screened = jnp.concatenate(
        [
            apply_static_screening(screening, lvv[:, start : start + block_size, :])
            for start in range(0, nv, block_size)
        ],
        axis=1,
    )
    gap = qp[va][None, :] - qp[oi][:, None]
    kappa = 2.0 if singlet else 0.0
    diagonal = (
        gap
        + kappa * jnp.sum(lov**2, axis=0)
        - jnp.einsum(
            "Pi,Pa->ia",
            jnp.diagonal(loo, axis1=1, axis2=2),
            jnp.diagonal(screened, axis1=1, axis2=2),
        )
    ).reshape(-1)
    naux = l.shape[0]
    aux_block_size = min(block_size, max(1, naux))
    blocks = (naux + aux_block_size - 1) // aux_block_size
    pad = blocks * aux_block_size - naux
    lo = jnp.pad(loo, ((0, pad), (0, 0), (0, 0))).reshape(
        blocks, aux_block_size, no, no
    )
    lv = jnp.pad(screened, ((0, pad), (0, 0), (0, 0))).reshape(
        blocks, aux_block_size, nv, nv
    )

    def apply(values):
        x = values.reshape(no, nv, -1)
        y = gap[:, :, None] * x
        charge = jnp.einsum("Pjb,jbk->Pk", lov, x)
        y += kappa * jnp.einsum("Pia,Pk->iak", lov, charge)
        # Static physical blocks also support jax.linear_transpose used by the
        # shared adjoint solver. A scan capturing x in its carry body does not.
        for index in range(blocks):
            partial = jnp.einsum("Pij,jbk->Pibk", lo[index], x)
            y -= jnp.einsum("Pibk,Pab->iak", partial, lv[index])
        return require_converged_derivative(y.reshape(space.size, -1), screening.valid)

    return LinearOperator(
        (space.size, space.size),
        jnp.result_type(qp, l),
        lambda x: apply(x[:, None])[:, 0],
        diagonal=diagonal,
        matmat=apply,
    )
