"""Factorized static BSE actions with bounded screening right-hand sides."""

import jax
import jax.numpy as jnp
from ..solvers import LinearOperator
from ..solvers.diagnostics import require_converged_derivative
from ..gw.screened import apply_static_screening, solve_static_screening


def build_tda_operator(
    qp_energy, mo_factors, space, screening, *, singlet=True, block_size=16
):
    qp, l = jnp.asarray(qp_energy), jnp.asarray(mo_factors)
    if qp.shape != (space.nmo,) or l.ndim != 3 or l.shape[1:] != (space.nmo, space.nmo):
        raise ValueError("BSE energy/factor shapes do not match the orbital space")
    if jnp.iscomplexobj(qp) or jnp.iscomplexobj(l):
        raise NotImplementedError("Molecular TDA-BSE currently requires real inputs")
    if screening.naux != l.shape[0]:
        raise ValueError("Screening and factors have different auxiliary dimensions")
    if not space.size or block_size < 1:
        raise ValueError("BSE requires nonempty transitions and a positive block_size")
    oi, va = jnp.asarray(space.occupied), jnp.asarray(space.virtual)
    no, nv = len(space.occupied), len(space.virtual)
    loo = l[:, oi[:, None], oi[None, :]]
    lov = l[:, oi[:, None], va[None, :]]
    gap = qp[va][None, :] - qp[oi][:, None]
    kappa = 2.0 if singlet else 0.0
    # Only diagonal virtual pairs are needed for the Davidson preconditioner.
    diagonal = gap + kappa * jnp.sum(lov**2, axis=0)
    for start in range(0, nv, block_size):
        vb = va[start : start + block_size]
        screened = apply_static_screening(screening, l[:, vb, vb])
        diagonal = diagonal.at[:, start : start + block_size].add(
            -jnp.einsum("Pi,Pa->ia", jnp.diagonal(loo, axis1=1, axis2=2), screened)
        )

    @jax.checkpoint
    def direct_block(x, virtual_pairs):
        # Contract before screening: RHS is (naux,nocc,block,nvec), not
        # (naux,nvir,nvir). Rematerialize these temporaries in reverse mode.
        if no * x.shape[-1] <= nv:
            rhs = jnp.einsum("Pab,jbk->Pjak", virtual_pairs, x)
            screened = solve_static_screening(screening, rhs).solution
            return jnp.einsum("Pij,Pjak->iak", loo, screened)
        # Wide Davidson blocks use fewer RHS by screening a bare pair slab.
        # Auxiliary blocking also bounds the subsequent trial-vector workspace.
        screened = apply_static_screening(screening, virtual_pairs)
        out = jnp.zeros((no, virtual_pairs.shape[1], x.shape[-1]), dtype=x.dtype)
        for start in range(0, l.shape[0], block_size):
            partial = jnp.einsum("Pab,jbk->Pjak", screened[start:start+block_size], x)
            out += jnp.einsum("Pij,Pjak->iak", loo[start:start+block_size], partial)
        return out

    def apply(values):
        x = values.reshape(no, nv, -1)
        charge = jnp.einsum("Pjb,jbk->Pk", lov, x)
        y = gap[:, :, None] * x + kappa * jnp.einsum("Pia,Pk->iak", lov, charge)
        for start in range(0, nv, block_size):
            pairs = l[:, va[start : start + block_size, None], va[None, :]]
            y = y.at[:, start : start + block_size, :].add(-direct_block(x, pairs))
        return require_converged_derivative(y.reshape(space.size, -1), screening.valid)

    return LinearOperator(
        (space.size, space.size),
        jnp.result_type(qp, l),
        lambda x: apply(x[:, None])[:, 0],
        diagonal=diagonal.reshape(-1),
        matmat=apply,
    )


def build_bse_operators(
    qp_energy, mo_factors, space, screening, *, singlet=True, block_size=16
):
    """Return A/B actions; screened pair factors are never cached.

    B[ia,jb] = kappa (ia|jb) - W[ib,aj]. Screening is applied after
    contraction with the trial vectors, in bounded occupied-index blocks.
    """
    a = build_tda_operator(
        qp_energy, mo_factors, space, screening, singlet=singlet, block_size=block_size
    )
    l = jnp.asarray(mo_factors)
    oi, va = jnp.asarray(space.occupied), jnp.asarray(space.virtual)
    no, nv = len(space.occupied), len(space.virtual)
    lov = l[:, oi[:, None], va[None, :]]
    kappa = 2.0 if singlet else 0.0
    diagonal = kappa * jnp.sum(lov**2, axis=0).reshape(-1)
    for start in range(0, space.size, block_size):
        indices = jnp.arange(start, min(start + block_size, space.size))
        pairs = l[:, oi[indices // nv], va[indices % nv]]
        screened = apply_static_screening(screening, pairs)
        diagonal = diagonal.at[start : start + block_size].add(
            -jnp.sum(pairs * screened, axis=0)
        )

    @jax.checkpoint
    def direct_block(x, occupied_virtual_pairs):
        if no * x.shape[-1] <= nv:
            rhs = jnp.einsum("Pib,jbk->Pijk", occupied_virtual_pairs, x)
            screened = solve_static_screening(screening, rhs).solution
            return jnp.einsum("Pja,Pijk->iak", lov, screened)
        screened = apply_static_screening(screening, occupied_virtual_pairs)
        out = jnp.zeros((occupied_virtual_pairs.shape[1], nv, x.shape[-1]), dtype=x.dtype)
        for start in range(0, l.shape[0], block_size):
            partial = jnp.einsum("Pib,jbk->Pijk", screened[start:start+block_size], x)
            out += jnp.einsum("Pja,Pijk->iak", lov[start:start+block_size], partial)
        return out

    def apply(values):
        x = values.reshape(no, nv, -1)
        charge = jnp.einsum("Pjb,jbk->Pk", lov, x)
        y = kappa * jnp.einsum("Pia,Pk->iak", lov, charge)
        for start in range(0, no, block_size):
            y = y.at[start : start + block_size].add(
                -direct_block(x, lov[:, start : start + block_size])
            )
        return require_converged_derivative(y.reshape(space.size, -1), screening.valid)

    b = LinearOperator(
        a.shape,
        a.dtype,
        lambda x: apply(x[:, None])[:, 0],
        diagonal=diagonal,
        matmat=apply,
    )
    return a, b
