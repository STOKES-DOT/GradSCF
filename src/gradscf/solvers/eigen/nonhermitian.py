"""Bounded real non-Hermitian eigenpairs with biorthogonal energy response."""

import jax
import jax.numpy as jnp
from ..types import NonHermitianSolverConfig, NonHermitianResult
from ..operators import as_operator, validate_real_square
from ..diagnostics import require_converged_derivative


def solve_nonhermitian(matrix_or_operator, *, config=None):
    """Lowest-real-part eigenpairs; no Hermitianization or hidden root filtering.

    Only isolated real eigenvalue JVP/VJP is exposed. Left/right vectors are
    stopped numerical outputs. Complex selected roots give NaN real outputs;
    the full complex spectrum is retained in raw_eigenvalues. Exceptional or
    poorly conditioned roots invalidate energy response. This bounded reference
    deliberately has no matrix-free iterative fallback.
    """
    cfg = NonHermitianSolverConfig() if config is None else config
    op = as_operator(matrix_or_operator)
    validate_real_square(op)
    n = op.shape[0]
    k = cfg.nroots
    if n > cfg.max_dense:
        raise ValueError("Non-Hermitian dense reference exceeds max_dense")
    if k > n:
        raise ValueError("nroots exceeds operator dimension")
    blocks = []
    for start in range(0, n, cfg.block_size):
        indices = jnp.arange(start, min(start + cfg.block_size, n))
        probes = jax.nn.one_hot(indices, n, dtype=op.dtype).T
        blocks.append(op.apply(probes))
    a = jax.lax.stop_gradient(jnp.concatenate(blocks, axis=1))
    wr, vr = jnp.linalg.eig(a)
    wl, vl = jnp.linalg.eig(a.T)
    order = jnp.lexsort((wr.imag, wr.real))
    wr, vr = wr[order], vr[:, order]

    # Pair spectra without assigning one left vector to multiple right roots.
    def pair(i, carry):
        indices, used = carry
        score = jnp.where(used, jnp.inf, jnp.abs(wl - wr[i]))
        index = jnp.argmin(score).astype(jnp.int32)
        return indices.at[i].set(index), used.at[index].set(True)

    indices, _ = jax.lax.fori_loop(
        0, k, pair, (jnp.zeros(k, jnp.int32), jnp.zeros(n, bool))
    )
    r = vr[:, :k].real
    r = r / jnp.maximum(jnp.linalg.norm(r, axis=0), jnp.finfo(r.dtype).tiny)
    l0 = vl[:, indices].real
    gram = l0.T @ r
    singular = jnp.linalg.svd(gram, compute_uv=False)
    pairing = jnp.all(jnp.isfinite(singular)) & (
        singular[-1] > 64 * jnp.finfo(r.dtype).eps * jnp.maximum(1.0, singular[0])
    )
    l = l0 @ jnp.linalg.solve(
        jnp.where(pairing, gram.T, jnp.eye(k, dtype=r.dtype)), jnp.eye(k, dtype=r.dtype)
    )
    energies = wr[:k].real
    right_res = jnp.linalg.norm(a @ r - r * energies, axis=0)
    left_norm = jnp.linalg.norm(l, axis=0)
    left_res = jnp.linalg.norm(a.T @ l - l * energies, axis=0) / jnp.maximum(
        left_norm, 1.0
    )
    bio = jnp.linalg.norm(l.T @ r - jnp.eye(k, dtype=r.dtype))
    real = (jnp.abs(wr[:k].imag) <= cfg.imaginary_tol) & (
        jnp.abs(wl[indices].imag) <= cfg.imaginary_tol
    )
    conv = (
        jnp.all(jnp.isfinite(a))
        & pairing
        & real
        & (right_res <= cfg.atol)
        & (left_res <= cfg.atol)
        & (bio <= 1e-8)
    )
    condition = jnp.where(pairing, left_norm, jnp.inf)
    distances = jnp.abs(wr[:k, None] - wr[None, :])
    distances = distances.at[jnp.arange(k), jnp.arange(k)].set(jnp.inf)
    margin = cfg.gap_atol + cfg.gap_rtol * jnp.maximum(
        jnp.abs(wr[:k, None]), jnp.abs(wr[None, :])
    )
    margin += 2 * condition[:, None] * jnp.maximum(right_res, left_res)[:, None]
    valid = (
        jnp.all(conv)
        & jnp.all(condition < cfg.max_condition)
        & jnp.all(distances > margin)
        & jnp.all(jnp.isfinite(wr))
    )
    r, l = jax.lax.stop_gradient(r), jax.lax.stop_gradient(l)
    applied = op.apply(r)
    live_zero = applied - jax.lax.stop_gradient(applied)
    values = energies + jnp.sum(l * live_zero, axis=0)
    values = require_converged_derivative(values, valid)
    values = require_converged_derivative(jnp.where(real, values, jnp.nan), valid)
    return NonHermitianResult(
        values,
        jnp.where(real[None, :], r, jnp.nan),
        jnp.where(real[None, :], l, jnp.nan),
        right_res,
        left_res,
        conv,
        jnp.broadcast_to(valid, (k,)),
        condition,
        bio,
        wr,
    )
