"""Shared DIIS ring history and normalized residual Gram solve.

History size is inferred from the supplied buffers. SCF retains its existing
default size and start count; other residual equations can choose their own.
"""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

DIIS_START_CYCLE = 2
DIIS_SPACE = 8


def diis_solve(
    fock_hist: Array,
    err_hist: Array,
    hist_count: Any,
) -> Array:
    history_size = fock_hist.shape[0]
    valid = (jnp.arange(history_size) < hist_count).astype(fock_hist.dtype)
    gram = err_hist @ err_hist.T
    # The DIIS coefficients are invariant to a common residual scale. Use
    # relative regularization so small physical gradients are not overwhelmed
    # by an absolute diagonal floor near SCF convergence.
    gram_scale = jnp.max(jnp.abs(gram))
    gram_scale = jnp.where(gram_scale > 0.0, gram_scale, 1.0)
    gram = gram / gram_scale
    diag_reg = jnp.asarray(
        jnp.finfo(fock_hist.dtype).eps * 50.0,
        dtype=fock_hist.dtype,
    )
    top = gram * (valid[:, None] * valid[None, :])
    top = top + jnp.diag(valid * diag_reg + (1.0 - valid))
    b = jnp.zeros(
        (history_size + 1, history_size + 1),
        dtype=fock_hist.dtype,
    )
    b = b.at[:history_size, :history_size].set(top)
    b = b.at[:history_size, history_size].set(-valid)
    b = b.at[history_size, :history_size].set(-valid)
    rhs = jnp.zeros((history_size + 1,), dtype=fock_hist.dtype)
    rhs = rhs.at[history_size].set(-1.0)
    coeff = jnp.linalg.solve(b, rhs)[:history_size]
    coeff = coeff * valid
    return jnp.tensordot(coeff, fock_hist, axes=(0, 0))


def diis_push(
    fock: Array,
    error: Array,
    fock_hist: Array,
    err_hist: Array,
    hist_head: Array,
    hist_count: Array,
) -> tuple[Array, Array, Array, Array]:
    history_size = fock_hist.shape[0]
    fock_hist = fock_hist.at[hist_head].set(fock)
    err_hist = err_hist.at[hist_head].set(error.reshape(-1))
    hist_head = (hist_head + 1) % jnp.asarray(history_size, dtype=hist_head.dtype)
    hist_count = jnp.minimum(
        hist_count + jnp.asarray(1, dtype=hist_count.dtype),
        jnp.asarray(history_size, dtype=hist_count.dtype),
    )
    return fock_hist, err_hist, hist_head, hist_count


def diis_extrapolate(
    fock: Array,
    error: Array,
    fock_hist: Array,
    err_hist: Array,
    hist_head: Array,
    hist_count: Array,
    *,
    solve: Callable = diis_solve,
    push: Callable = diis_push,
) -> tuple[Array, Array, Array, Array, Array]:
    """Insert one entry and extrapolate once two history entries are present.

    Callers own whether this operation runs on a given SCF cycle. Residual
    construction, including real/imaginary spinor packing, is method-specific.
    Explicit callables preserve the historical RKS instrumentation hooks.
    """
    fock_hist, err_hist, hist_head, hist_count = push(
        fock,
        error,
        fock_hist,
        err_hist,
        hist_head,
        hist_count,
    )

    fock_eff = jax.lax.cond(
        hist_count >= DIIS_START_CYCLE,
        lambda operand: solve(operand[0], operand[1], operand[2]),
        lambda operand: operand[3],
        operand=(fock_hist, err_hist, hist_count, fock),
    )
    return fock_eff, fock_hist, err_hist, hist_head, hist_count
