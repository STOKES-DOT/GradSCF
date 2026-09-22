"""Screened Coulomb interaction W on the imaginary frequency axis.

Within the random phase approximation the screened Coulomb interaction in a
low-rank auxiliary representation is

    W_mn(iw) = sum_PQ B_P[m,n] [(1 - Pi(iw))^{-1} - 1]_PQ B_Q[m,n]

evaluated with the density response ``Pi`` from
:mod:`gradscf.gw.polarizability`.  The matrix inverse is written as the
linear solve ``solve(1 - Pi, Pi)`` (algebraically identical to
``inv(1 - Pi) - I``) which keeps the operation smooth for JAX autodiff.

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- X. Ren et al., New J. Phys. 14, 053020 (2012).
  DOI:10.1088/1367-2630/14/5/053020
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array
from typing import NamedTuple
from math import isfinite
from numbers import Integral
from ..solvers import LinearSolverConfig, solve_linear
from ..solvers.diagnostics import require_converged_derivative


class StaticScreening(NamedTuple):
    """Full auxiliary dielectric, not the self-energy's contracted W-v."""
    dielectric: Array
    valid: Array
    min_gap: Array


def build_static_screening(
    mo_energy, mo_factors, *, occupied, virtual, gap_tol=1e-10, max_aux=1024
):
    """Closed-shell static RPA epsilon=I-Pi(0) for metric-whitened real factors.

    Screening indices are independent of the optical excitation window. Empty
    screening transitions give epsilon=I (bare interaction). Real symmetric
    MO factors and positive screening gaps are required. Invalid numerical
    inputs retain diagnostics with valid=False; there is no gap clipping model.
    """
    e, l = jnp.asarray(mo_energy), jnp.asarray(mo_factors)
    if e.ndim != 1 or l.ndim != 3 or l.shape[1:] != (e.size, e.size):
        raise ValueError("Screening requires energy (nmo,) and factors (naux,nmo,nmo)")
    if jnp.iscomplexobj(e) or jnp.iscomplexobj(l):
        raise NotImplementedError("Static molecular screening requires real inputs")
    if (
        not isfinite(gap_tol)
        or gap_tol <= 0
        or not isinstance(max_aux, Integral)
        or max_aux < 1
    ):
        raise ValueError("Invalid static screening tolerance or auxiliary limit")
    if l.shape[0] > max_aux:
        raise ValueError("Static screening exceeds max_aux")
    occ, vir = tuple(occupied), tuple(virtual)
    if any(
        not isinstance(p, Integral) or not 0 <= p < e.size for p in occ + vir
    ) or len(set(occ + vir)) != len(occ) + len(vir):
        raise ValueError("Screening indices must be distinct, disjoint and in range")
    dtype = jnp.result_type(e, l, 1.0)
    e, l = e.astype(dtype), l.astype(dtype)
    oi, va = jnp.asarray(occ, dtype=jnp.int32), jnp.asarray(vir, dtype=jnp.int32)
    gaps = e[va][None, :] - e[oi][:, None]
    minimum = jnp.min(gaps, initial=jnp.inf)
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(l), initial=0.0))
    valid = (
        jnp.all(jnp.isfinite(l))
        & jnp.all(jnp.isfinite(gaps))
        & (minimum > gap_tol)
        & (
            jnp.max(jnp.abs(l - l.swapaxes(1, 2)), initial=0.0)
            <= 64 * jnp.finfo(dtype).eps * scale
        )
    )
    lov = l[:, oi[:, None], va[None, :]]
    safe_gaps = jnp.where(gaps > gap_tol, gaps, 1.0)
    dielectric = jnp.eye(l.shape[0], dtype=dtype) + 4 * jnp.einsum(
        "Pia,Qia,ia->PQ", lov, lov, 1 / safe_gaps
    )
    return StaticScreening(dielectric, valid, minimum)


def apply_static_screening(state, values):
    """Apply the full screened auxiliary metric epsilon^-1 to one/block RHS.

    Shared direct linear solves factor the matrix for a block of right sides;
    they own primal, transpose and implicit response checks. No inverse is built.
    """
    values = jnp.asarray(values)
    naux = state.dielectric.shape[0]
    if values.ndim < 1 or values.shape[0] != naux:
        raise ValueError("Screening RHS must have leading dimension naux")
    columns = 1
    for n in values.shape[1:]:
        columns *= n
    rhs = values.reshape(naux, columns)
    out = solve_linear(
        state.dielectric,
        rhs,
        config=LinearSolverConfig(
            method="direct", rtol=1e-11, atol=1e-13, max_dense=max(1, naux)
        ),
    )
    valid = state.valid & out.converged
    result = require_converged_derivative(out.solution, valid)
    return jnp.where(valid, result, jnp.nan).reshape(values.shape)


def screened_w_imag_axis(
    b_mn: Array,
    response_fn: Callable[[Array], Array],
    freqs: Array,
    *,
    conjugate: bool = False,
) -> Array:
    """Compute ``W_mn(iw)`` for every imaginary grid frequency.

    Parameters
    ----------
    b_mn:
        Low-rank factors in the full MO basis, shape ``(naux, nmo, nmo)``.
    response_fn:
        Callable mapping a real scalar frequency to the density response
        matrix ``Pi`` of shape ``(naux, naux)`` (real or complex dtype; the
        dtype propagates to the output).
    freqs:
        1D array of imaginary-axis quadrature nodes.

    Returns
    -------
    Array of shape ``(nw, nmo, nmo)`` with ``W_mn`` at each frequency.
    """
    b_mn = jnp.asarray(b_mn)
    freqs = jnp.asarray(freqs)
    naux = b_mn.shape[0]

    def at_frequency(omega: Array) -> Array:
        pi = response_fn(omega)
        eye = jnp.eye(naux, dtype=pi.dtype)
        # (1 - Pi)^{-1} - I == solve(1 - Pi, Pi); solve has a well-defined
        # adjoint and avoids explicitly forming the inverse.
        screened = jnp.linalg.solve(eye - pi, pi)
        first = b_mn.conj() if conjugate else b_mn
        return jnp.einsum("Pmn,PQ,Qmn->mn", first, screened, b_mn, precision=Precision.HIGHEST)

    # Batch over frequencies sequentially when the batched (naux, naux)
    # intermediates would exceed the memory budget (plane-wave bases).
    est_bytes = freqs.shape[0] * naux * naux * 16 * 2
    if est_bytes > (1 << 31):
        return jax.lax.map(at_frequency, freqs)
    return jax.vmap(at_frequency)(freqs)


__all__ = ["screened_w_imag_axis", "screened_w_imag_axis_matrix",
           "StaticScreening", "build_static_screening", "apply_static_screening"]


def screened_w_imag_axis_matrix(
    b_mn: Array,
    response_fn: Callable[[Array], Array],
    freqs: Array,
    *,
    conjugate: bool = False,
) -> Array:
    """Full screened tensor ``W[q; m, n](iw)`` for off-diagonal self-energies.

    W[q; m, n] = sum_PQ conj(B_P[m,q]) [(1-Pi)^{-1} - 1]_PQ B_Q[q,n]

    Returns shape ``(nw, nq, nmo, nmo)`` (q = intermediate state index).
    """
    b_mn = jnp.asarray(b_mn)
    freqs = jnp.asarray(freqs)
    naux = b_mn.shape[0]

    def at_frequency(omega: Array) -> Array:
        pi = response_fn(omega)
        eye = jnp.eye(naux, dtype=pi.dtype)
        screened = jnp.linalg.solve(eye - pi, pi)
        first = b_mn.conj() if conjugate else b_mn
        return jnp.einsum("Pmq,PQ,Qqn->qmn", first, screened, b_mn, precision=Precision.HIGHEST)

    est_bytes = freqs.shape[0] * naux * naux * 16 * 2
    if est_bytes > (1 << 31):
        return jax.lax.map(at_frequency, freqs)
    return jax.vmap(at_frequency)(freqs)
