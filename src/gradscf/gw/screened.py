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


__all__ = ["screened_w_imag_axis", "screened_w_imag_axis_matrix"]


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
