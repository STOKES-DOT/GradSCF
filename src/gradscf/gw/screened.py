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
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from ..solvers import LinearOperator, LinearSolverConfig, solve_linear
from ..scf._pytree import pytree_dataclass
from ..solvers.diagnostics import require_converged_derivative


@pytree_dataclass(static_fields=("config",))
@dataclass(frozen=True)
class StaticScreening:
    """Full static W metric, stored densely or through transition factors."""
    dielectric: object
    valid: Array
    min_gap: Array
    transition_factors: object = None
    inverse_gaps: object = None
    config: object = None
    cholesky: object = None

    @property
    def naux(self):
        data = self.dielectric if self.dielectric is not None else self.transition_factors
        return data.shape[0]

    def operator(self):
        if self.dielectric is not None:
            from ..solvers.operators import as_operator
            return as_operator(self.dielectric)
        l, inverse_gaps = self.transition_factors, self.inverse_gaps

        def apply(v):
            return v + 4 * (l @ (inverse_gaps * (l.T @ v)))

        diagonal = 1 + 4 * jnp.sum(l**2 * inverse_gaps, axis=1)
        return LinearOperator((self.naux, self.naux), l.dtype, apply,
                              diagonal=diagonal, transpose_matvec=apply)


def build_static_screening(
    mo_energy, mo_factors, *, occupied, virtual, gap_tol=1e-10, max_aux=1024,
    config=None
):
    """Static RPA epsilon=I-Pi(0) for metric-whitened real factors.

    Restricted energies/factors have shape (nmo,)/(naux,nmo,nmo). Spin inputs
    have a leading axis of length two and paired occupied/virtual windows;
    both spin responses contribute to one charge dielectric.

    Screening indices are independent of the optical excitation window. Empty
    screening transitions give epsilon=I (bare interaction). Real symmetric
    MO factors and positive screening gaps are required. Invalid numerical
    inputs retain diagnostics with valid=False; there is no gap clipping model.
    A GMRES LinearSolverConfig stores only transition factors and gaps, never
    the dense dielectric. Direct screening is the default bounded reference.
    """
    e, l = jnp.asarray(mo_energy), jnp.asarray(mo_factors)
    unrestricted = e.ndim == 2
    if unrestricted:
        if e.shape[0] != 2 or l.ndim != 4 or l.shape[0] != 2:
            raise ValueError("Spin screening requires two energy/factor channels")
        if len(occupied) != 2 or len(virtual) != 2:
            raise ValueError("Spin screening requires alpha/beta windows")
        energies, factors = e, l
        occs, virs = occupied, virtual
    else:
        energies, factors = (e,), (l,)
        occs, virs = (occupied,), (virtual,)
    if jnp.iscomplexobj(e) or jnp.iscomplexobj(l):
        raise NotImplementedError("Static molecular screening requires real inputs")
    if (not isfinite(gap_tol) or gap_tol <= 0
            or not isinstance(max_aux, Integral) or max_aux < 1):
        raise ValueError("Invalid static screening tolerance or auxiliary limit")
    if config is not None and not isinstance(config, LinearSolverConfig):
        raise TypeError("screening config must be a LinearSolverConfig")
    cfg = config or LinearSolverConfig(method="direct", rtol=1e-11, atol=1e-13,
                                       max_dense=max_aux)
    dtype = jnp.result_type(e, l, 1.0)
    valid, minimum = jnp.asarray(True), jnp.asarray(jnp.inf, dtype)
    vertices, weights = [], []
    for energy, factor, occupied, virtual in zip(energies, factors, occs, virs):
        if energy.ndim != 1 or factor.ndim != 3 or factor.shape[1:] != (energy.size,) * 2:
            raise ValueError("Screening requires energy (nmo,) and factors (naux,nmo,nmo)")
        naux = factor.shape[0]
        if naux > max_aux:
            raise ValueError("Static screening exceeds max_aux")
        if cfg.method == "direct" and naux > cfg.max_dense:
            raise ValueError("Static screening direct solve exceeds max_dense")
        occ, vir = tuple(occupied), tuple(virtual)
        if any(not isinstance(p, Integral) or not 0 <= p < energy.size for p in occ + vir
               ) or len(set(occ + vir)) != len(occ) + len(vir):
            raise ValueError("Screening indices must be distinct, disjoint and in range")
        energy, factor = energy.astype(dtype), factor.astype(dtype)
        oi, va = jnp.asarray(occ, dtype=jnp.int32), jnp.asarray(vir, dtype=jnp.int32)
        gaps = energy[va][None, :] - energy[oi][:, None]
        minimum = jnp.minimum(minimum, jnp.min(gaps, initial=jnp.inf))
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(factor), initial=0.0))
        valid &= (jnp.all(jnp.isfinite(factor)) & jnp.all(jnp.isfinite(gaps))
                  & jnp.all(gaps > gap_tol)
                  & (jnp.max(jnp.abs(factor - factor.swapaxes(1, 2)), initial=0.)
                     <= 64 * jnp.finfo(dtype).eps * scale))
        lov = factor[:, oi[:, None], va[None, :]].reshape(naux, len(occ) * len(vir))
        vertices.append(lov)
        # StaticScreening uses coefficient 4: each unrestricted spin contributes 2.
        weights.append(((0.5 if unrestricted else 1.) /
                        jnp.where(gaps > gap_tol, gaps, 1.)).reshape(-1))
    lov = jnp.concatenate(vertices, axis=1)
    inverse_gaps = jnp.concatenate(weights)
    if cfg.method == "gmres":
        return StaticScreening(None, valid, minimum, lov, inverse_gaps, cfg)
    dielectric = jnp.eye(lov.shape[0], dtype=dtype) + 4 * (lov * inverse_gaps) @ lov.T
    return StaticScreening(dielectric, valid, minimum, config=cfg,
                           cholesky=jnp.linalg.cholesky(dielectric))


def solve_static_screening(state, values):
    """Return the checked full-W solve and residual diagnostics.

    Direct mode reuses the state's dense Cholesky factor; GMRES processes
    columns sequentially with the shared implicit primal/transpose rules.
    The solution alone remains a linear action in RHS for kernel transposes.
    Input validity and true solve residuals are included in converged/status.
    """
    values = jnp.asarray(values)
    naux = state.naux
    if values.ndim < 1 or values.shape[0] != naux:
        raise ValueError("Screening RHS must have leading dimension naux")
    columns = 1
    for n in values.shape[1:]:
        columns *= n
    rhs = values.reshape(naux, columns)
    cfg = state.config or LinearSolverConfig(
        method="direct", rtol=1e-11, atol=1e-13, max_dense=max(1, naux)
    )
    op = state.operator()
    # Use the physical residual metric. Left diagonal preconditioning can
    # satisfy JAX's stopping test before the unpreconditioned residual passes.
    out = solve_linear(op, rhs, config=cfg, cholesky=state.cholesky)
    valid = state.valid & out.converged
    solution = require_converged_derivative(out.solution, state.valid)
    solution = jnp.where(state.valid, solution, jnp.nan).reshape(values.shape)
    return out._replace(solution=solution, converged=valid, status=jnp.where(valid, 0, 1))


def apply_static_screening(state, values):
    """Apply epsilon^-1 with physical input and primal/AD validity guards."""
    out = solve_static_screening(state, values)
    valid = state.valid & out.converged
    result = require_converged_derivative(out.solution, valid)
    return jnp.where(valid, result, jnp.nan)


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
           "StaticScreening", "build_static_screening", "apply_static_screening",
           "solve_static_screening"]


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
