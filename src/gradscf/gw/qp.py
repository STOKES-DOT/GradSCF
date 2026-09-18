"""Quasiparticle equation solver with implicit differentiation.

The quasiparticle energy e_p^QP is the graphical (Dyson) solution of

    f(w) = w - e_p^mf - [ Re Sigma_p(w) + v^x_pp - v^mf_pp ] = 0

solved by a secant iteration (matching scipy.optimize.newton as used by
PySCF ``gw_cd``).  All requested orbitals are solved **simultaneously** in
one vectorized ``lax.while_loop`` (one compiled XLA program instead of
hundreds of eager dispatches -- this matters for CPU-bound hosts).

Two differentiation modes are provided:

- ``implicit`` (default): the roots are differentiated through the
  implicit function theorem,

      d e_p^QP / d theta = -(df_p/dtheta) / (df_p/dw)

  where df_p/dw = 1 - d(Re Sigma_p)/dw is the inverse quasiparticle
  renormalization factor 1/Z.  Only one extra VJP of Sigma at the
  converged roots is needed; the secant trajectory is not unrolled.

- ``unrolled``: differentiates through every secant iteration via a
  static-length ``lax.scan`` (useful for cross-checking the implicit rule
  on small systems).

The pole-selection mask in the self-energy is piecewise constant, so in
pole-dense regions the secant iteration can oscillate without meeting the
step tolerance; the solver tracks the best (min |f|) iterate and reports
per-orbital convergence flags explicitly -- never a silent fallback.

Failure of the implicit rule (df/dw ~ 0, i.e. Z diverging near satellite
structures) raises an explicit error, per the repository AD policy.

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- M. S. Hybertsen and S. G. Louie, "Electron correlation in semiconductors
  and insulators: Band gaps and quasiparticle energies", Phys. Rev. B 34,
  5390 (1986). DOI:10.1103/PhysRevB.34.5390 (Dyson equation for QP states)
- Implicit differentiation convention follows the project-wide
  ``SCFDifferentiationConfig`` pattern in ``gradscf.scf.autodiff``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .self_energy import sigma_cd

_DFDW_MIN = 1e-10
_TOL = 1e-6
_MAXITER = 100
_STEP_CAP = 0.05
# Memory threshold (bytes) above which the orbital batch is evaluated with
# a sequential lax.map instead of vmap (see gradscf.gw.self_energy).
_BATCH_MEM_LIMIT = 1 << 31


def qp_residual(omega: Array, e_mf: Array, delta_v: Array, ctx: dict) -> Array:
    """Dyson residual f(w) = w - e_mf - Re Sigma_p(w) - delta_v (real)."""
    sigma = sigma_cd(omega, ctx)
    return omega - e_mf - (jnp.real(sigma) + delta_v)


def _ctx_from_parts(shared: dict, wmn_p: Array, b_pm: Array, b_mp: Array) -> dict:
    ctx = {
        "mo_energy": shared["mo_energy"],
        "wmn_p": wmn_p,
        "b_pm": b_pm,
        "b_mp": b_mp,
        "channels": shared["channels"],
        "ef": shared["ef"],
        "eta": shared["eta"],
        "freqs": shared["freqs"],
        "wts": shared["wts"],
        "conjugate": shared.get("conjugate", False),
    }
    if "q0" in shared:
        ctx["q0"] = shared["q0"]
    return ctx


def sigma_cd_batch(omega: Array, shared: dict, stacked: dict) -> Array:
    """Vectorized self-energy Sigma_p(omega) for a batch of orbitals.

    Returns complex ``(norb,)``.  Same stacking conventions as
    :func:`qp_residual_batch`.
    """

    def fn(w, wp, bp, bm, *extra):
        ctx = _ctx_from_parts(shared, wp, bp, bm)
        if extra:
            ctx["del_w"], ctx["p_index"] = extra
            if "q0" in shared:
                ctx["q0"] = {**shared["q0"], "p_index": extra[1]}
        return sigma_cd(w, ctx)

    args = (omega, stacked["wmn_p"], stacked["b_pm"], stacked["b_mp"])
    if "del_w" in stacked:
        args = args + (stacked["del_w"], stacked["p_index"])
    naux = stacked["b_pm"].shape[1]
    est_bytes = omega.shape[0] * naux * naux * 16 * 2
    if est_bytes > _BATCH_MEM_LIMIT:
        return jax.lax.map(lambda a: fn(*a), args)
    return jax.vmap(fn)(*args)


def qp_residual_batch(
    omega: Array, e_mf: Array, delta_v: Array, shared: dict, stacked: dict
) -> Array:
    """Vectorized Dyson residual for a batch of orbitals.

    ``omega``, ``e_mf``, ``delta_v`` have shape ``(norb,)``; ``stacked``
    holds ``wmn_p`` ``(norb, nw, nmo)``, ``b_pm``/``b_mp``
    ``(norb, naux, nmo)``; ``shared`` holds the orbital-independent
    quantities (see :func:`gradscf.gw.self_energy.sigma_cd`).

    Falls back to a sequential ``lax.map`` when the batched residue
    intermediates would exceed ``_BATCH_MEM_LIMIT``.

    When ``stacked`` carries ``del_w`` ``(norb, nw)`` and ``p_index``
    ``(norb,)`` (periodic q -> 0 correction), they are mapped over the
    orbital batch as well.
    """
    has_q0 = "del_w" in stacked

    def fn(w, e, dv, wp, bp, bm, *extra):
        ctx = _ctx_from_parts(shared, wp, bp, bm)
        if extra:
            ctx["del_w"], ctx["p_index"] = extra
            if "q0" in shared:
                ctx["q0"] = {**shared["q0"], "p_index": extra[1]}
        return qp_residual(w, e, dv, ctx)

    args = (omega, e_mf, delta_v, stacked["wmn_p"], stacked["b_pm"], stacked["b_mp"])
    if has_q0:
        args = args + (stacked["del_w"], stacked["p_index"])
    naux = stacked["b_pm"].shape[1]
    est_bytes = omega.shape[0] * naux * naux * 16 * 2
    if est_bytes > _BATCH_MEM_LIMIT:
        return jax.lax.map(lambda a: fn(*a), args)
    return jax.vmap(fn)(*args)


def _df_dw_batch(root, e_mf, delta_v, shared, stacked):
    """Per-orbital df/dw at the roots (inverse Z factors), memory-bounded."""

    def fn(w, e, dv, wp, bp, bm, *extra):
        ctx = _ctx_from_parts(shared, wp, bp, bm)
        if extra:
            ctx["del_w"], ctx["p_index"] = extra
            if "q0" in shared:
                ctx["q0"] = {**shared["q0"], "p_index": extra[1]}
        return jax.grad(lambda ww: qp_residual(ww, e, dv, ctx))(w)

    args = (root, e_mf, delta_v, stacked["wmn_p"], stacked["b_pm"], stacked["b_mp"])
    if "del_w" in stacked:
        args = args + (stacked["del_w"], stacked["p_index"])
    naux = stacked["b_pm"].shape[1]
    est_bytes = root.shape[0] * naux * naux * 16 * 2
    if est_bytes > _BATCH_MEM_LIMIT:
        return jax.lax.map(lambda a: fn(*a), args)
    return jax.vmap(fn)(*args)


def _secant_step(f, x_prev, x, f_prev, fx):
    raw_step = fx * (x - x_prev) / (fx - f_prev)
    step = jnp.clip(raw_step, -_STEP_CAP, _STEP_CAP)
    x_new = x - step
    return x_new, f(x_new)


def _secant_batch(f, x0, x1, *, tol: float, maxiter: int):
    """Vectorized secant solver (while_loop, forward-only).

    Returns (roots, converged_mask); unconverged orbitals return their
    best (min |f|) iterate.
    """

    def cond(state):
        _, _, _, _, i, done, _, _ = state
        return (~jnp.all(done)) & (i < maxiter)

    def body(state):
        x_prev, x, f_prev, fx, i, done, best_x, best_f = state
        x_cand, f_cand = _secant_step(f, x_prev, x, f_prev, fx)
        converged = jnp.abs(x_cand - x) < tol
        take = ~done
        x_new = jnp.where(take, x_cand, x)
        f_new = jnp.where(take, f_cand, fx)
        xp_new = jnp.where(take, x, x_prev)
        fp_new = jnp.where(take, fx, f_prev)
        better = take & (jnp.abs(f_new) < jnp.abs(best_f))
        best_x = jnp.where(better, x_new, best_x)
        best_f = jnp.where(better, f_new, best_f)
        return (xp_new, x_new, fp_new, f_new, i + 1, done | converged, best_x, best_f)

    f0 = f(x0)
    f1 = f(x1)
    better1 = jnp.abs(f1) < jnp.abs(f0)
    state = (
        x0,
        x1,
        f0,
        f1,
        0,
        jnp.zeros_like(x0, dtype=bool),
        jnp.where(better1, x1, x0),
        jnp.where(better1, f1, f0),
    )
    _, root, _, _, _, done, best_x, _ = jax.lax.while_loop(cond, body, state)
    return jnp.where(done, root, best_x), done


def _secant_batch_scan(f, x0, x1, *, tol: float, maxiter: int):
    """Vectorized secant solver (static scan, reverse-mode differentiable)."""
    f0 = f(x0)
    f1 = f(x1)
    better1 = jnp.abs(f1) < jnp.abs(f0)

    def body(carry, _):
        x_prev, x, f_prev, fx, done, best_x, best_f = carry
        x_cand, f_cand = _secant_step(f, x_prev, x, f_prev, fx)
        converged = jnp.abs(x_cand - x) < tol
        take = ~done
        better = take & (jnp.abs(f_cand) < jnp.abs(best_f))
        carry_out = (
            jnp.where(take, x, x_prev),
            jnp.where(take, x_cand, x),
            jnp.where(take, fx, f_prev),
            jnp.where(take, f_cand, fx),
            done | converged,
            jnp.where(better, x_cand, best_x),
            jnp.where(better, f_cand, best_f),
        )
        return carry_out, None

    init = (
        x0,
        x1,
        f0,
        f1,
        jnp.zeros_like(x0, dtype=bool),
        jnp.where(better1, x1, x0),
        jnp.where(better1, f1, f0),
    )
    (_, root, _, _, done, best_x, _), _ = jax.lax.scan(body, init, None, length=int(maxiter))
    return jnp.where(done, root, best_x), done


@jax.custom_vjp
def _qp_solve_implicit(x0: Array, e_mf: Array, delta_v: Array, shared: dict, stacked: dict) -> Array:
    f = lambda w: qp_residual_batch(w, e_mf, delta_v, shared, stacked)
    root, _ = _secant_batch(f, x0, x0 + 1e-4, tol=_TOL, maxiter=_MAXITER)
    return root


def _qp_solve_implicit_fwd(x0, e_mf, delta_v, shared, stacked):
    f = lambda w: qp_residual_batch(w, e_mf, delta_v, shared, stacked)
    root, _ = _secant_batch(f, x0, x0 + 1e-4, tol=_TOL, maxiter=_MAXITER)
    return root, (root, e_mf, delta_v, shared, stacked)


def _qp_solve_implicit_bwd(res, g):
    root, e_mf, delta_v, shared, stacked = res
    # df_p/dw at the converged roots (inverse Z factors), one per orbital.
    df_dw = _df_dw_batch(root, e_mf, delta_v, shared, stacked)
    # Implicit function theorem per orbital; the VJP of the batched
    # residual accumulates shared-parameter cotangents across orbitals.
    _, pullback = jax.vjp(
        lambda em, dv, sh, st: qp_residual_batch(root, em, dv, sh, st), e_mf, delta_v, shared, stacked
    )
    cot_e_mf, cot_delta_v, cot_shared, cot_stacked = pullback(-g / df_dw)
    return (None, cot_e_mf, cot_delta_v, cot_shared, cot_stacked)


_qp_solve_implicit.defvjp(_qp_solve_implicit_fwd, _qp_solve_implicit_bwd)


def solve_qp_batch(
    e_mf: Array,
    delta_v: Array,
    shared: dict,
    stacked: dict,
    *,
    occupied: Array,
    tol: float = 1e-6,
    maxiter: int = 100,
    diff_mode: str = "implicit",
) -> tuple[Array, Array]:
    """Solve the quasiparticle equation for a batch of orbitals.

    Parameters
    ----------
    e_mf, delta_v:
        ``(norb,)`` mean-field energies and ``v^x - v^mf`` corrections.
    shared, stacked:
        Self-energy contexts, see :func:`qp_residual_batch`.
    occupied:
        ``(norb,)`` boolean; selects the initial-guess offset sign
        (-1e-2 occupied, +1e-2 virtual, following PySCF ``gw_cd``).
    tol, maxiter:
        Secant step tolerance and iteration cap.
    diff_mode:
        ``"implicit"`` (custom VJP via the implicit function theorem) or
        ``"unrolled"`` (AD through a static secant scan).

    Returns
    -------
    (qp_energies, converged_mask), both shape ``(norb,)``.  Under a JAX
    transformation the energies are traced and the mask is all-True (the
    caller validates convergence eagerly).
    """
    if diff_mode not in ("implicit", "unrolled"):
        raise ValueError(f"diff_mode must be 'implicit' or 'unrolled', got {diff_mode!r}")
    e_mf = jnp.asarray(e_mf, dtype=jnp.float64)
    delta_v = jnp.asarray(delta_v, dtype=jnp.float64)
    occupied = jnp.asarray(occupied, dtype=bool)
    x0 = e_mf + jnp.where(occupied, -1e-2, 1e-2)
    f = lambda w: qp_residual_batch(w, e_mf, delta_v, shared, stacked)

    if isinstance(e_mf, jax.core.Tracer) or isinstance(delta_v, jax.core.Tracer):
        if diff_mode == "implicit":
            if float(tol) != _TOL or int(maxiter) != _MAXITER:
                raise ValueError(
                    "traced diff_mode='implicit' uses the module defaults "
                    f"tol={_TOL}, maxiter={_MAXITER}; use diff_mode="
                    "'unrolled' for custom solver tolerances."
                )
            root = _qp_solve_implicit(x0, e_mf, delta_v, shared, stacked)
        else:
            root, _ = _secant_batch_scan(f, x0, x0 + 1e-4, tol=float(tol), maxiter=int(maxiter))
        return root, jnp.ones_like(e_mf, dtype=bool)

    if diff_mode == "implicit":
        root, done = _secant_batch(f, x0, x0 + 1e-4, tol=float(tol), maxiter=int(maxiter))
    else:
        root, done = _secant_batch_scan(f, x0, x0 + 1e-4, tol=float(tol), maxiter=int(maxiter))

    # Host-side singularity check per orbital (explicit errors only):
    df_dw = _df_dw_batch(root, e_mf, delta_v, shared, stacked)
    import numpy as _np

    bad = _np.asarray(jax.device_get(df_dw))
    if _np.any(_np.abs(bad) < _DFDW_MIN):
        idx = int(_np.argmin(_np.abs(bad)))
        raise ArithmeticError(
            "QP implicit differentiation is singular: |df/dw| = "
            f"{abs(bad[idx]):.3e} < {_DFDW_MIN} at batch index {idx} "
            "(Z factor diverges; likely a satellite/multiple-root region). "
            "Refusing to return a meaningless gradient."
        )
    return root, done


def solve_qp_orbital(
    e_mf: float,
    delta_v: float,
    ctx: dict,
    *,
    occupied: bool,
    tol: float = 1e-6,
    maxiter: int = 100,
    diff_mode: str = "implicit",
) -> tuple[float, bool]:
    """Single-orbital convenience wrapper around :func:`solve_qp_batch`."""
    shared = {
        "mo_energy": ctx["mo_energy"],
        "channels": ctx["channels"],
        "ef": ctx["ef"],
        "eta": ctx["eta"],
        "freqs": ctx["freqs"],
        "wts": ctx["wts"],
        "conjugate": ctx.get("conjugate", False),
    }
    if "q0" in ctx:
        shared["q0"] = ctx["q0"]
    stacked = {
        "wmn_p": ctx["wmn_p"][None],
        "b_pm": ctx["b_pm"][None],
        "b_mp": ctx["b_mp"][None],
    }
    if "del_w" in ctx:
        stacked["del_w"] = ctx["del_w"][None]
        stacked["p_index"] = jnp.asarray([ctx["p_index"]])
    root, done = solve_qp_batch(
        jnp.asarray([e_mf]),
        jnp.asarray([delta_v]),
        shared,
        stacked,
        occupied=jnp.asarray([occupied]),
        tol=tol,
        maxiter=maxiter,
        diff_mode=diff_mode,
    )
    if isinstance(root, jax.core.Tracer):
        return root[0], True
    return float(root[0]), bool(done[0])


__all__ = ["qp_residual", "qp_residual_batch", "sigma_cd_batch", "solve_qp_batch", "solve_qp_orbital"]
