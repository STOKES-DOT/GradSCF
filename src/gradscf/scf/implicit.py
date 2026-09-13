from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres as jax_gmres
from jaxtyping import Array, PyTree


@dataclass(frozen=True)
class ImplicitFixedPointConfig:
    """Controls the adjoint solve for an implicit fixed-point state."""

    tolerance: float = 1e-6
    max_iter: int = 6
    restart: int | None = 20
    regularization: float = 0.0


def implicit_fixed_point_solution(
    params: PyTree,
    *,
    solution: Array,
    fixed_point: Callable[..., Array],
    fixed_point_args: PyTree | None = None,
    config: ImplicitFixedPointConfig | None = None,
    apply_fixed_point_transpose: Callable[..., Array] | None = None,
    apply_fixed_point_transpose_factory: Callable[..., Callable[[Array], Array]] | None = None,
    params_vjp_from_adjoint: Callable[..., PyTree] | None = None,
    callback_aux: PyTree | None = None,
    converged: Array | bool = True,
    require_converged: bool = False,
) -> Array:
    """Attach differentiable implicit response to a primal fixed-point solution.

    The optimality condition is `fixed_point(solution, params) - solution = 0`.
    The primal `solution` is supplied by the caller, usually from a normal SCF
    loop. Backward solves the transposed fixed-point linear system instead of
    differentiating through that loop. The default root rule supports JVP/VJP
    composition, retaining both the solution and linear-solve response at higher
    orders. Differentiable ``fixed_point_args`` also receive derivatives.
    Failed response solves return NaNs;
    ``require_converged`` applies the same policy to unconverged primal states.
    """

    cfg = ImplicitFixedPointConfig() if config is None else config
    primal_solution = jax.lax.stop_gradient(jnp.asarray(solution))
    has_fixed_point_args = fixed_point_args is not None
    fixed_point_args_tree = () if fixed_point_args is None else fixed_point_args

    @jax.custom_vjp
    def _solution_from_params(params_local: PyTree, solution_local: Array, args_local: PyTree, converged_local: Array) -> Array:
        del params_local, args_local, converged_local
        return solution_local

    def _call_fixed_point(solution_value: Array, params_value: PyTree, args_value: PyTree) -> Array:
        if has_fixed_point_args:
            return fixed_point(solution_value, params_value, args_value)
        return fixed_point(solution_value, params_value)

    def _call_with_optional_aux(fn: Callable[..., Any], *args: Any) -> Any:
        if callback_aux is None:
            return fn(*args)
        return fn(*args, callback_aux)

    def _root(params_local, solution_local, args_local, converged_local):
        def residual(value):
            return _call_fixed_point(value, params_local, args_local) - value

        def tangent_solve(matvec, rhs):
            shape = rhs.shape
            regularization = jnp.asarray(max(float(cfg.regularization),0.), dtype=rhs.dtype)
            operator = lambda v: matvec(v.reshape(shape)).reshape(-1)-regularization*v
            return solve_implicit_linear_system(operator, rhs.reshape(-1),
                tol=cfg.tolerance, max_iter=cfg.max_iter, restart=cfg.restart,
                converged=converged_local if require_converged else True).reshape(shape)

        # JAX's root JVP rebinds the root primitive when linearizing it. This
        # preserves the implicit state dependence under grad-of-grad while the
        # arbitrary forward solver/initial guess remain outside the AD path.
        return jax.lax.custom_root(residual, solution_local,
            solve=lambda _function, _initial: solution_local, tangent_solve=tangent_solve)

    callbacks = (apply_fixed_point_transpose, apply_fixed_point_transpose_factory,
                 params_vjp_from_adjoint)
    if all(callback is None for callback in callbacks):
        return _root(params, primal_solution, fixed_point_args_tree, jnp.asarray(converged))

    def _fwd(
        params_local: PyTree,
        solution_local: Array,
        args_local: PyTree,
        converged_local: Array,
    ) -> tuple[Array, tuple[PyTree, Array, PyTree, Array]]:
        # Legacy optimized VJP hooks still use the differentiable root as their
        # saved state, so differentiable hooks can themselves be differentiated.
        state = _root(params_local, solution_local, args_local, converged_local)
        return state, (params_local, state, args_local, converged_local)

    def _bwd(
        res: tuple[PyTree, Array, PyTree, Array],
        cotangent_solution: Array,
    ) -> tuple[PyTree, Array, PyTree, None]:
        params_local, solution_local, args_local, converged_local = res
        rhs = jnp.asarray(cotangent_solution)

        if apply_fixed_point_transpose_factory is not None:
            fixed_point_transpose = apply_fixed_point_transpose_factory(
                solution_local,
                params_local,
            )
        elif apply_fixed_point_transpose is not None:
            fixed_point_transpose = lambda vec: _call_with_optional_aux(
                apply_fixed_point_transpose,
                solution_local,
                params_local,
                vec,
            )
        else:
            _, solution_vjp = jax.vjp(
                lambda solution_var: _call_fixed_point(solution_var, params_local, args_local),
                solution_local,
            )
            fixed_point_transpose = lambda vec: solution_vjp(vec)[0]

        def _optimality_transpose(vec: Array) -> Array:
            return fixed_point_transpose(vec) - vec

        regularization = jnp.asarray(
            max(float(cfg.regularization), 0.0),
            dtype=solution_local.dtype,
        )

        def _adjoint_op(vec_flat: Array) -> Array:
            vec = vec_flat.reshape(solution_local.shape)
            return (_optimality_transpose(vec) - regularization * vec).reshape(-1)

        lambda_flat = solve_implicit_linear_system(
            _adjoint_op,
            -rhs.reshape(-1),
            tol=cfg.tolerance,
            max_iter=cfg.max_iter,
            restart=cfg.restart,
        )
        adjoint = lambda_flat.reshape(solution_local.shape)

        if params_vjp_from_adjoint is not None:
            grad_params = _call_with_optional_aux(
                params_vjp_from_adjoint,
                solution_local,
                params_local,
                adjoint,
            )
        else:
            _, params_vjp = jax.vjp(
                lambda params_var: _call_fixed_point(solution_local, params_var, args_local),
                params_local,
            )
            grad_params = params_vjp(adjoint)[0]

        if has_fixed_point_args:
            _, args_vjp = jax.vjp(
                lambda args_var: _call_fixed_point(solution_local, params_local, args_var),
                args_local,
            )
            grad_args = args_vjp(adjoint)[0]
        else:
            grad_args = None

        valid = jnp.all(jnp.isfinite(adjoint))
        if require_converged:
            valid = jnp.logical_and(valid, converged_local)

        def _checked_gradient(value: Array) -> Array:
            # Integer/static PyTree leaves have float0 cotangents.
            if value.dtype == jax.dtypes.float0:
                return value
            return jnp.where(valid, value, jnp.full_like(value, jnp.nan))

        return (
            jax.tree_util.tree_map(_checked_gradient, grad_params),
            jnp.zeros_like(solution_local),
            jax.tree_util.tree_map(_checked_gradient, grad_args),
            None,
        )

    _solution_from_params.defvjp(_fwd, _bwd)
    return _solution_from_params(params, primal_solution, fixed_point_args_tree, jnp.asarray(converged))


def solve_implicit_linear_system(
    matvec: Callable[[Array], Array],
    b_flat: Array,
    *,
    tol: float,
    max_iter: int,
    restart: int | None = None,
    converged: Array | bool = True,
) -> Array:
    """Solve an adjoint system, returning NaNs if its true residual fails.

    GMRES status alone does not certify convergence on all JAX versions. Check
    the actual operator residual, allowing only a dtype-sized roundoff floor.
    """
    if b_flat.size == 0:
        return jnp.zeros_like(b_flat)
    def checked_solve(operator, rhs):
        # GMRES breakdown tests can classify a machine-sized RHS as zero even
        # with atol=0. Normalize inside the opaque solve, then restore scale.
        scale = jnp.max(jnp.abs(rhs))
        def nonzero(_):
            unit, _ = jax_gmres(operator, rhs/scale, tol=float(tol), atol=0.0,
                restart=20 if restart is None else max(1,int(restart)),
                maxiter=max(1,int(max_iter)), solve_method="incremental")
            return unit*scale
        sol = jax.lax.cond(scale>0,nonzero,lambda _:jnp.zeros_like(rhs),operand=None)
        applied = operator(sol)
        rhs_norm = jnp.linalg.norm(rhs)
        roundoff = 32*jnp.finfo(rhs.dtype).eps*(rhs_norm+jnp.linalg.norm(applied))
        valid = (jnp.all(jnp.isfinite(sol)) & jnp.asarray(converged)
                 & (jnp.linalg.norm(applied-rhs) <= float(tol)*rhs_norm+roundoff))
        return jnp.where(valid, sol, jnp.full_like(sol,jnp.nan))

    # Keep convergence tests inside the opaque solve. A nonlinear validity
    # predicate outside a linear JVP cannot be transposed correctly. The custom
    # linear solve also retains derivatives of both the operator and its RHS.
    return jax.lax.custom_linear_solve(matvec,b_flat,solve=checked_solve,
                                       transpose_solve=checked_solve)
