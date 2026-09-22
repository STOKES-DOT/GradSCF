"""Checked implicit linear solves, including their transpose and higher response."""
import jax
import jax.numpy as jnp

from ..diagnostics import linear_residual
from ..operators import as_operator, validate_real_square
from ..types import LinearSolverConfig, LinearResult
from .direct import direct_solve
from .gmres import gmres_solve


def checked_linear_solve(matvec, rhs, *, config, converged=True,
                         preconditioner=None, transpose_preconditioner=None):
    """Internal scalar-vector kernel; validity is checked inside the opaque solve.

    Placing a nonlinear residual predicate in a linear tangent map would break
    transposition. Both primal and transposed numerical solves are checked here.
    """
    if rhs.size == 0:
        return rhs, (jnp.asarray(0., dtype=rhs.real.dtype), jnp.asarray(converged))

    if rhs.ndim == 2 and config.method == "direct":
        if rhs.shape[0] > config.max_dense:
            raise ValueError("Direct solve exceeds max_dense; use GMRES")
        matrix = jax.vmap(matvec, in_axes=1, out_axes=1)(jnp.eye(rhs.shape[0], dtype=rhs.dtype))
        def block_solve(transpose):
            a = matrix.T if transpose else matrix
            def solve(operator, value):
                candidate = jnp.linalg.solve(a, value)
                norm, valid = linear_residual(operator, candidate, value, rtol=config.rtol,
                                               atol=config.atol, converged=converged)
                return jnp.where(valid, candidate, jnp.nan), (norm, valid)
            return solve
        return jax.lax.custom_linear_solve(lambda x: matrix @ x, rhs,
            solve=block_solve(False), transpose_solve=block_solve(True), has_aux=True)

    def factory(precond):
        def solve(operator, value):
            if config.method == "direct":
                result = direct_solve(operator, value, max_dense=config.max_dense)
            else:
                result = gmres_solve(operator, value, rtol=config.rtol, atol=config.atol,
                                     maxiter=config.maxiter, restart=config.restart,
                                     preconditioner=precond)
            norm, valid = linear_residual(operator, result, value, rtol=config.rtol,
                                           atol=config.atol, converged=converged)
            return jnp.where(valid, result, jnp.full_like(result, jnp.nan)), (norm, valid)
        return solve

    return jax.lax.custom_linear_solve(matvec, rhs, solve=factory(preconditioner),
        transpose_solve=factory(transpose_preconditioner), has_aux=True)


def solve_linear(matrix_or_operator, rhs, *, config=None, preconditioner=None,
                 transpose_preconditioner=None):
    """Solve a real square system; direct solves also accept a (n,nrhs) block.

    maxiter counts GMRES restart cycles. True residuals certify convergence;
    the upstream GMRES info flag is never treated as a convergence certificate.
    """
    config = LinearSolverConfig() if config is None else config
    op = as_operator(matrix_or_operator)
    validate_real_square(op)
    rhs = jnp.asarray(rhs)
    if not (rhs.shape == (op.shape[0],) or
            (config.method == "direct" and rhs.ndim == 2 and rhs.shape[0] == op.shape[0])):
        raise ValueError("rhs shape must be (n,), or (n,nrhs) for a direct solve")
    if not jnp.issubdtype(rhs.dtype, jnp.floating):
        raise ValueError("rhs must have real floating-point dtype")
    rhs = rhs.astype(jnp.result_type(rhs.dtype, op.dtype))
    solution, (norm, valid) = checked_linear_solve(op.apply, rhs, config=config,
        preconditioner=preconditioner, transpose_preconditioner=transpose_preconditioner)
    return LinearResult(solution, norm, valid, jnp.where(valid, 0, 1))


def solve_implicit_linear_system(matvec, b_flat, *, tol, max_iter, restart=None, converged=True):
    """Compatibility contract used by implicit SCF and scGW responses."""
    config = LinearSolverConfig(rtol=float(tol), maxiter=max(1, int(max_iter)),
                                 restart=20 if restart is None else max(1, int(restart)))
    return checked_linear_solve(matvec, b_flat, config=config, converged=converged)[0]
