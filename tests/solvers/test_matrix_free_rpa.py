"""Structured RPA action and metric response against the stable dense oracle."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.solvers import EigenSolverConfig, LinearOperator, solve_stable_rpa


def model(n=12):
    rng = np.random.default_rng(39)
    u = rng.normal(size=(n, 3)) * 0.05
    v = rng.normal(size=(n, 2)) * 0.04
    a = jnp.diag(jnp.linspace(1.0, 3.0, n)) + jnp.asarray(u @ u.T)
    b = jnp.asarray(v @ v.T)
    return a, b


def operator(matrix, max_width=8):
    def apply(x):
        if x.ndim == 2:
            assert x.shape[1] <= max_width, "full physical matrix was requested"
        return matrix @ x

    return LinearOperator(
        matrix.shape, matrix.dtype, apply, diagonal=jnp.diag(matrix), matmat=apply
    )


def solve(a, b, **kw):
    from gradscf.solvers import solve_rpa

    config = EigenSolverConfig(
        nroots=kw.pop("nroots", 2),
        atol=1e-10,
        max_subspace=8,
        gradient_mode=kw.pop("gradient_mode", "implicit_eigenvector"),
        **kw,
    )
    return solve_rpa(operator(a), operator(b), config=config)


def test_matrix_free_forward_metric_and_dense_agreement():
    a, b = model()
    out = jax.jit(solve)(a, b)
    ref = solve_stable_rpa(
        a, b, config=EigenSolverConfig(method="dense", nroots=2, atol=1e-10)
    )
    np.testing.assert_allclose(out.values, ref.values, atol=2e-10, rtol=0)
    np.testing.assert_allclose(out.x.T @ out.x - out.y.T @ out.y, np.eye(2), atol=1e-9)
    np.testing.assert_allclose((out.x + out.y) ** 2, (ref.x + ref.y) ** 2, atol=2e-9)
    assert np.all(out.converged & out.response_valid) and out.stable
    assert not out.stability_certified
    assert np.max(out.stability_residual_norms) < 1e-10


def test_matrix_free_optical_jvp_vjp_match_dense_and_fd():
    a, b = model()
    da, db = b * 0.3, a * 0.01
    probe = jnp.linspace(-0.3, 0.2, a.shape[0])

    def loss(t, dense=False):
        if dense:
            out = solve_stable_rpa(
                a + t * da,
                b + t * db,
                config=EigenSolverConfig(
                    method="dense",
                    nroots=2,
                    atol=1e-10,
                    gradient_mode="implicit_eigenvector",
                ),
            )
        else:
            out = solve(a + t * da, b + t * db)
        return jnp.sum(out.values * (0.2 + ((out.x + out.y).T @ probe) ** 2))

    grad = jax.jit(jax.grad(loss))(0.0)
    np.testing.assert_allclose(grad, jax.grad(lambda t: loss(t, True))(0.0), atol=2e-8)
    np.testing.assert_allclose(jax.jvp(loss, (0.0,), (1.0,))[1], grad, atol=1e-10)
    np.testing.assert_allclose(grad, (loss(1e-4) - loss(-1e-4)) / 2e-4, atol=2e-8)


def test_instability_and_incomplete_iteration_fail_closed():
    a, b = model()
    failed = solve(a, b, maxiter=1)
    assert not np.any(failed.response_valid)
    assert not np.isfinite(
        jax.grad(lambda t: solve(a * t, b, maxiter=1).values.sum())(1.0)
    )
    a = a.at[-1, -1].set(-1.0)
    bad = solve(a, b)
    assert not bad.stable
    assert np.min(bad.stability_margins) < 0
    assert np.isnan(bad.values).all()


def test_degenerate_root_set_and_energy_only_vectors():
    a, b = jnp.diag(jnp.array([1.0, 2.0, 2.0, 4.0])), jnp.zeros((4, 4))
    result = solve(a, b)
    assert not np.any(result.response_valid)
    assert not np.isfinite(jax.grad(lambda t: solve(a * t, b).values.sum())(1.0))
    a, b = model()
    dx = jax.jvp(
        lambda t: solve(a * t, b, gradient_mode="eigenvalue_only").x, (1.0,), (1.0,)
    )[1]
    np.testing.assert_array_equal(dx, 0)


def test_large_operator_never_requests_full_identity():
    from gradscf.solvers import solve_rpa

    n = 300
    diagonal = jnp.linspace(1.0, 4.0, n)
    a = LinearOperator(
        (n, n),
        jnp.float64,
        lambda x: diagonal * x,
        diagonal=diagonal,
        matmat=lambda x: diagonal[:, None] * x,
    )
    b = LinearOperator(
        (n, n),
        jnp.float64,
        lambda x: jnp.zeros_like(x),
        diagonal=jnp.zeros(n),
        matmat=lambda x: jnp.zeros_like(x),
    )
    cfg = EigenSolverConfig(nroots=2, max_subspace=8, atol=1e-9, max_dense=2)
    result = solve_rpa(a, b, config=cfg)
    np.testing.assert_allclose(result.values, [diagonal[0], diagonal[1]], atol=1e-9)
    assert np.all(result.converged)


def test_failed_stability_has_invalid_jvp_and_vjp():
    a, b = model()
    a = a.at[-1, -1].set(-1.0)

    def f(t):
        return solve(a * t, b).values[0]

    assert not np.isfinite(jax.jvp(f, (1.0,), (1.0,))[1])
    assert not np.isfinite(jax.grad(f)(1.0))


def test_forward_and_reverse_jaxpr_have_no_full_physical_matrix():
    from gradscf.solvers import solve_rpa

    n = 300
    diagonal = jnp.linspace(1.0, 4.0, n)
    u = jnp.asarray(np.random.default_rng(41).normal(size=(n, 2)) * 0.002)

    def objective(factor):
        def apply(x):
            assert x.ndim == 1 or x.shape[1] <= 8
            return (diagonal * x if x.ndim == 1 else diagonal[:, None] * x) + factor @ (
                factor.T @ x
            )

        a = LinearOperator(
            (n, n),
            jnp.float64,
            apply,
            diagonal=diagonal + jnp.sum(factor**2, axis=1),
            matmat=apply,
        )
        b = LinearOperator(
            (n, n),
            jnp.float64,
            lambda x: jnp.zeros_like(x),
            diagonal=jnp.zeros(n),
            matmat=lambda x: jnp.zeros_like(x),
        )
        result = solve_rpa(
            a,
            b,
            config=EigenSolverConfig(
                nroots=2, max_subspace=8, gradient_mode="implicit_eigenvector"
            ),
        )
        return result.values.sum() + jnp.sum(result.x[:3] ** 2)

    def visit(value):
        if hasattr(value, "jaxpr"):
            visit(value.jaxpr)
        elif hasattr(value, "eqns"):
            for eqn in value.eqns:
                for var in eqn.outvars:
                    shape = getattr(getattr(var, "aval", None), "shape", ())
                    assert shape not in ((n, n), (2 * n, 2 * n)), (eqn.primitive, shape)
                visit(eqn.params)
        elif isinstance(value, dict):
            for x in value.values():
                visit(x)
        elif isinstance(value, (list, tuple)):
            for x in value:
                visit(x)

    visit(jax.make_jaxpr(objective)(u))
    visit(jax.make_jaxpr(jax.grad(objective))(u))


@pytest.mark.parametrize("seed", range(4))
def test_restart_for_general_spd_blocks(seed):
    from gradscf.solvers import solve_rpa

    rng = np.random.default_rng(seed)
    n = 10
    u, v = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    minus = np.eye(n) + 0.08 * u @ u.T
    plus = np.eye(n) + 0.08 * v @ v.T
    a, b = jnp.asarray((minus + plus) / 2), jnp.asarray((plus - minus) / 2)
    cfg = EigenSolverConfig(
        nroots=2,
        max_subspace=8,
        maxiter=200,
        atol=1e-9,
        gradient_mode="implicit_eigenvector",
    )
    out = solve_rpa(operator(a), operator(b), config=cfg)
    ref = solve_stable_rpa(a, b, config=EigenSolverConfig(method="dense", nroots=2))
    assert np.all(out.response_valid), out.residual_norms
    np.testing.assert_allclose(out.values, ref.values, atol=2e-9, rtol=0)


def test_large_coupled_optical_adjoint_converges():
    from gradscf.solvers import solve_rpa

    n = 300
    d = jnp.linspace(1.0, 4.0, n)
    u = jnp.asarray(np.random.default_rng(41).normal(size=(n, 2)) * 0.002)
    cfg = EigenSolverConfig(
        nroots=2, max_subspace=8, atol=1e-9, gradient_mode="implicit_eigenvector"
    )

    def f(t):
        factor = u * t

        def action(x):
            return (d * x if x.ndim == 1 else d[:, None] * x) + factor @ (factor.T @ x)

        a = LinearOperator(
            (n, n),
            jnp.float64,
            action,
            diagonal=d + jnp.sum(factor**2, axis=1),
            matmat=action,
        )
        b = LinearOperator(
            (n, n),
            jnp.float64,
            lambda x: 0.05 * x,
            diagonal=jnp.full((n,), 0.05),
            matmat=lambda x: 0.05 * x,
        )
        out = solve_rpa(a, b, config=cfg)
        return out.values.sum() + jnp.sum(out.x[:3] ** 2)

    value, grad = jax.jit(jax.value_and_grad(f))(1.0)
    assert np.isfinite(value) and np.isfinite(grad)
    evaluate = jax.jit(f)
    fd = (evaluate(1.0001) - evaluate(0.9999)) / 0.0002
    np.testing.assert_allclose(grad, fd, atol=2e-8, rtol=2e-5)
