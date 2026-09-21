"""Isolated spectral subspaces, including exact internal degeneracy; float64."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def problem():
    a = jnp.diag(jnp.array([1., 1., 3., 5.]))
    b = jnp.array([[.2, .6, .4, .1], [.6, -.3, .2, .3],
                   [.4, .2, .1, -.2], [.1, .3, -.2, .4]])
    probe = jnp.array([.3, .4, .5, .2])
    return a, b, probe


def dense_project(a, rhs, rank):
    _, x = np.linalg.eigh(np.asarray(a))
    return x[:, :rank] @ (x[:, :rank].T @ np.asarray(rhs))


@pytest.mark.parametrize('method', ['dense', 'davidson'])
def test_degenerate_projector_jvp_vjp_and_trace(method):
    from gradscf.solvers import EigenSolverConfig, solve_spectral_projector
    a, b, probe = problem()
    cfg = EigenSolverConfig(method=method, nroots=2, atol=1e-11, adjoint_tol=1e-11)
    def result(t):
        return solve_spectral_projector(a+t*b, probe, config=cfg)
    out = jax.jit(result)(0.)
    assert out.converged and out.response_valid
    assert out.status == 0
    np.testing.assert_allclose(out.projection, dense_project(a, probe, 2), atol=2e-12)
    np.testing.assert_allclose(out.eigenvalue_sum, 2., atol=2e-12)
    np.testing.assert_allclose(out.boundary_gap, 2., atol=2e-12)
    loss = lambda t: probe @ result(t).projection + .3*result(t).eigenvalue_sum
    expected = -.115 + .3*(b[0, 0]+b[1, 1])
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.), expected, atol=2e-10)
    np.testing.assert_allclose(jax.jvp(loss, (0.,), (1.,))[1], expected, atol=2e-10)
    step = 1e-4
    reference = lambda t: probe @ dense_project(a+t*b, probe, 2) + .3*np.linalg.eigvalsh(a+t*b)[:2].sum()
    np.testing.assert_allclose((reference(step)-reference(-step))/(2*step), expected, atol=2e-9)


def test_matrix_free_projector_and_rhs_response():
    from gradscf.solvers import EigenSolverConfig, LinearOperator, solve_spectral_projector
    a, b, probe = problem()
    rhs = jnp.stack([probe, .7*probe+jnp.array([0., 0., .1, 0.])], axis=1)
    cfg = EigenSolverConfig(nroots=2, max_subspace=4, adjoint_tol=1e-11)
    def calculate(t, rhs):
        op = LinearOperator(a.shape, a.dtype, lambda v:(a+t*b)@v,
                             diagonal=jnp.diag(a+t*b))
        return solve_spectral_projector(op, rhs, config=cfg).projection
    np.testing.assert_allclose(jax.jit(calculate)(0., rhs), dense_project(a, rhs, 2), atol=1e-11)
    actual = jax.jacrev(lambda v: calculate(0., v))(probe)
    np.testing.assert_allclose(actual, np.diag([1., 1., 0., 0.]), atol=1e-11)
    direction = jnp.arange(8, dtype=jnp.float64).reshape(4, 2)/10
    _, tangent = jax.jvp(calculate, (0., rhs), (1., direction))
    step = 1e-4
    expected = (dense_project(a+step*b, rhs+step*direction, 2)
                - dense_project(a-step*b, rhs-step*direction, 2))/(2*step)
    np.testing.assert_allclose(tangent, expected, atol=2e-9)


@pytest.mark.parametrize('method', ['dense', 'davidson'])
def test_cut_degenerate_boundary_has_invalid_response(method):
    from gradscf.solvers import EigenSolverConfig, solve_spectral_projector
    a, b, probe = problem()
    cfg = EigenSolverConfig(method=method, nroots=1)
    fn = lambda t: solve_spectral_projector(a+t*b, probe, config=cfg)
    result = fn(0.)
    assert result.converged and not result.response_valid and result.status == 2
    assert np.all(np.isfinite(result.projection))
    assert not np.isfinite(jax.grad(lambda t:jnp.sum(fn(t).projection))(0.))
    assert not np.isfinite(jax.grad(lambda t:fn(t).eigenvalue_sum)(0.))


def test_internal_small_gaps_are_not_regularized():
    from gradscf.solvers import EigenSolverConfig, solve_spectral_projector
    a, b, probe = problem()
    cfg = EigenSolverConfig(nroots=2, adjoint_tol=1e-11)
    for gap in (0., 1e-12, 1e-6, .2):
        base = a.at[1, 1].add(gap)
        fn = lambda t: probe @ solve_spectral_projector(base+t*b, probe, config=cfg).projection
        ad = jax.grad(fn)(0.)
        step = 1e-4
        fd = probe @ (dense_project(base+step*b, probe, 2)-dense_project(base-step*b, probe, 2))/(2*step)
        np.testing.assert_allclose(ad, fd, atol=2e-9)


def test_full_subspace_is_identity_and_trace():
    from gradscf.solvers import EigenSolverConfig, solve_spectral_projector
    a, b, probe = problem()
    cfg = EigenSolverConfig(nroots=4)
    def loss(t):
        out = solve_spectral_projector(a+t*b, probe*(1+t), config=cfg)
        return jnp.sum(out.projection) + out.eigenvalue_sum
    out = solve_spectral_projector(a, probe, config=cfg)
    assert out.response_valid and jnp.isinf(out.boundary_gap)
    np.testing.assert_allclose(out.projection, probe, atol=1e-12)
    np.testing.assert_allclose(jax.grad(loss)(0.), probe.sum()+jnp.trace(b), atol=1e-11)


@pytest.mark.parametrize('angle', [0., .37, 1.51])
def test_private_frame_rotation_does_not_change_observable_response(angle):
    from gradscf.solvers.eigen.subspace import _attach_subspace_response
    from gradscf.solvers import LinearSolverConfig
    a, b, probe = problem()
    cfg = LinearSolverConfig(rtol=1e-11, restart=8)
    c, s = jnp.cos(angle), jnp.sin(angle)
    rotation = jnp.array([[c, -s], [s, c]])
    x = jnp.eye(4)[:, :2]@rotation
    # Also allow a nondiagonal Ritz block for a rotated nondegenerate span.
    for splitting in (0., .2):
        base = a.at[1, 1].add(splitting)
        def loss(t):
            y = _attach_subspace_response(lambda v:(base+t*b)@v, x, config=cfg)
            return jnp.sum((y.T@probe)**2)
        step = 1e-4
        fd = probe@(dense_project(base+step*b, probe, 2)-dense_project(base-step*b, probe, 2))/(2*step)
        np.testing.assert_allclose(jax.grad(loss)(0.), fd, atol=2e-9)


def test_projector_identities_and_vmap():
    from gradscf.solvers import EigenSolverConfig, solve_spectral_projector
    a, b, _ = problem()
    cfg = EigenSolverConfig(nroots=2, adjoint_tol=1e-11)
    fn = lambda t: solve_spectral_projector(a+t*b, jnp.eye(4), config=cfg).projection
    p, dp = jax.jvp(fn, (0.,), (1.,))
    np.testing.assert_allclose(p@p, p, atol=1e-12)
    np.testing.assert_allclose(dp, dp.T, atol=1e-12)
    np.testing.assert_allclose(p@dp@p, 0., atol=1e-12)
    np.testing.assert_allclose(dp@p+p@dp, dp, atol=1e-12)
    np.testing.assert_allclose(jnp.trace(dp), 0., atol=1e-12)
    batch = jax.jit(jax.vmap(fn))(jnp.array([-.01, 0., .01]))
    for t, actual in zip([-.01, 0., .01], batch):
        np.testing.assert_allclose(actual, dense_project(a+t*b, np.eye(4), 2), atol=1e-10)


def test_unconverged_primal_and_response_linear_solve_are_invalid():
    from gradscf.solvers import EigenSolverConfig, LinearSolverConfig, solve_spectral_projector
    rng = np.random.default_rng(4)
    raw = jnp.asarray(rng.normal(size=(7, 7)))
    a = (raw+raw.T)/2
    probe = jnp.asarray(rng.normal(size=7))
    cfg = EigenSolverConfig(nroots=2, maxiter=1, atol=1e-14)
    fn = lambda t: solve_spectral_projector(a+t*jnp.outer(probe, probe), probe, config=cfg)
    out = fn(0.)
    assert not out.converged and not out.response_valid and out.status == 1
    assert not np.isfinite(jax.grad(lambda t:probe@fn(t).projection)(0.))
    assert not np.isfinite(jax.grad(lambda t:fn(t).eigenvalue_sum)(0.))
    good = EigenSolverConfig(method='dense', nroots=2)
    bad_linear = LinearSolverConfig(rtol=1e-14, restart=1, maxiter=1)
    def bad_response(t):
        out = solve_spectral_projector(a+t*jnp.outer(probe, probe), probe,
                                       config=good, linear_config=bad_linear)
        return probe@out.projection
    assert np.isfinite(bad_response(0.))
    assert not np.isfinite(jax.grad(bad_response)(0.))


def test_matrix_free_davidson_never_materializes_a_full_operator_block():
    from gradscf.solvers import EigenSolverConfig, LinearOperator, solve_spectral_projector
    n = 24
    diagonal = jnp.concatenate([jnp.ones(2), jnp.arange(3., n+1.)])
    probe = jnp.linspace(.1, .9, n)
    widths = []
    def build(t):
        def matmat(x):
            widths.append(x.shape[1])
            assert x.shape[1] < n
            return diagonal[:, None]*x+t*probe[:, None]*(probe@x)[None, :]
        return LinearOperator((n, n), diagonal.dtype,
            lambda v:diagonal*v+t*probe*(probe@v),
            diagonal=diagonal+t*probe**2, matmat=matmat)
    cfg = EigenSolverConfig(nroots=2, max_subspace=8, atol=1e-10, adjoint_tol=1e-10)
    fn = lambda t: probe@solve_spectral_projector(build(t), probe, config=cfg).projection
    actual = jax.jit(jax.grad(fn))(0.)
    expected = 2*jnp.sum(probe[:2]**2)*jnp.sum(probe[2:]**2/(1-diagonal[2:]))
    np.testing.assert_allclose(actual, expected, atol=2e-10)
    assert widths and max(widths) <= 8


def test_projector_boundary_tolerances_and_input_validation():
    from gradscf.solvers import EigenSolverConfig, solve_spectral_projector
    a, _, probe = problem()
    cfg = EigenSolverConfig(method='dense', nroots=2)
    close = a.at[2, 2].set(1.+1e-9)
    out = solve_spectral_projector(close, probe, config=cfg)
    assert out.converged and not out.response_valid
    assert solve_spectral_projector(close, probe, config=cfg, gap_atol=1e-12, gap_rtol=0.).response_valid
    with pytest.raises(ValueError, match='boundary root'):
        solve_spectral_projector(a, probe, config=EigenSolverConfig(nroots=2, max_subspace=2))
    with pytest.raises(ValueError, match='shape'):
        solve_spectral_projector(a, jnp.ones(3), config=cfg)
    with pytest.raises(ValueError, match='positive'):
        solve_spectral_projector(a, probe, config=cfg, gap_atol=0., gap_rtol=0.)
    with pytest.raises(NotImplementedError, match='real'):
        solve_spectral_projector(a.astype(jnp.complex128), probe, config=cfg)
    bad = solve_spectral_projector(a.at[0, 2].set(.1), probe, config=cfg)
    assert not bad.converged


def test_rank_one_reduces_to_isolated_root_response():
    from gradscf.solvers import EigenSolverConfig, solve_hermitian, solve_spectral_projector
    a, b, probe = problem()
    a = a.at[0, 0].set(.5)
    cfg = EigenSolverConfig(nroots=1, gradient_mode='implicit_eigenvector', adjoint_tol=1e-11)
    old = lambda t: (probe@solve_hermitian(a+t*b, config=cfg).vectors[:, 0])**2
    new = lambda t: probe@solve_spectral_projector(a+t*b, probe, config=cfg).projection
    np.testing.assert_allclose(new(0.), old(0.), atol=1e-12)
    np.testing.assert_allclose(jax.grad(new)(0.), jax.grad(old)(0.), atol=2e-11)


def test_unconverged_guard_root_invalidates_even_a_converged_selected_root():
    from gradscf.solvers import EigenSolverConfig, solve_spectral_projector
    a = jnp.diag(jnp.array([0., 1., 2., 3.])).at[1, 3].set(.2).at[3, 1].set(.2)
    cfg = EigenSolverConfig(nroots=1, maxiter=1, atol=1e-14)
    out = solve_spectral_projector(a, jnp.ones(4), config=cfg,
                                   initial_vectors=jnp.eye(4)[:, :2])
    assert out.residual_norms[0] == 0
    assert out.residual_norms[1] > cfg.atol
    assert not out.converged and not out.response_valid


def test_invalid_boundary_retains_primal_with_direct_response_solver():
    from gradscf.solvers import EigenSolverConfig, LinearSolverConfig, solve_spectral_projector
    a, b, probe = problem()
    cfg = EigenSolverConfig(method='dense', nroots=1)
    linear = LinearSolverConfig(method='direct')
    def result(t):
        return solve_spectral_projector(a+t*b, probe, config=cfg, linear_config=linear)
    out = result(0.)
    assert out.converged and not out.response_valid
    assert np.all(np.isfinite(out.projection))
    assert not np.isfinite(jax.grad(lambda t:probe@result(t).projection)(0.))


def test_bounded_davidson_restart_with_guard_root():
    from gradscf.solvers import EigenSolverConfig, solve_spectral_projector
    rng = np.random.default_rng(43)
    perturbation = rng.normal(size=(10, 10))*.03
    a = jnp.asarray(np.diag(np.arange(1., 11.))+perturbation+perturbation.T)
    probe = jnp.asarray(rng.normal(size=10))
    out = solve_spectral_projector(a, probe, config=EigenSolverConfig(
        nroots=2, max_subspace=5, maxiter=150, atol=1e-10))
    assert out.converged and out.response_valid
    np.testing.assert_allclose(out.projection, dense_project(a, probe, 2), atol=2e-10)


@pytest.mark.parametrize('size', [6, 20])
def test_default_start_explores_disconnected_invariant_sectors(size):
    from gradscf.solvers import EigenSolverConfig, solve_spectral_projector
    # Lowest diagonal guesses alone miss the hidden eigenvalue zero, returning
    # a false gap above [1,1]. The actual lowest two roots cut a degenerate pair.
    diagonal = jnp.concatenate((jnp.array([1., 1., 3., 5.]), jnp.full((size-4,), 100.)))
    a = jnp.diag(diagonal).at[-1, -2].set(-100.).at[-2, -1].set(-100.)
    out = solve_spectral_projector(a, jnp.ones(size), config=EigenSolverConfig(nroots=2))
    assert out.converged and not out.response_valid
    np.testing.assert_allclose(out.eigenvalue_sum, 1., atol=1e-10)
    np.testing.assert_allclose(out.boundary_gap, 0., atol=1e-10)


def test_davidson_arbitrary_start_on_diagonal_matrix():
    from gradscf.solvers import EigenSolverConfig, solve_hermitian
    a = jnp.diag(jnp.arange(1., 11.))
    initial = jnp.asarray(np.random.default_rng(17).normal(size=(10, 4)))
    out = solve_hermitian(a, config=EigenSolverConfig(nroots=2, max_subspace=5, atol=1e-10),
                          initial_vectors=initial)
    assert np.all(out.converged)
    np.testing.assert_allclose(out.values, [1., 2.], atol=1e-10)
