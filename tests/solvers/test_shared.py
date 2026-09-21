"""Shared solver contracts: real symmetric eigenproblems and real linear systems."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_shared_namespace():
    import gradscf
    assert "solvers" in dir(gradscf)


@pytest.mark.parametrize("method", ["gmres", "direct"])
def test_linear_nonsymmetric_jit_transpose_and_second_derivative(method):
    from gradscf.solvers import LinearSolverConfig, solve_linear
    config = LinearSolverConfig(method=method, rtol=1e-11, maxiter=10, restart=3)
    rhs = jnp.array([.5, -.2, .8])
    def matrix(t):
        return jnp.array([[2.+t, .4, -.1], [-.2, 1.7, .3], [.1, -.3, 1.1]])
    def loss(t):
        return jnp.sum(solve_linear(matrix(t), rhs, config=config).solution**2)
    oracle = lambda t: jnp.sum(jnp.linalg.solve(matrix(t), rhs)**2)
    result = jax.jit(lambda a,b: solve_linear(a,b,config=config))(matrix(.2),rhs)
    assert bool(result.converged)
    np.testing.assert_allclose(result.solution, jnp.linalg.solve(matrix(.2),rhs), atol=1e-11)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(.2),jax.grad(oracle)(.2),atol=1e-10)
    np.testing.assert_allclose(jax.jit(jax.hessian(loss))(.2),jax.hessian(oracle)(.2),atol=1e-9)


def test_operator_column_layout_and_explicit_transpose():
    from gradscf.solvers import LinearOperator, solve_linear
    a = jnp.array([[2., .7], [-.1, 1.]])
    op = LinearOperator(shape=(2,2), dtype=a.dtype, matvec=lambda v:a@v,
                        transpose_matvec=lambda v:a.T@v, diagonal=jnp.diag(a))
    np.testing.assert_allclose(op.apply(jnp.eye(2)), a)
    np.testing.assert_allclose(solve_linear(op.T, jnp.ones(2)).solution,
                               jnp.linalg.solve(a.T,jnp.ones(2)),atol=1e-9)
    with pytest.raises(ValueError, match="shape"):
        op.apply(jnp.ones(3))


def test_linear_zero_tiny_rhs_and_failed_solve():
    from gradscf.solvers import LinearOperator, LinearSolverConfig, solve_linear
    a = jnp.array([[1.65,.54],[.54,1.65]])
    op = LinearOperator((2,2), a.dtype, lambda v:a@v)
    cfg = LinearSolverConfig(rtol=1e-11, maxiter=10,restart=2)
    fn = lambda b: solve_linear(op,b,config=cfg).solution
    b = jnp.array([1e-16,1e-16])
    np.testing.assert_allclose(fn(b), jnp.linalg.solve(a,b), atol=1e-28,rtol=0)
    np.testing.assert_allclose(jax.jacfwd(fn)(jnp.zeros(2)),jnp.linalg.inv(a),atol=1e-11)
    bad = solve_linear(jnp.zeros((2,2)), jnp.ones(2), config=cfg)
    assert not bool(bad.converged)
    assert np.all(np.isnan(bad.solution))


@pytest.mark.parametrize("method", ["dense", "davidson"])
def test_hermitian_root_and_vector_response(method):
    from gradscf.solvers import EigenSolverConfig, LinearOperator, solve_hermitian
    a = jnp.array([[.8,.07,.01],[.07,1.3,.04],[.01,.04,1.9]])
    p = jnp.array([.3,-.5,.7])
    cfg = EigenSolverConfig(method=method,nroots=2,atol=1e-10,
                            gradient_mode="implicit_eigenvector",adjoint_tol=1e-11)
    def solve(x):
        return solve_hermitian(LinearOperator(x.shape,x.dtype,lambda v:x@v,
                               diagonal=jnp.diag(x)), config=cfg)
    def loss(t):
        result = solve(a+t*jnp.outer(p,p))
        return result.values[0]+.2*(p@result.vectors[:,0])**2
    def oracle(t):
        w,v = jnp.linalg.eigh(a+t*jnp.outer(p,p))
        return w[0]+.2*(p@v[:,0])**2
    result = jax.jit(solve)(a)
    assert np.all(result.converged)
    assert result.residual_norms.shape == (2,)
    np.testing.assert_allclose(result.values,jnp.linalg.eigvalsh(a)[:2],atol=1e-11)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.),jax.grad(oracle)(0.),atol=1e-9)
    np.testing.assert_allclose(jax.jvp(loss,(0.,),(1.,))[1],jax.grad(oracle)(0.),atol=1e-9)


def test_failed_eigenproblem_derivatives_are_invalid():
    from gradscf.solvers import EigenSolverConfig,solve_hermitian
    a = jnp.array([[.8,.07,.01],[.07,1.3,.04],[.01,.04,1.9]])
    cfg = EigenSolverConfig(maxiter=1,atol=1e-14,gradient_mode="implicit_eigenvector")
    fn = lambda t: solve_hermitian(a+t*jnp.ones_like(a),config=cfg)
    assert not np.all(fn(0.).converged)
    assert not np.isfinite(jax.grad(lambda t:fn(t).values[0])(0.))
    assert not np.isfinite(jax.grad(lambda t:fn(t).vectors[0,0]**2)(0.))


def test_public_eigen_does_not_silently_symmetrize_invalid_inputs():
    from gradscf.solvers import solve_hermitian
    bad = solve_hermitian(jnp.array([[1.,.9],[0.,2.]]))
    assert not np.all(bad.converged)


def test_solver_layer_has_no_electronic_structure_or_private_jax_dependencies():
    import ast
    from pathlib import Path
    import gradscf.solvers as solvers
    root = Path(solvers.__file__).parent
    for path in root.rglob('*.py'):
        source = path.read_text()
        assert 'jax._src' not in source, path
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(part in {'ci','tddft','scf','gw','pbc'}
                               for part in (node.module or '').split('.')), path


def test_failed_roots_use_shared_invalid_derivative_policy():
    from gradscf.solvers.eigen.davidson import implicit_differential_davidson_lowest_symmetric
    from gradscf.solvers.eigen.response import (
        implicit_differential_davidson_lowest_symmetric_with_eigenvectors,
    )
    a = jnp.array([[.8,.07,.01],[.07,1.3,.04],[.01,.04,1.9]])
    def energy(t):
        return implicit_differential_davidson_lowest_symmetric(
            a+t*jnp.ones_like(a), nroots=1, tol=1e-14, max_iter=1)[0][0]
    def weight(t):
        return implicit_differential_davidson_lowest_symmetric_with_eigenvectors(
            a+t*jnp.ones_like(a), nroots=1, tol=1e-14, max_iter=1)[1][0,0]**2
    assert np.isfinite(energy(0.))
    assert not np.isfinite(jax.grad(energy)(0.))
    assert not np.isfinite(jax.grad(weight)(0.))


def test_linear_preconditioned_vmap_and_shape_boundaries():
    from gradscf.solvers import LinearSolverConfig, LinearOperator, solve_linear, solve_hermitian
    a = jnp.array([[2.,.4],[-.1,1.]])
    cfg = LinearSolverConfig(rtol=1e-11)
    solve = lambda b: solve_linear(a,b,config=cfg,preconditioner=lambda v:v/jnp.diag(a))
    rhs = jnp.array([[1.,2.],[-.5,.7]])
    results = jax.jit(jax.vmap(solve))(rhs)
    np.testing.assert_allclose(results.solution,jnp.linalg.solve(a,rhs.T).T,atol=1e-11)
    assert np.all(results.converged)
    with pytest.raises(ValueError, match="square"):
        solve_linear(jnp.ones((2,3)),jnp.ones(2))
    with pytest.raises(NotImplementedError, match="real"):
        solve_hermitian(a.astype(complex))
    empty = solve_linear(jnp.empty((0,0)),jnp.empty(0))
    assert bool(empty.converged) and empty.solution.shape == (0,)


def test_method_solver_compatibility_files_are_removed():
    """Numerical clients must import the canonical solver modules directly."""
    from pathlib import Path
    import gradscf
    root = Path(gradscf.__file__).parent
    for relative in ('tddft/eigensolvers.py', 'tddft/eigenvector_differentiation.py',
                     'scf/implicit.py', 'scf/diis.py', 'scf/_orbital_solver.py', 'ci/response.py'):
        assert not (root/relative).exists(), relative
    import gradscf.scf as scf
    assert not hasattr(scf, 'implicit_fixed_point_solution')
    assert not hasattr(scf, 'ImplicitFixedPointConfig')


def test_repository_clients_do_not_import_removed_solver_paths():
    import ast
    from importlib.util import resolve_name
    from pathlib import Path
    import gradscf
    package = Path(gradscf.__file__).parent
    removed = {
        'gradscf.tddft.eigensolvers', 'gradscf.tddft.eigenvector_differentiation',
        'gradscf.scf.implicit', 'gradscf.scf.diis',
        'gradscf.scf._orbital_solver', 'gradscf.ci.response',
    }
    repository = package.parent.parent
    for directory in ('src', 'tests', 'tools', 'examples'):
        base = repository / directory
        for path in base.rglob('*.py'):
            namespace = '.'.join(path.relative_to(base).parts[:-1])
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                if isinstance(node, ast.Import):
                    targets = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    if node.level:
                        module = resolve_name('.'*node.level + module, namespace)
                    targets = [module] + [module+'.'+item.name for item in node.names]
                else:
                    continue
                assert removed.isdisjoint(targets), (path, targets)


def test_rpa_forward_and_backward_share_public_owner():
    from gradscf.solvers.eigen.rpa import implicit_differential_davidson_lowest_tdhf as shared
    a = jnp.array([[1.,.06],[.06,1.5]])
    b = jnp.array([[.1,.01],[.01,.12]])
    def energy(t):
        aa = a+t*jnp.eye(2)
        def vind(rows):
            x,y = rows[:,:2],rows[:,2:]
            return jnp.concatenate([x@aa+y@b,-x@b-y@aa],axis=1)
        return shared(vind,nroots=1,size=2,diag=jnp.diag(aa),tol=1e-10)[0][0]
    ad = jax.jit(jax.grad(energy))(0.)
    fd = (energy(1e-4)-energy(-1e-4))/2e-4
    np.testing.assert_allclose(ad,fd,atol=1e-8)


def test_implicit_root_and_scalar_schur_response():
    from gradscf.solvers.nonlinear import attach_root
    from gradscf.solvers.linear import solve_scalar_border
    from gradscf.solvers import LinearSolverConfig
    cfg = LinearSolverConfig(rtol=1e-11,maxiter=10,restart=3)
    a = jnp.array([[2.,.2,.1],[-.1,1.4,.3],[.2,-.1,.8]])
    b = jnp.array([.5,-.7,.9])
    fn = lambda t: solve_scalar_border(lambda v:(a+t*jnp.eye(3))@v,b,config=cfg)
    np.testing.assert_allclose(fn(0.),jnp.linalg.solve(a,b),atol=1e-11)
    np.testing.assert_allclose(jax.jacfwd(fn)(0.),-jnp.linalg.solve(a,jnp.linalg.solve(a,b)),atol=1e-10)
    def root(p):
        seed = jax.lax.stop_gradient(jnp.sqrt(p))
        return attach_root(lambda x:x*x-p,seed)
    np.testing.assert_allclose(jax.jit(jax.hessian(root))(2.),-1/(4*2**1.5),atol=1e-10)


def test_complex_dense_rpa_energy_response():
    from gradscf.solvers.eigen.rpa import solve_dense_rpa
    a = jnp.array([[1.,.05j],[-.05j,1.6]])
    b = jnp.array([[.1,.01j],[.01j,.12]])
    fn = lambda t: solve_dense_rpa(a+t*jnp.eye(2),b,nroots=1,tol=1e-9)[0][0]
    np.testing.assert_allclose(jax.jit(jax.grad(fn))(0.),(fn(1e-4)-fn(-1e-4))/2e-4,atol=1e-8)
