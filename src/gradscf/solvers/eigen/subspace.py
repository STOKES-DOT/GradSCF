"""First-order response of an isolated real spectral subspace.

Inspired by Kasim, arXiv:2011.04366, Sections 4-5 (degenerate-space projection
and compatibility conditions). This API differentiates P=XX.T, not individual
states. It uses a whole-subspace Sylvester equation, so internal degeneracy
and splitting perturbations do not require choosing an eigenvector gauge.
See ../DEGENERACY.md for derivation, scope, and boundary diagnostics.
"""
from dataclasses import replace
from math import isfinite

import jax
import jax.numpy as jnp

from ..diagnostics import require_converged_derivative
from ..linear.implicit import checked_linear_solve
from ..operators import as_operator, validate_real_square
from ..types import EigenSolverConfig, LinearSolverConfig, SpectralProjectorResult
from .api import solve_hermitian


def _attach_subspace_response(apply, basis, *, config, valid=True):
    """Private horizontal frame; only basis-invariant outputs may use it.

    Q(AZ-ZH)= -Q(dA)X, X.T Z=0. The PZ term makes the extension to all
    matrices invertible, without changing the physical Q-space equation.
    The live-zero RHS avoids differentiating the Davidson iteration. As with
    the isolated-vector attachment, this is a first-order-only construction.
    """
    x = jax.lax.stop_gradient(basis)
    n, k = x.shape
    ax = apply(x)
    h = jax.lax.stop_gradient(x.T @ ax)
    # Symmetrizing the small Ritz block removes roundoff; input symmetry is
    # checked by solve_hermitian (asserted by matrix-free callers).
    h = .5*(h+h.T)
    project = lambda z: x @ (x.T @ z)
    complement = lambda z: z-project(z)

    def sylvester(vector):
        z = vector.reshape(n, k)
        qz = complement(z)
        physical = (complement(apply(qz)-qz@h)+project(z)).reshape(-1)
        # Invalid boundaries have no promised response. Use an identity only
        # there to keep the zero primal correction finite even for direct solves;
        # response_apply still marks their operator derivatives as invalid.
        return jnp.where(valid, physical, vector)

    rhs = -complement(ax-jax.lax.stop_gradient(ax))
    correction, _ = checked_linear_solve(sylvester, rhs.reshape(-1), config=config)
    return x+correction.reshape(n, k)


def solve_spectral_projector(matrix_or_operator, vectors, *, config=None,
                             linear_config=None, gap_atol=1e-8, gap_rtol=1e-8,
                             initial_vectors=None):
    """Apply the lowest-nroots spectral projector, with JVP and VJP support.

    Internal degeneracies are allowed. The selected subspace must be separated
    from its complement. One additional Ritz root tests the upper boundary;
    an unresolved gap retains primal diagnostics but gives NaN derivatives.
    Increase nroots to include a complete cluster. The response uses no gap
    flooring or artificial level splitting. Fixed nroots keeps JIT shapes static.

    config is an EigenSolverConfig for the primal solver and default adjoint
    controls; gradient_mode does not select the derivative here (this API
    always differentiates the entire subspace). linear_config optionally
    controls the common Sylvester linear solve. vectors is (n,) or (n,m).
    Passing an identity matrix explicitly materializes P for small oracles;
    arbitrary probes avoid an n-by-n projector. Only real symmetric A and a
    fixed Euclidean metric are supported, not RPA or generalized AX=MX Lambda.
    By default Davidson combines diagonal guesses with seed-0 full-support
    probes; initial_vectors can supply a caller-chosen starting block instead.
    """
    cfg = EigenSolverConfig() if config is None else config
    if any(not isfinite(t) or t < 0 for t in (gap_atol, gap_rtol)):
        raise ValueError("Spectral gap tolerances must be finite and nonnegative")
    if gap_atol == gap_rtol == 0:
        raise ValueError("At least one spectral gap tolerance must be positive")
    op = as_operator(matrix_or_operator)
    validate_real_square(op)
    n, k = op.shape[0], cfg.nroots
    if k > n:
        raise ValueError("nroots exceeds operator dimension")
    vectors = jnp.asarray(vectors)
    if vectors.ndim not in (1, 2) or vectors.shape[0] != n:
        raise ValueError("Projector probes must have shape (n,) or (n, nvec)")
    if not jnp.issubdtype(vectors.dtype, jnp.floating):
        raise NotImplementedError("Projector probes must be real floating-point data")
    vectors = vectors.astype(jnp.result_type(vectors.dtype, op.dtype))
    count = min(k+1, n)
    if (cfg.method == 'davidson' and cfg.max_subspace is not None
            and cfg.max_subspace < min(n, count+2)):
        raise ValueError("max_subspace must include nroots plus one boundary root and two expansion slots (or the full dimension)")
    if cfg.method == 'davidson' and initial_vectors is None:
        if op.diagonal is None:
            raise ValueError("Davidson requires an operator diagonal approximation")
        capacity = n if cfg.max_subspace is None else min(n, cfg.max_subspace)
        width = min(capacity, 2*count)
        indices = jnp.argsort(op.diagonal)[:count]
        guesses = jax.nn.one_hot(indices, n, dtype=op.dtype).T
        probes = jax.random.normal(jax.random.PRNGKey(0), (n, width), dtype=op.dtype)/jnp.sqrt(float(n))
        # Mix full-support probes into every diagonal guess, rather than just
        # appending probes: exact higher eigenvectors can otherwise terminate
        # Davidson before a hidden lower invariant sector is explored.
        initial_vectors = probes.at[:, :count].add(guesses)
    primal_cfg = replace(cfg, nroots=count, gradient_mode='eigenvalue_only',
                         max_subspace=cfg.max_subspace if cfg.method == 'davidson' else None)
    roots = jax.tree.map(jax.lax.stop_gradient,
                        solve_hermitian(matrix_or_operator, config=primal_cfg,
                                         initial_vectors=initial_vectors))
    x = roots.vectors[:, :k]
    orth_error = jnp.linalg.norm(roots.vectors.T@roots.vectors-jnp.eye(count, dtype=x.dtype))
    converged = jnp.all(roots.converged) & (orth_error <= 128*jnp.finfo(x.dtype).eps*max(n, 1))
    if k < n:
        gap = roots.values[k]-roots.values[k-1]
        scale = jnp.maximum(jnp.abs(roots.values[k]), jnp.abs(roots.values[k-1]))
        # The residual margin avoids accepting an apparent gap below the
        # uncertainty of the two boundary Ritz values.
        threshold = gap_atol+gap_rtol*scale+roots.residual_norms[k]+roots.residual_norms[k-1]
        valid = converged & (gap > threshold)
    else:
        gap = jnp.asarray(jnp.inf, x.dtype)
        valid = converged

    def response_apply(z):
        # Guard before transposed solves: invalid cotangents applied only after
        # GMRES can otherwise be swallowed by its zero-RHS termination.
        return require_converged_derivative(op.apply(z), valid)

    if k == n:
        projection = vectors
    else:
        response_cfg = linear_config or LinearSolverConfig(
            rtol=cfg.adjoint_tol, maxiter=cfg.adjoint_maxiter, restart=min(40, n*k))
        y = _attach_subspace_response(response_apply, x, config=response_cfg, valid=valid)
        projection = y@(y.T@vectors)
    # tr(A dP)=0 for an invariant subspace. Stopping X here gives the complete
    # first derivative of the trace without an unnecessary response solve.
    trace = jnp.sum(x*response_apply(x))
    projection = require_converged_derivative(projection, valid)
    trace = require_converged_derivative(trace, valid)
    status = jnp.where(~converged, 1, jnp.where(valid, 0, 2))
    return SpectralProjectorResult(projection, trace, roots.residual_norms, gap,
                                    converged, valid, status)
