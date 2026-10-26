"""Explicit finite-temperature matrix inversion and local-potential projection.

This is an auxiliary ensemble reference, not zero-temperature Wu–Yang or a
self-consistent inverse-GKS solver. Matching the full 1-RDM is stronger than
matching its real-space diagonal. All method/temperature choices are explicit.
"""
from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from gradscf._memory import tree_storage_bytes, workspace_bytes
from gradscf.solvers import LinearOperator, LinearSolverConfig, EigenSolverConfig, solve_linear, solve_hermitian
from gradscf.solvers.eigen.spectral import inverse_sqrt, matrix_function
from gradscf.pbc.occupations import fermi_density
from .types import _check


def metric(target):
    s = target.overlap
    if target.layout == 'gks': s = jnp.kron(jnp.eye(2, dtype=s.dtype), s)
    x = inverse_sqrt(s, 1e-12)
    return s, x, s@x


def orthogonal_gauge(matrix, x):
    """Metric-whitened matrix with its one scalar constant per block removed."""
    value = x.conj().T@matrix@x
    shift = jnp.trace(value, axis1=-2, axis2=-1).real/value.shape[-1]
    return value-shift[..., None, None]*jnp.eye(value.shape[-1], dtype=value.dtype)


class InversionResult(NamedTuple):
    target: object
    density: object
    fock: object
    potential_coefficients: object
    temperature: object
    density_error: object
    operator_error: object
    linear_residual: object
    projection_rank: object
    projection_condition: object
    valid: object

    @property
    def method(self):
        return 'thermal_matrix'


def invert(target, potential, base_fock, *, method, temperature,
           regularization=1e-12, tolerance=1e-7, rank_tolerance=1e-12, linear_config=None):
    """Generate a finite-basis local ensemble reference from an interior 1-RDM.

    In an orthonormal frame, H-mu=-T*log(P/(1-P)). Project this operator into
    the supplied local PBS modulo the charge/spin constants, then independently
    reproduce the density with fixed-N Fermi filling. T is in Hartree (kB=1).
    No clipping of target occupations or substitution of a different method.
    The shared solvers handle a column-normalized, rank-reduced projection.
    regularization penalizes coefficients in the normalized column metric;
    rank_tolerance explicitly removes numerical null directions. This is an
    eager offline reference generator; inverse-map AD/JIT is not implemented.
    Model evaluation/training against the returned fixed reference is traceable.
    """
    if method != 'thermal_matrix':
        raise NotImplementedError('This entry implements method=thermal_matrix only; WY is not yet available')
    if any(isinstance(a, jax.core.Tracer) for a in jax.tree.leaves(
            (target, potential, base_fock, temperature, regularization))):
        raise NotImplementedError('Offline reference generation has no inverse-map AD/JIT contract yet')
    if potential.layout != target.layout or potential.overlap.shape != target.overlap.shape:
        raise ValueError('Target and potential must use the same AO/spin layout')
    _check(jnp.allclose(potential.overlap, target.overlap, atol=1e-12, rtol=1e-12),
           'Target and potential overlap matrices differ')
    t, alpha = jnp.asarray(temperature), jnp.asarray(regularization)
    if t.shape != () or alpha.shape != () or tolerance <= 0 or not 0 < rank_tolerance < 1:
        raise ValueError('Supply scalar temperature/regularization and positive tolerance')
    _check(jnp.isfinite(t) & (t > 0), 'A thermal reference requires positive finite temperature')
    _check(jnp.isfinite(alpha) & (alpha >= 0), 'Regularization must be finite and nonnegative')
    s, x, root = metric(target)
    base = jnp.asarray(base_fock)
    if base.shape != target.density.shape:
        raise ValueError('base_fock must have the target spatial/spinor shape')
    _check(jnp.all(jnp.isfinite(base)) & jnp.allclose(base, base.conj().swapaxes(-1, -2)),
           'base_fock must be finite and Hermitian')
    degeneracy = 2 if target.layout == 'rks' else 1
    blocks = target.density if target.layout == 'uks' else target.density[None]
    q = root@blocks@root/degeneracy
    occupations = jnp.linalg.eigvalsh(q)
    margin = 64*jnp.finfo(q.real.dtype).eps
    interior = jnp.all((occupations > margin) & (occupations < 1-margin))
    _check(interior, 'Thermal matrix inversion requires strictly interior occupations; target is not clipped')
    h = jax.vmap(lambda p: matrix_function(p, t,
        lambda values, temp: -temp*(jnp.log(values)-jnp.log1p(-values))))(q)
    desired = root@h@root
    if target.layout != 'uks': desired = desired[0]
    difference = orthogonal_gauge(desired-base, x)
    resident = tree_storage_bytes(target, base, desired, q, root, x)
    available = workspace_bytes(potential.max_memory, resident_bytes=resident)
    # The callback's common reserve leaves no more than this remaining action
    # workspace; integral/grid and solver ownership are conservatively separate.
    action = replace(potential, max_memory=min(potential.max_memory, available/(.8e6)))
    rhs = action.moments(x@difference@x)
    shape = rhs.shape
    def normal(vector):
        value = action.matrix(vector.reshape(shape))
        projected = orthogonal_gauge(value, x)
        return action.moments(x@projected@x).reshape(-1)
    config = (LinearSolverConfig(method='direct', rtol=1e-9, atol=1e-12, max_dense=256)
              if linear_config is None else linear_config)
    if config.method != 'direct' or rhs.size > config.max_dense:
        raise ValueError('thermal_matrix uses a bounded dense potential-space projection')
    workspace_bytes(potential.max_memory,
                    resident_bytes=resident+6*rhs.size**2*rhs.dtype.itemsize)
    # Assemble only the bounded coefficient-space normal matrix. Keeping the
    # checkpointed grid scan outside custom_linear_solve also avoids a scan
    # linear-transpose limitation in the supported local JAX 0.8 runtime.
    normal_matrix = jax.vmap(normal)(jnp.eye(rhs.size, dtype=rhs.dtype)).T
    scales = jnp.sqrt(jnp.maximum(jnp.diag(normal_matrix), 0.))
    scales = jnp.where(scales > 0, scales, 1.)
    scaled = normal_matrix/scales[:, None]/scales[None, :]
    scaled = .5*(scaled+scaled.T)
    factor = solve_hermitian(scaled,
        config=EigenSolverConfig(method='dense', nroots=rhs.size, max_dense=config.max_dense,
                                 atol=1e-10), max_memory=potential.max_memory,
        reserved_bytes=resident)
    values, vectors = factor.values, factor.vectors
    cutoff = rank_tolerance*jnp.maximum(jnp.max(values), jnp.finfo(values.dtype).tiny)
    active = values > cutoff
    diagonal = jnp.where(active, values+alpha, 1.)
    projected_rhs = jnp.where(active, vectors.T@(rhs.reshape(-1)/scales), 0.)
    solved = solve_linear(LinearOperator((rhs.size, rhs.size), rhs.dtype,
        lambda v: diagonal*v, diagonal=diagonal, exact_diagonal=True), projected_rhs,
        config=config, max_memory=potential.max_memory, reserved_bytes=resident)
    coefficients = ((vectors@solved.solution)/scales).reshape(shape)
    fock = base+action.matrix(coefficients)
    fblocks = fock if target.layout == 'uks' else fock[None]
    numbers = (jnp.asarray(target.nelec) if target.layout == 'uks' else
               jnp.asarray([target.nelec/degeneracy]))
    recovered, filling = fermi_density(fblocks[:, None], s, numbers, t, jnp.ones(1))
    density = recovered[:, 0] if target.layout == 'uks' else degeneracy*recovered[0, 0]
    density_error = jnp.max(jnp.abs(root@(density-target.density)@root))
    operator_error = jnp.max(jnp.abs(orthogonal_gauge(fock-desired, x)))
    valid = (target.valid & interior & solved.converged & jnp.all(factor.converged) & filling.converged
             & jnp.isfinite(density_error) & (density_error <= tolerance)
             & jnp.isfinite(operator_error) & (operator_error <= tolerance))
    smallest = jnp.min(jnp.where(active, values, jnp.inf))
    condition = jnp.where(jnp.any(active), jnp.max(values)/smallest, jnp.inf)
    return InversionResult(target, density, fock, coefficients, t, density_error,
                           operator_error, solved.residual_norm, jnp.sum(active), condition, valid)
