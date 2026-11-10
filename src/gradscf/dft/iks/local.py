"""Offline zero-temperature local inversions sharing one bounded numerical kernel.

RKS and collinear UKS; real orbitals, fixed integer spin populations, a finite
occupied-virtual gap. The target may be a correlated non-idempotent AO1RDM:
only its density moments are fitted. The regularizer acts on the observable,
metric-whitened AO potential, not on an asserted unique continuum potential.
"""
from dataclasses import dataclass, replace
from math import isfinite

import jax
import jax.numpy as jnp
import numpy as np

from gradscf._memory import tree_storage_bytes, workspace_bytes
from gradscf.solvers import EigenSolverConfig, solve_hermitian
from gradscf.solvers.eigen.spectral import inverse_sqrt, occupied_projector
from gradscf.solvers.nonlinear import minimize_eager
from .orbital import OrbitalReference
from .response import density_norm2
from .types import _check


@dataclass(frozen=True)
class LocalInverseResult:
    target: object
    reference: object
    density: object
    fock: object
    potential_coefficients: object
    constant: object
    objective: float
    gradient_norm: float
    moment_error: float
    raw_moment_error: float
    density_error: float
    relative_density_error: float
    density_metric: str
    gap: float
    projection_rank: int
    iterations: int
    converged: bool
    valid: bool
    optimizer_method: str
    optimizer_evaluations: int
    refinement_evaluations: int

    spectral_converged: bool
    eigen_residual_max: float
    preprojection_symmetry_error: float
    method: str = 'wu_yang'

    @property
    def objective_units(self):
        return 'Hartree' if self.method == 'wu_yang' else 'dimensionless'


def invert(target, potential, base_fock, *, initial=None, regularization=1e-8,
           tolerance=1e-5, gradient_tolerance=1e-8, maxiter=300,
           rank_tolerance=1e-10, gap_tolerance=1e-7, density_metric='absolute_l2',
           objective_kind='wu_yang'):
    """Prepare a fixed orbital reference; do not differentiate through inversion.

    tolerance bounds normalized potential moments and the explicitly selected
    density norm: absolute_l2 (default, bohr^-3/2) or relative_l2 (dimensionless).
    Both raw density norms remain reported. maxiter/gradient_tolerance control
    host optimization. A failed inverse returns diagnostics with reference=None.
    A bounded dense AO potential cache is prepared once, avoiding a grid pass on
    every optimizer evaluation; the provider's memory budget includes it.

    base_fock must be kinetic plus a fixed LOCAL operator; a frozen HF Fock
    retains nonlocal exchange and is not this local KS problem.

    Wu-Yang minimizes minus its potential functional. density_match minimizes
    half the relative squared grid-density error. Both add the declared
    quadratic observable-operator regularizer. A fixed target kinetic-energy offset is
    omitted; objective is therefore NOT the reference kinetic energy T_s.
    """
    if any(isinstance(a, jax.core.Tracer) for a in jax.tree.leaves(
            (target, potential, base_fock, initial, regularization))):
        raise NotImplementedError('Local inversion is offline; inverse-map AD/JIT is unsupported')
    if objective_kind not in ('wu_yang', 'density_match'):
        raise ValueError('Choose a declared local inverse objective')
    if density_metric not in ('absolute_l2', 'relative_l2'):
        raise ValueError('density_metric must be absolute_l2 or relative_l2')
    if target.layout not in ('rks', 'uks'):
        raise NotImplementedError('Wu-Yang currently supports real RKS/UKS; spinor inverse GKS is separate')
    if potential.layout != target.layout or potential.overlap.shape != target.overlap.shape:
        raise ValueError('Target and potential AO/spin layouts differ')
    _check(jnp.allclose(target.overlap, potential.overlap, atol=1e-12, rtol=1e-12),
           'Target and potential overlap matrices differ')
    if not isfinite(regularization) or regularization < 0 or any(
            not isfinite(v) or v <= 0 for v in (tolerance, gradient_tolerance, gap_tolerance)):
        raise ValueError('Supply finite nonnegative regularization and positive tolerances')
    if not 0 < rank_tolerance < 1:
        raise ValueError('rank_tolerance must lie between0 and1')
    s, base = target.overlap, jnp.asarray(base_fock)
    if base.shape != target.density.shape or jnp.iscomplexobj(base):
        raise ValueError('base_fock must be real and match the target density shape')
    symmetry_tolerance = 1e-12+128*jnp.finfo(base.dtype).eps*jnp.maximum(1., jnp.linalg.norm(base))
    _check(jnp.all(jnp.isfinite(base)) & (jnp.max(jnp.abs(base-base.swapaxes(-1, -2))) <= symmetry_tolerance),
           'base_fock must be finite and symmetric')
    _check(jnp.min(jnp.linalg.eigvalsh(s)) > 1e-12, 'Overlap must be positive definite without clipping')
    n, p = s.shape[0], potential.nparam
    config = EigenSolverConfig(method='dense', nroots=p)
    if p > config.max_dense:
        raise ValueError('Wu-Yang potential-space rank analysis exceeds the bounded dense solver')
    counts = (target.nelec//2,) if target.layout == 'rks' else target.nelec
    if any(not 0 < number < n for number in counts):
        raise ValueError('Wu-Yang needs occupied and virtual orbitals in every active spin block')
    degeneracy = 2 if target.layout == 'rks' else 1
    resident = tree_storage_bytes(target, potential, base)
    workspace_bytes(potential.max_memory, resident_bytes=resident+
                    (8*p*n*n+12*p*p+16*n*n)*s.dtype.itemsize)
    x = inverse_sqrt(s, 1e-12)
    root = s@x
    scalar = replace(potential, layout='rks')
    # This consumer intentionally caches a budget-checked small PBS, while the
    # generic Potential continues to evaluate arbitrary large grids by blocks.
    panels = jax.jit(lambda: jax.lax.map(scalar.matrix, jnp.eye(p, dtype=s.dtype)))()
    white = x@panels@x
    white = .5*(white+white.swapaxes(-1,-2))
    centers = jnp.trace(white, axis1=-2, axis2=-1)/n
    centered = white-centers[:, None, None]*jnp.eye(n)
    scales = jnp.linalg.norm(centered.reshape(p, -1), axis=1)
    visible = scales > rank_tolerance*jnp.linalg.norm(white.reshape(p, -1), axis=1)
    scales = jnp.where(visible, scales, 1.)
    normalized = jnp.where(visible[:, None, None], centered/scales[:, None, None], 0.)
    gram = jnp.einsum('pij,qij->pq', normalized, normalized)
    factor = solve_hermitian(gram, config=config, max_memory=potential.max_memory,
                             reserved_bytes=resident+tree_storage_bytes(panels, centered))
    values, vectors = factor.raw_values[:p], factor.raw_vectors[:, :p]
    active = np.asarray(values > rank_tolerance*jnp.maximum(jnp.max(values), 1.))
    rank = int(active.sum())
    if not rank:
        raise ValueError('The potential basis contains no observable direction after gauge reduction')
    transform = vectors[:, active]/(scales[:, None]*jnp.sqrt(values[active])[None, :])
    observed = jnp.einsum('pr,pij->rij', transform, centered)
    qtarget = root@(target.density if target.layout == 'uks' else target.density[None])@root
    bbase = x@(base if target.layout == 'uks' else base[None])@x
    bbase = .5*(bbase+bbase.swapaxes(-1,-2))
    nspin = len(counts)

    density_loss = None
    if objective_kind == 'density_match':
        from .density_matching import grid_density_loss
        density_loss = grid_density_loss(potential.grid, x, qtarget, rank=nspin*rank,
            max_memory=potential.max_memory,
            resident_bytes=resident+tree_storage_bytes(panels, centered, transform, observed))

    def objective(vector):
        angles = vector.reshape(nspin, rank)
        focks = bbase+jnp.einsum('sr,rij->sij', angles, observed)
        states = jnp.stack([degeneracy*occupied_projector(focks[c], count)
                            for c, count in enumerate(counts)])
        physical = (density_loss(states) if density_loss is not None else
                    -jnp.sum(focks*(states-qtarget).swapaxes(-1, -2)))
        return physical+.5*regularization*jnp.sum(angles**2)

    value_grad = jax.jit(jax.value_and_grad(objective))
    seed = jnp.zeros((nspin, rank), dtype=s.dtype)
    if initial is not None:
        supplied = jnp.asarray(initial).reshape(nspin, p)
        if supplied.size != nspin*p or not bool(jnp.all(jnp.isfinite(supplied))):
            raise ValueError('Initial coefficients must be finite and match the potential shape')
        seed = jnp.einsum('sp,pij,rij->sr', supplied, centered, observed)
    # Small reference problems use actual AD curvature. L-BFGS objective-change
    # termination can stop before a tight density/stationarity tolerance.
    hessian = jax.jit(jax.hessian(objective)) if nspin*rank <= 256 else None
    optimized = minimize_eager(value_grad, seed.reshape(-1), maxiter=maxiter,
                               gradient_tolerance=gradient_tolerance, hessian=hessian)
    angles = jnp.asarray(optimized.x).reshape(nspin, rank)
    coefficients = angles@transform.T
    constants = -coefficients@centers
    coefficients = coefficients[0] if target.layout == 'rks' else coefficients
    constants = constants[0] if target.layout == 'rks' else constants
    fock = base+potential.matrix(coefficients, constant=constants)
    raw_blocks = fock if target.layout == 'uks' else fock[None]
    normal_blocks = x@raw_blocks@x
    symmetry_error = float(jnp.max(jnp.linalg.norm(
        normal_blocks-normal_blocks.swapaxes(-1,-2), axis=(-2,-1))))
    # These are analytically Hermitian local operators. Remove assembly and
    # frame-transform roundoff before the shared solver's strict structure audit.
    fock = .5*(fock+fock.swapaxes(-1,-2))
    normal_blocks = .5*(normal_blocks+normal_blocks.swapaxes(-1,-2))
    spectral = [solve_hermitian(f,
        config=EigenSolverConfig(method='dense', nroots=n), max_memory=potential.max_memory,
        reserved_bytes=resident+tree_storage_bytes(panels, observed)) for f in normal_blocks]
    orbitals = jnp.stack([x@r.raw_vectors[:, :n] for r in spectral])
    energies = jnp.stack([r.raw_values[:n] for r in spectral])
    occupations = jnp.stack([degeneracy*(jnp.arange(n) < number) for number in counts])
    gap = min(float(energies[c, number]-energies[c, number-1]) for c, number in enumerate(counts))
    if target.layout == 'rks':
        orbitals, energies, occupations = orbitals[0], energies[0], occupations[0]
    candidate = OrbitalReference(target, orbitals, occupations, energies, method=objective_kind)
    difference = candidate.density-target.density
    moments = potential.moments(difference).reshape(nspin, p)
    raw_moment_error = float(jnp.max(jnp.abs(moments)))
    # Report the residual in the same unit-operator-norm coordinates as the
    # optimizer; raw moments alone change under a harmless PBS rescaling.
    moment_error = float(jnp.linalg.norm(moments@transform))
    density_error = float(jnp.sqrt(density_norm2(candidate, jnp.zeros_like(difference), potential.grid)[1]))
    reference_norm = float(jnp.sqrt(density_norm2(candidate, -candidate.density, potential.grid)[0]))
    if not isfinite(reference_norm) or reference_norm <= 0:
        raise ValueError('Target grid density must have a finite nonzero L2 norm')
    relative_error = density_error/reference_norm
    accepted_density = relative_error if density_metric == 'relative_l2' else density_error
    gradnorm = float(jnp.max(jnp.abs(value_grad(jnp.asarray(optimized.x))[1])))
    converged = bool(gradnorm <= gradient_tolerance and np.isfinite(optimized.fun))
    moment_resolved = moment_error <= tolerance if objective_kind == 'wu_yang' else True
    spectral_converged = all(bool(jnp.all(r.converged)) for r in spectral)
    residual_max = max(float(jnp.max(r.residual_norms)) for r in spectral)
    valid = bool(converged and gap > gap_tolerance and moment_resolved
                 and accepted_density <= tolerance and spectral_converged)
    return LocalInverseResult(
        target=target, reference=candidate if valid else None, density=candidate.density,
        fock=fock, potential_coefficients=coefficients, constant=constants,
        objective=float(optimized.fun), gradient_norm=gradnorm, moment_error=moment_error,
        raw_moment_error=raw_moment_error, density_error=density_error,
        relative_density_error=relative_error, density_metric=density_metric,
        gap=gap, projection_rank=rank, iterations=int(optimized.nit), converged=converged,
        valid=valid, optimizer_method=optimized.method,
        optimizer_evaluations=int(optimized.nfev),
        refinement_evaluations=int(optimized.get('refinement_evaluations', 0)),
        spectral_converged=spectral_converged, eigen_residual_max=residual_max,
        preprojection_symmetry_error=symmetry_error, method=objective_kind)
