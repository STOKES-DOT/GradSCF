"""Direct fixed-target Wu–Yang assessment; no inverse reference or candidate SCF."""
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp

from gradscf.solvers.eigen.spectral import inverse_sqrt
from .response import density_errors
from .reward import _aufbau
from .potential import Potential
from .types import Target, _check


class WuYangScore(NamedTuple):
    """W, delta W, Es, G and gap are Ha; reward and relative density error are dimensionless.

    Unrequested grid error is NaN. Invalid candidates have NaN reward; the raw
    diagnostics remain available. This is not a pointwise exact-potential error.
    """
    value: object
    delta: object
    reward: object
    spectral: object
    density: object
    fock: object
    gap: object
    variational_gap: object
    density_error: object
    valid: object
    density_l1: object = float('nan')


@jax.tree_util.register_pytree_node_class
class WuYang:
    """Prepare a real RKS/UKS density target for repeated local XC assessment.

    kinetic is T, base_fock is T+Vext+J[Dtarget], and baseline is the complete
    baseline XC potential. score accepts a complete candidate XC matrix, with
    no additional baseline added. assess accepts scalar XC energy(params,D)
    and obtains its AO potential by AD. Traditional/custom/neural callbacks
    use the same interface. A nonlocal GKS operator is not a local WY inverse.

    Preparation is eager. score/assess preserve JIT and model-parameter AD,
    including internal degeneracy of an isolated occupied subspace. They do
    not certify the candidate's eventual SCF solution or functional integrability.
    All AO matrices must use the target's identical basis/order and atomic units.
    """

    def __init__(self, target, kinetic, base_fock, *, baseline, grid=None,
                 energy_scale=1e-3, gap_tolerance=1e-7):
        if not isinstance(target, Target):
            raise TypeError('WuYang requires an iKS Target')
        if target.layout not in ('rks', 'uks'):
            raise NotImplementedError('Direct local Wu-Yang supports real RKS/UKS')
        for value in (energy_scale, gap_tolerance):
            if not isfinite(value) or value <= 0:
                raise ValueError('Energy scale and gap tolerance must be positive and finite')
        _check(target.valid, 'Invalid Wu-Yang target')
        self.target, self.grid = target, grid
        self.energy_scale, self.gap_tolerance = energy_scale, gap_tolerance
        self.x = inverse_sqrt(target.overlap, 1e-12)
        _check(jnp.min(jnp.linalg.eigvalsh(target.overlap)) > 1e-12,
               'Wu-Yang overlap must be positive definite without clipping')
        self.kinetic = self._matrix(kinetic, spatial=True)
        self.base_fock = self._matrix(base_fock)
        for matrix in (self.kinetic, self.base_fock):
            _check(self._valid_matrix(matrix), 'Fixed operators must be finite and symmetric')
        self.counts = (target.nelec//2,) if target.layout == 'rks' else target.nelec
        if not 0 < sum(self.counts):
            raise ValueError('A Wu-Yang target needs electrons')
        self.baseline_potential = self.gauge(self._matrix(baseline))
        self.baseline = self._evaluate(self.baseline_potential)
        _check(self.baseline.valid, 'Baseline must have a finite, isolated occupied subspace')

    def tree_flatten(self):
        children = (self.target, self.kinetic, self.base_fock, self.x,
                    self.baseline_potential, self.baseline, self.grid)
        return children, (self.energy_scale, self.gap_tolerance, self.counts)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        obj = object.__new__(cls)
        names = ('target', 'kinetic', 'base_fock', 'x', 'baseline_potential',
                 'baseline', 'grid', 'energy_scale', 'gap_tolerance', 'counts')
        for name, value in zip(names, (*children, *metadata)):
            setattr(obj, name, value)
        return obj

    def _matrix(self, matrix, *, spatial=False):
        matrix = jnp.asarray(matrix)
        shape = self.target.overlap.shape if spatial else self.target.density.shape
        if matrix.shape != shape or jnp.iscomplexobj(matrix):
            raise ValueError(f'Expected real matrix shape {shape}')
        return matrix

    @staticmethod
    def _valid_matrix(matrix):
        return (jnp.all(jnp.isfinite(matrix)) &
                (jnp.max(jnp.abs(matrix-matrix.swapaxes(-1, -2))) <= 1e-10))

    def gauge(self, potential):
        """Remove the scalar S channel, independently for fixed UKS populations."""
        white = self.x@potential@self.x
        constant = jnp.trace(white, axis1=-2, axis2=-1)/white.shape[-1]
        return potential-constant[..., None, None]*self.target.overlap

    def _evaluate(self, potential):
        fock = self.base_fock+potential
        blocks = fock[None] if self.target.layout == 'rks' else fock
        weight = 2 if self.target.layout == 'rks' else 1
        densities, gaps, valid, energies = [], [], [], []
        for block, count in zip(blocks, self.counts):
            # The validated AO operator is symmetric. Whitening can accumulate
            # antisymmetric roundoff when S is conditioned poorly; enforce its
            # exact mathematical symmetry before the strict shared solver guard.
            orthogonal = self.x@block@self.x
            orthogonal = .5*(orthogonal+orthogonal.T)
            projector, gap, good, energy = _aufbau(orthogonal, count, self.gap_tolerance,
                                                  return_energy=True)
            densities.append(weight*self.x@projector@self.x)
            energies.append(weight*energy)
            gaps.append(gap)
            valid.append(good)
        density = jnp.stack(densities)
        if self.target.layout == 'rks':
            density = density[0]
        trace = lambda a, d: jnp.einsum('...ij,...ji->', a, d).real
        # Hellmann–Feynman sum: first parameter derivatives need no projector
        # response solve. The shared spectral primitive retains higher response.
        spectral = jnp.sum(jnp.stack(energies))
        value = spectral-trace(fock-self.kinetic, self.target.density)
        excess = trace(fock, self.target.density)-spectral
        good = (self._valid_matrix(potential) & jnp.all(jnp.stack(valid)) &
                jnp.isfinite(value) & jnp.all(jnp.isfinite(density)))
        result = WuYangScore(value, jnp.asarray(0.), jnp.asarray(0.), spectral, density,
                             fock, jnp.min(jnp.stack(gaps)), excess, jnp.asarray(jnp.nan), good)
        return self.diagnose(result)

    def diagnose(self, result):
        """Add the grid error to a score from this same target/operator setup."""
        if self.grid is None:
            return result
        errors = density_errors(self.target,result.density,self.grid)
        error = jnp.sqrt(errors.l2_squared)
        valid = result.valid & errors.valid
        return result._replace(density_error=error,density_l1=errors.l1, valid=valid,
                               reward=jnp.where(valid, result.reward, jnp.nan))

    def score(self, potential):
        """One candidate solve; the prepared baseline is reused without global caches."""
        potential = self.gauge(self._matrix(potential))
        result = self._evaluate(potential)
        delta = (result.spectral-self.baseline.spectral-jnp.einsum('...ij,...ji->',
                 potential-self.baseline_potential, self.target.density).real)
        electrons = self.target.nelec if self.target.layout == 'rks' else sum(self.target.nelec)
        valid = result.valid & self.baseline.valid & jnp.isfinite(delta)
        reward = jnp.where(valid, delta/(electrons*self.energy_scale), jnp.nan)
        return result._replace(delta=delta, reward=reward, valid=valid)

    def assess(self, energy_fn, params=None):
        """energy_fn(params,D) is COMPLETE scalar XC energy in Ha, not total energy."""
        energy, potential = jax.value_and_grad(lambda d: energy_fn(params, d))(self.target.density)
        result = self.score(potential)
        valid = result.valid & jnp.isfinite(energy)
        return result._replace(valid=valid, reward=jnp.where(valid, result.reward, jnp.nan))

    def local(self, potential_fn, params=None):
        """Assess an external local field using the existing blocked Potential map.

        potential_fn(params, coordinates) returns (points,) for RKS or
        (2,points) for UKS, in Ha. Its external input provider evaluates fields
        from the FIXED target density. grid must be an AOGrid in the target's
        identical AO representation. No interpolation or Cartesian enlargement
        is inferred. Repeated affine directions can be assembled once by bind.
        """
        if self.grid is None or not hasattr(self.grid, 'reduce'):
            raise ValueError('A local field requires a matching AOGrid provider')
        channels = 1 if self.target.layout == 'rks' else 2
        def basis(p, coordinates):
            value = jnp.asarray(potential_fn(p, coordinates))
            shape = ((coordinates.shape[0],) if channels == 1 else (2, coordinates.shape[0]))
            if value.shape != shape:
                raise ValueError(f'Local potential field requires shape {shape}')
            return value[:, None] if channels == 1 else value.T
        provider = Potential(self.grid, self.target.overlap, basis, channels,
                             params=params, layout=self.target.layout)
        coefficients = jnp.ones(1) if channels == 1 else jnp.eye(2)
        return self.score(provider.matrix(coefficients))


__all__ = ['WuYang', 'WuYangScore']
