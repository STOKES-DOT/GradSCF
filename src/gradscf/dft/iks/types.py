"""Physical density targets with explicit spatial or spinor AO conventions."""
from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp


CHANNELS = {'rks': 1, 'uks': 2, 'gks': 4}


def _layout(layout):
    if layout not in CHANNELS:
        raise ValueError('layout must be rks, uks or gks')
    return CHANNELS[layout]


def _check(condition, message):
    """Reject concrete bad inputs; traced callers retain the explicit flag."""
    if not isinstance(condition, jax.core.Tracer) and not bool(condition):
        raise ValueError(message)


def _components(density, nao, layout):
    density = jnp.asarray(density)
    expected = {'rks': (nao, nao), 'uks': (2, nao, nao), 'gks': (2*nao, 2*nao)}
    _layout(layout)
    if density.shape != expected[layout]:
        raise ValueError(f'{layout} density requires shape {expected[layout]}')
    if layout != 'gks':
        if jnp.iscomplexobj(density):
            raise ValueError('Complex spinor densities require layout=gks')
        return density[None] if layout == 'rks' else density
    aa, ab = density[:nao, :nao], density[:nao, nao:]
    ba, bb = density[nao:, :nao], density[nao:, nao:]
    return jnp.stack(((aa+bb).real, (ab+ba).real, (1j*(ab-ba)).real, (aa-bb).real))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Target:
    """AO density, spatial overlap and fixed electron counts; atomic units.

    RKS density is spin summed, UKS has two spin blocks, GKS is an alpha-then-
    beta spinor matrix. Correlated fractional natural occupations are allowed.
    Construction checks concrete data; ``valid`` also diagnoses traced inputs.
    Optional energy is the reference total energy INCLUDING nuclear repulsion.
    """

    density: object
    overlap: object
    nelec: object
    layout: str = 'rks'
    energy: object = None

    def __post_init__(self):
        _layout(self.layout)
        density, overlap = jnp.asarray(self.density), jnp.asarray(self.overlap)
        if not jnp.issubdtype(density.dtype, jnp.inexact): density = density.astype(float)
        if not jnp.issubdtype(overlap.dtype, jnp.inexact): overlap = overlap.astype(float)
        if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1] or not overlap.shape[0]:
            raise ValueError('overlap must be a nonempty spatial square matrix')
        _components(density, overlap.shape[0], self.layout)
        counts = tuple(self.nelec) if self.layout == 'uks' and isinstance(self.nelec, (tuple, list)) else (self.nelec,)
        if len(counts) != (2 if self.layout == 'uks' else 1) or any(
                isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in counts):
            raise ValueError('nelec requires nonnegative integer counts; UKS requires (nalpha, nbeta)')
        if self.layout == 'rks' and counts[0] % 2:
            raise ValueError('Restricted closed-shell targets require an even electron count')
        energy = None if self.energy is None else jnp.asarray(self.energy)
        if energy is not None and (energy.shape != () or jnp.iscomplexobj(energy)):
            raise ValueError('Reference total energy must be a real scalar')
        for name, value in (('density', density), ('overlap', overlap), ('energy', energy),
                            ('nelec', counts if self.layout == 'uks' else counts[0])):
            object.__setattr__(self, name, value)
        _check(self.valid, 'Invalid target: check finite Hermitian density/overlap, occupations, electron count and energy')

    def tree_flatten(self):
        return (self.density, self.overlap, self.energy), (self.nelec, self.layout)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        # PyTree reconstruction can use abstract placeholders; validate data at
        # construction and through valid, rather than in tree transformations.
        obj = object.__new__(cls)
        for name, value in zip(('density', 'overlap', 'energy', 'nelec', 'layout'),
                               (*children, *metadata)):
            object.__setattr__(obj, name, value)
        return obj

    @property
    def components(self):
        """AO charge or spin blocks; GKS order is charge, mx, my, mz."""
        return _components(self.density, self.overlap.shape[0], self.layout)

    @property
    def electron_numbers(self):
        blocks = self.components[:1] if self.layout == 'gks' else self.components
        return jnp.einsum('ij,cji->c', self.overlap, blocks).real

    @property
    def valid(self):
        s, d = self.overlap, self.density
        tolerance = jnp.maximum(1e-9, 100*jnp.finfo(d.real.dtype).eps)
        adjoint = lambda a: a.conj().swapaxes(-1, -2)
        good = jnp.all(jnp.isfinite(s)) & jnp.all(jnp.isfinite(d))
        good &= jnp.max(jnp.abs(s-adjoint(s))) <= tolerance
        good &= jnp.max(jnp.abs(d-adjoint(d))) <= tolerance
        good &= jnp.min(jnp.linalg.eigvalsh(s)) > 0
        root = jnp.linalg.cholesky(s)
        if self.layout == 'gks': root = jnp.kron(jnp.eye(2, dtype=s.dtype), root)
        occupations = jnp.linalg.eigvalsh(adjoint(root) @ d @ root)
        good &= jnp.all((occupations >= -tolerance) & (occupations <= (2 if self.layout == 'rks' else 1)+tolerance))
        good &= jnp.all(jnp.abs(self.electron_numbers-jnp.atleast_1d(jnp.asarray(self.nelec))) <= tolerance)
        if self.energy is not None: good &= jnp.isfinite(self.energy)
        return good
