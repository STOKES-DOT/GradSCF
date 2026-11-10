"""Fixed orbital references; correlated targets need not share their 1-RDM."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .types import _check


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OrbitalReference:
    """Complete spatial/spinor orbitals supplied by a declared reference method.

    Energies are Hartree. RKS occupations lie in [0,2]; UKS/spinor occupations
    in [0,1]. ``valid`` checks representation and electron counts, not density
    inversion accuracy or ground-state stability. In particular, the target
    can be a correlated 1-RDM different from ``density``. ``method`` records
    provenance; supplying orbitals does not execute an inverse calculation.
    Optional xc_potential is an AO XC operator, not a total Fock matrix.
    """

    target: object
    coefficients: object
    occupations: object
    energies: object
    xc_potential: object = None
    method: str = 'supplied_orbitals'
    reference_valid: object = True

    def __post_init__(self):
        for name in ('coefficients', 'occupations', 'energies', 'reference_valid'):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name)))
        n = self.target.density.shape[-1]
        expected = (2, n, n) if self.target.layout == 'uks' else (n, n)
        if self.coefficients.shape != expected:
            raise ValueError('An orbital reference requires a complete square AO/MO space')
        if self.occupations.shape != expected[:-1] or self.energies.shape != expected[:-1]:
            raise ValueError('Reference occupations and energies must match the orbital columns')
        if jnp.iscomplexobj(self.occupations) or jnp.iscomplexobj(self.energies):
            raise ValueError('Reference occupations and energies must be real')
        if self.target.layout != 'gks' and jnp.iscomplexobj(self.coefficients):
            raise ValueError('Complex orbital references require layout=gks')
        if not isinstance(self.method, str) or not self.method:
            raise ValueError('Declare the orbital reference method')
        if self.reference_valid.shape != ():
            raise ValueError('reference_valid must be scalar')
        if self.xc_potential is not None:
            value = jnp.asarray(self.xc_potential)
            if value.shape != self.target.density.shape:
                raise ValueError('Reference XC operator must match the AO/spin density shape')
            object.__setattr__(self, 'xc_potential', value)
        _check(self.valid, 'Invalid orbital reference: check orthonormality, electron counts and finite data')

    @property
    def overlap(self):
        s = self.target.overlap
        return jnp.kron(jnp.eye(2, dtype=s.dtype), s) if self.target.layout == 'gks' else s

    @property
    def density(self):
        c = self.coefficients
        return (c*self.occupations[..., None, :])@c.conj().swapaxes(-1, -2)

    @property
    def fock(self):
        """Reference spectral operator in the declared complete orbital space."""
        c, s = self.coefficients, self.overlap
        return s@(c*self.energies[..., None, :])@c.conj().swapaxes(-1, -2)@s

    @property
    def integer_occupations(self):
        maximum = 2 if self.target.layout == 'rks' else 1
        return jnp.all((jnp.abs(self.occupations) <= 1e-10)
                       | (jnp.abs(self.occupations-maximum) <= 1e-10))

    @property
    def valid(self):
        c, f, s = self.coefficients, self.occupations, self.overlap
        tolerance = jnp.maximum(1e-9, 100*jnp.finfo(c.real.dtype).eps)
        maximum = 2 if self.target.layout == 'rks' else 1
        gram = c.conj().swapaxes(-1, -2)@s@c
        counts = jnp.sum(f, axis=-1)
        good = self.target.valid & jnp.isfinite(self.reference_valid) & (self.reference_valid == 1)
        good &= jnp.all(jnp.isfinite(c)) & jnp.all(jnp.isfinite(f)) & jnp.all(jnp.isfinite(self.energies))
        good &= jnp.all(jnp.abs(gram-jnp.eye(c.shape[-1])) <= tolerance)
        good &= jnp.all((f >= -tolerance) & (f <= maximum+tolerance))
        good &= jnp.all(jnp.abs(counts-jnp.asarray(self.target.nelec)) <= tolerance)
        if self.xc_potential is not None:
            v = self.xc_potential
            good &= jnp.all(jnp.isfinite(v)) & jnp.all(jnp.abs(v-v.conj().swapaxes(-1, -2)) <= tolerance)
        return good

    def tree_flatten(self):
        return (self.target, self.coefficients, self.occupations, self.energies,
                self.xc_potential, self.reference_valid), self.method

    @classmethod
    def tree_unflatten(cls, method, children):
        obj = object.__new__(cls)
        for name, value in zip(('target', 'coefficients', 'occupations', 'energies',
                                'xc_potential', 'reference_valid'), children):
            object.__setattr__(obj, name, value)
        object.__setattr__(obj, 'method', method)
        return obj
