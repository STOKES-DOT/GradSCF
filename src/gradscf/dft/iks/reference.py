"""Eager RKS/UKS/GKS snapshots without reference rebuilding or SCF iteration."""
from dataclasses import dataclass

import jax.numpy as jnp

from gradscf._memory import resolve_memory_mb
from .types import Target, _check


@dataclass(frozen=True)
class Reference:
    """Preparation metadata, not a live inner-SCF or inverse-GKS implementation.

    Retain the exact Functional/TraditionalXCFunctional and explicit params.
    Unsupported downstream operator combinations must be checked by their
    consumer; accepting a snapshot does not imply forward/response support.
    """

    layout: str
    overlap: object
    hcore: object
    density: object
    nelec: object
    functional: object
    params: object
    max_memory: float

    @classmethod
    def from_source(cls, source, *, max_memory=None):
        from gradscf.scf.facade import RKS, UKS
        from gradscf.scf.gks import GKS
        from gradscf.dft.functional import Functional, custom_xc
        from gradscf.dft.xc import TraditionalXCFunctional

        if not isinstance(source, (RKS, UKS, GKS)):
            raise TypeError('An inverse reference requires a molecular RKS, UKS or GKS source')
        result = source.scf_result
        if result is None or not source.converged:
            raise ValueError('Run a converged source before taking an inverse reference')
        cached = getattr(source, '_cached_scf_key', None)
        if cached is not None and cached != source._scf_signature():
            raise ValueError('SCF source changed after convergence; rebuild it before taking a snapshot')
        layout = 'gks' if isinstance(source, GKS) else ('uks' if isinstance(source, UKS) else 'rks')
        density = (jnp.stack((result.density_matrix_alpha, result.density_matrix_beta))
                   if layout == 'uks' else jnp.asarray(result.density_matrix))
        s, h = jnp.asarray(result.overlap_matrix), jnp.asarray(result.hcore_matrix)
        if layout == 'gks' and s.shape == density.shape:
            n = density.shape[0]//2
            _check(jnp.allclose(s[:n, :n], s[n:, n:]) & jnp.all(s[:n, n:] == 0)
                   & jnp.all(s[n:, :n] == 0), 'Spinor overlap must contain identical spatial blocks')
            s = s[:n, :n]
        number = source.mol.nelectron
        nelec = ((number+source.mol.spin)//2, (number-source.mol.spin)//2) if layout == 'uks' else number
        functional = source.xc
        if isinstance(functional, str): functional = custom_xc(functional)
        if not isinstance(functional, (Functional, TraditionalXCFunctional)):
            raise TypeError('Use a named recipe or the existing custom_xc functional')
        target = Target(density, s, nelec, layout=layout)
        allowed = (s.shape, density.shape) if layout == 'gks' else (s.shape,)
        if h.shape not in allowed:
            raise ValueError('Core Hamiltonian does not match the spatial/spinor reference')
        _check(jnp.all(jnp.isfinite(h)) & jnp.allclose(h, h.conj().T), 'Core Hamiltonian must be finite and Hermitian')
        return cls(layout, target.overlap, h, density, nelec, functional,
                   source.params, resolve_memory_mb(max_memory, source))

    def target(self, density=None, *, energy=None):
        return Target(self.density if density is None else density, self.overlap,
                      self.nelec, layout=self.layout, energy=energy)
