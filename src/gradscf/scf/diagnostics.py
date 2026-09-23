"""Read-only restricted SCF stationarity diagnostics from a fresh stored state."""

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from .reference import _array_signature
from .rks import _mo_residual_norm


@dataclass(frozen=True)
class RestrictedSCFDiagnostics:
    gradient_norm: float
    gradient_tolerance: float
    orthogonality_error: float
    electron_count_error: float
    converged: bool
    stationary: bool


def _restricted_scf_state(source):
    from .facade import RKS

    if not isinstance(source, RKS):
        raise TypeError("Restricted diagnostics require a GradSCF RKS source")
    if source.scf_result is None or source._scf_inputs is None:
        raise RuntimeError("Run SCF before requesting diagnostics")
    if source._cached_scf_key != source._scf_signature():
        raise RuntimeError("SCF inputs or settings changed; run SCF again")
    result = source.scf_result
    for name in ("mo_coeff", "mo_occ"):
        if _array_signature(getattr(source, name)) != _array_signature(
            getattr(result, name)
        ):
            raise RuntimeError("SCF orbital arrays changed; run SCF again")
    if jnp.iscomplexobj(result.mo_coeff):
        raise NotImplementedError("Restricted diagnostics require real orbitals")
    return result, source._scf_inputs


def restricted_scf_diagnostics(source, *, gradient_tol=None):
    """Report ||2 F_vo||, the existing RKS half-angle gradient convention.

    This is not a stability check or an excitation-energy error bound. No final
    delta-energy/delta-density values are fabricated: SCF does not store them.
    """
    result, inputs = _restricted_scf_state(source)
    tol = source.conv_tol_grad if gradient_tol is None else float(gradient_tol)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("gradient_tol must be finite and positive")
    c, s = result.mo_coeff, result.overlap_matrix
    gradient = float(_mo_residual_norm(result.fock_matrix, c, result.mo_occ))
    orth = float(jnp.linalg.norm(c.T @ s @ c - jnp.eye(c.shape[1])))
    electrons = float(jnp.abs(jnp.trace(result.density_matrix @ s) - inputs.nelectron))
    converged = bool(source.converged and result.converged)
    stationary = bool(
        converged
        and np.isfinite(gradient + orth + electrons)
        and gradient <= tol
        and orth <= 1e-8
        and electrons <= 1e-8
    )
    return RestrictedSCFDiagnostics(
        gradient, tol, orth, electrons, converged, stationary
    )
