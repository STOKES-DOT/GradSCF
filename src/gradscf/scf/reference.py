"""Shared eager conversion from converged restricted HF to MO Hamiltonians."""

from dataclasses import dataclass
import numpy as np
from ..integrals.mo import transform_integrals, validate_integrals


@dataclass(frozen=True)
class RestrictedReference:
    h1: object
    eri: object
    nocc: int
    nuclear_repulsion: object = 0.0
    mo_energy: object = None


def reference_from_source(source):
    if isinstance(source, RestrictedReference):
        h, g = validate_integrals(source.h1, source.eri)
        if not isinstance(source.nocc, int) or not 0 <= source.nocc <= h.shape[0]:
            raise ValueError("Invalid restricted nocc")
        return RestrictedReference(
            h, g, source.nocc, source.nuclear_repulsion, source.mo_energy
        )
    from .facade import RKS

    if not isinstance(source, RKS) or str(source.xc).strip().lower() != "hf":
        raise NotImplementedError(
            "Post-HF requires a GradSCF closed-shell HF reference (RKS with xc='hf') or RestrictedReference"
        )
    if not source.converged or source.scf_result is None or source._scf_inputs is None:
        raise RuntimeError("Run and converge the HF SCF before post-HF")
    # Detect stale molecular parameters using the facade's existing guard.
    if source._cached_scf_key != source._scf_signature():
        raise RuntimeError("SCF inputs changed; run SCF again before post-HF")
    inputs = source._scf_inputs
    nocc = inputs.nelectron // 2
    c = source.mo_coeff
    if c.ndim != 2 or inputs.nelectron % 2:
        raise NotImplementedError("Only real closed-shell HF references are supported")
    expected = np.zeros(c.shape[1])
    expected[:nocc] = 2
    if not np.allclose(np.asarray(source.mo_occ), expected, atol=1e-10, rtol=0):
        raise ValueError(
            "Post-HF requires integer occupations, occupied orbitals first"
        )
    if inputs.df_factors is not None:
        kwargs = {"df_factors": inputs.df_factors}
    elif inputs.eri is not None:
        kwargs = {"eri": inputs.eri}
    else:
        pair = inputs.response_eri_pair_matrix()
        if pair is None:
            raise ValueError(
                "SCF reference does not supply post-HF two-electron integrals"
            )
        kwargs = {"eri_pair_matrix": pair}
    h, g = transform_integrals(inputs.hcore, c, **kwargs)
    return RestrictedReference(h, g, nocc, inputs.nuclear_repulsion, source.mo_energy)
