"""Eager checked G0W0/W0 and explicit fixed-MO reference snapshots."""

from dataclasses import dataclass
import jax.numpy as jnp
from ..scf.reference import _array_signature


@dataclass(frozen=True)
class BSEReference:
    """Explicit real MO inputs. Omitted masks mean user-supplied QP energies.

    The caller supplies one orthonormal frame, common to the spectra, factors
    and optional dipoles. Screening energies are always explicit.
    """

    qp_energy: object
    screening_energy: object
    mo_factors: object
    nocc: int
    dipole_mo: object = None
    qp_computed_mask: object = None
    qp_converged_mask: object = None


def source_signature(source):
    if isinstance(source, BSEReference):
        arrays = (
            source.qp_energy,
            source.screening_energy,
            source.mo_factors,
            source.dipole_mo,
            source.qp_computed_mask,
            source.qp_converged_mask,
        )
        return source.nocc, tuple(
            None if x is None else _array_signature(x) for x in arrays
        )
    from ..gw.rgw import GW

    if not isinstance(source, GW):
        raise NotImplementedError(
            "BSE accepts a real restricted G0W0 facade or explicit BSEReference"
        )
    return source.state_signature()


def reference_from_source(source, *, max_aux=1024, max_factor_elements=20_000_000):
    if isinstance(source, BSEReference):
        return source
    from ..gw.rgw import GW

    if not isinstance(source, GW):
        raise NotImplementedError(
            "BSE accepts a real restricted G0W0 facade or explicit BSEReference"
        )
    data = source.get_bse_inputs(
        max_aux=max_aux, max_factor_elements=max_factor_elements
    )
    dipole = None
    if data["dipole_ao"] is not None:
        c = jnp.asarray(data["mo_coeff"])
        dipole = jnp.einsum(
            "xmn,mi,nj->xij", data["dipole_ao"], c, c, precision="highest"
        )
    return BSEReference(
        data["qp_energy"],
        data["screening_energy"],
        data["mo_factors"],
        data["nocc"],
        dipole,
        data["qp_computed_mask"],
        data["qp_converged_mask"],
    )
