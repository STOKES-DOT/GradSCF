"""Real CI total-spin expectation, including distinct alpha/beta MO frames.

Uses S^2 = S_z^2 + (S_+S_- + S_-S_+)/2 and the existing fermionic 2-RDM.
No total-spin projection or integer multiplicity assignment is performed.
"""
import jax.numpy as jnp

from .space import CISpace, UCISpace
from .properties import make_rdm2


def spin_square(coefficients, space, *, overlap_ab=None):
    """Return (<S^2>, sqrt(1+4<S^2>)) for a normalized real CI state.

    Normalization follows make_rdm2. Restricted spaces share spatial orbitals.
    UCISpace requires overlap_ab[p,q] = <p_alpha|q_beta> explicitly, even for
    ROHF/common frames (supply the identity in that case). Each spin frame must
    be orthonormal; consistency of overlap_ab with the orbitals is the caller's
    responsibility. Frozen electrons participate in the density contraction.
    The returned effective multiplicity need not be an integer for mixed spin.
    Integral response requires implicit-eigenvector CI coefficients; the default
    eigenvalue-only mode stops coefficient response.
    """
    if isinstance(space, CISpace):
        if overlap_ab is not None:
            raise ValueError("overlap_ab applies only to unrestricted CI spaces")
        spin_space = UCISpace(space.nmo, (space.nocc, space.nocc), space.max_excitation,
                              (space.frozen, space.frozen), space.determinants, space.ranks)
        overlap = jnp.eye(space.nmo)
    elif isinstance(space, UCISpace):
        if overlap_ab is None:
            raise ValueError("Unrestricted spin_square requires overlap_ab = C_alpha.T @ S @ C_beta")
        spin_space = space
        overlap = jnp.asarray(overlap_ab)
        if overlap.shape != (space.nmo, space.nmo):
            raise ValueError("overlap_ab must have shape (nmo,nmo)")
        if jnp.iscomplexobj(overlap):
            raise NotImplementedError("Spin diagnostics currently require real orbital overlaps")
    else:
        raise TypeError("Spin diagnostics require a determinant CISpace or UCISpace")
    na, nb = spin_space.nocc
    gamma_ab = make_rdm2(coefficients, spin_space)[1]
    exchange = jnp.einsum("pqrs,ps,qr->", gamma_ab, overlap, overlap, precision="highest")
    ss = .25*(na-nb)**2 + .5*(na+nb)-exchange
    ss = jnp.where(jnp.all(jnp.isfinite(overlap)), ss, jnp.nan)
    return ss, jnp.sqrt(1+4*ss)
