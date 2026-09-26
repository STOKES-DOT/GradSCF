"""Real CI reduced densities from fermionic operator connections.

dm1[p,q] = <a_q^+ a_p>; dm2[p,q,r,s] = <a_p^+ a_r^+ a_s a_q>.
Restricted results are spin summed; unrestricted results contain (a,b) and
(aa,ab,bb). Frozen electrons remain explicit. No integral or eigensolve is needed.
"""
from functools import lru_cache
from itertools import combinations

import jax.numpy as jnp
import numpy as np

from .space import CISpace, UCISpace
from ..fci.cistring import excite


@lru_cache(maxsize=8)
def _density_connections(space, rank):
    n = space.nmo
    lookup = {d: i for i, d in enumerate(space.determinants)}
    links = []
    for ket, det in enumerate(space.determinants):
        occupied = [p for p in range(2*n) if det & (1 << p)]
        for holes in combinations(occupied, rank):
            remaining, _ = excite(det, holes, ())
            available = [p for p in range(2*n) if not remaining & (1 << p)]
            for particles in combinations(available, rank):
                if sum(p//n for p in particles) != sum(q//n for q in holes):
                    continue
                target, sign = excite(det, holes, particles)
                bra = lookup.get(target)
                if bra is None:
                    continue
                if rank == 1:
                    p, q = particles[0], holes[0]
                    links.append((p//n, bra, ket, q % n, p % n, sign))
                else:
                    p, r = particles
                    q, s = holes
                    block = p//n + r//n
                    p, q, r, s = p % n, q % n, r % n, s % n
                    links.append((block, bra, ket, p, q, r, s, sign))
                    if block != 1:
                        links.extend(((block, bra, ket, r, q, p, s, -sign),
                                      (block, bra, ket, p, s, r, q, -sign),
                                      (block, bra, ket, r, s, p, q, sign)))
                if len(links) > 4_000_000:
                    raise ValueError("CI density connection limit exceeded; reduce the orbital space")
    return np.asarray(links, dtype=np.int32).reshape(-1, 2*rank+4)


def _density(coefficients, space, rank):
    if not isinstance(space, (CISpace, UCISpace)):
        raise TypeError("A determinant CISpace or UCISpace is required")
    c = jnp.asarray(coefficients)
    if c.shape != (space.size,):
        raise ValueError("CI coefficients must have shape (ndeterminants,)")
    if jnp.iscomplexobj(c):
        raise NotImplementedError("CI densities currently require real coefficients")
    c = c.astype(jnp.result_type(c, 1.0))
    norm = jnp.vdot(c, c).real
    links = _density_connections(space, rank)
    blocks = jnp.zeros((rank+1,) + (space.nmo,)*(2*rank), c.dtype)
    if len(links):
        block, bra, ket, *indices, signs = links.T
        values = signs*c[bra]*c[ket]/jnp.where(norm > 0, norm, 1.)
        blocks = blocks.at[(block, *indices)].add(values)
    blocks = jnp.where((norm > 0) & jnp.isfinite(norm), blocks, jnp.nan)
    if isinstance(space, UCISpace):
        return tuple(blocks[i] for i in range(rank+1))
    if rank == 1:
        return blocks[0]+blocks[1]
    return blocks[0]+blocks[1]+blocks[1].transpose(2, 3, 0, 1)+blocks[2]


def make_rdm1(coefficients, space):
    """Normalized real MO 1-RDM, with all frozen-core occupations restored.

    Coefficients may be unnormalized. A zero/nonfinite vector yields NaNs.
    AD with respect to the vector is supported; integral response requires
    coefficients from gradient_mode='implicit_eigenvector', on an isolated root.
    """
    return _density(coefficients, space, 1)


def make_rdm2(coefficients, space):
    """Normalized MO 2-RDM in chemists' order, including frozen electrons.

    E_ee = .5 * sum(eri * dm2) in the restricted case. For spin blocks the
    factors for (aa,ab,bb) are (.5,1,.5). This is a true fermionic density,
    not the eightfold-symmetrized derivative with respect to real ERIs.
    """
    return _density(coefficients, space, 2)


def make_rdm12(coefficients, space):
    """Return (dm1, dm2); see make_rdm1 and make_rdm2 for conventions."""
    return make_rdm1(coefficients, space), make_rdm2(coefficients, space)
