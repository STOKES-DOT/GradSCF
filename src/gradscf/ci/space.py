"""Static determinant spaces; alpha orbitals precede beta orbitals."""
from dataclasses import dataclass
from itertools import combinations
from math import comb
from numbers import Integral


@dataclass(frozen=True)
class CISpace:
    nmo: int
    nocc: int
    max_excitation: int
    frozen: tuple[int, ...]
    determinants: tuple[int, ...]
    ranks: tuple[int, ...]

    @property
    def size(self):
        return len(self.determinants)


def frozen_indices(nmo, nocc, frozen):
    if frozen is None:
        return ()
    if isinstance(frozen, Integral) and not isinstance(frozen, bool):
        if frozen < 0 or frozen > nocc:
            raise ValueError("frozen core count must lie between 0 and nocc")
        return tuple(range(frozen))
    try:
        indices = tuple(frozen)
    except TypeError as error:
        raise ValueError("frozen must be a core count or orbital index list") from error
    if any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0 or i >= nmo for i in indices):
        raise ValueError("Frozen orbital indices must be integers in [0, nmo)")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate frozen orbital indices")
    return tuple(sorted(int(i) for i in indices))


def make_ci_space(nmo, nocc, *, max_excitation=2, frozen=None, max_determinants=5000):
    """Generate only retained excitations, without enumerating the FCI space.

    Frozen occupied electrons remain explicitly occupied in all determinants;
    frozen virtual orbitals remain empty. Spin is conserved separately per channel.
    """
    if not isinstance(nmo, Integral) or nmo < 1:
        raise ValueError("nmo must be a positive integer")
    if not isinstance(nocc, Integral) or nocc < 0 or nocc > nmo:
        raise ValueError("nocc must be an integer in [0, nmo]")
    if not isinstance(max_excitation, Integral) or max_excitation < 0:
        raise ValueError("max_excitation must be a nonnegative integer")
    if not isinstance(max_determinants, Integral) or max_determinants < 1:
        raise ValueError("max_determinants must be a positive integer")
    frozen = frozen_indices(nmo, nocc, frozen)
    occ = tuple(i for i in range(nocc) if i not in frozen)
    vir = tuple(i for i in range(nocc, nmo) if i not in frozen)
    max_rank = min(max_excitation, len(occ), len(vir))
    counts = [comb(len(occ), k) * comb(len(vir), k) for k in range(max_rank + 1)]
    count = sum(a * b for i, a in enumerate(counts) for j, b in enumerate(counts)
                if i + j <= max_excitation)
    if count > max_determinants:
        raise ValueError(f"CI space requires {count} determinants; max_determinants={max_determinants}")
    reference = (1 << nocc) - 1
    channels = []
    for rank in range(max_rank + 1):
        strings = []
        for holes in combinations(occ, rank):
            for particles in combinations(vir, rank):
                strings.append(reference ^ sum(1 << p for p in holes + particles))
        channels.append(strings)
    entries = [(i + j, a | (b << nmo))
               for i, aa in enumerate(channels) for j, bb in enumerate(channels)
               if i + j <= max_excitation for a in aa for b in bb]
    entries.sort()
    return CISpace(int(nmo), int(nocc), int(max_excitation), frozen,
                   tuple(d for _, d in entries), tuple(r for r, _ in entries))


def excite(det, holes, particles):
    """Apply a_p^+ a_r^+ ... a_s a_q; ordered holes/particles, fermion phase."""
    phase = 1
    for orbital in holes:
        if not det & (1 << orbital):
            return None, 0
        phase *= -1 if (det & ((1 << orbital) - 1)).bit_count() % 2 else 1
        det ^= 1 << orbital
    for orbital in reversed(particles):
        if det & (1 << orbital):
            return None, 0
        phase *= -1 if (det & ((1 << orbital) - 1)).bit_count() % 2 else 1
        det ^= 1 << orbital
    return det, phase
