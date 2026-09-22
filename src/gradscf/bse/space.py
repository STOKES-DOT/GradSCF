"""Static occupied/virtual index spaces shared by optical and screening inputs."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class BSESpace:
    nmo: int
    nocc: int
    occupied: tuple[int, ...]
    virtual: tuple[int, ...]

    def __post_init__(self):
        if (
            not isinstance(self.nmo, Integral)
            or not isinstance(self.nocc, Integral)
            or not 0 < self.nocc < self.nmo
        ):
            raise ValueError(
                "BSE requires occupied and virtual orbitals in a finite closed-shell space"
            )
        if not isinstance(self.occupied, tuple) or not isinstance(self.virtual, tuple):
            raise TypeError("BSESpace indices must be static tuples")
        for indices, lo, hi in (
            (self.occupied, 0, self.nocc),
            (self.virtual, self.nocc, self.nmo),
        ):
            if any(
                not isinstance(p, Integral) or not lo <= p < hi for p in indices
            ) or len(set(indices)) != len(indices):
                raise ValueError(
                    "BSE occupied/virtual indices must be distinct and in their respective ranges"
                )

    @property
    def size(self):
        return len(self.occupied) * len(self.virtual)


def make_bse_space(nmo, nocc, *, occupied=None, virtual=None):
    return BSESpace(
        nmo,
        nocc,
        tuple(range(nocc)) if occupied is None else tuple(occupied),
        tuple(range(nocc, nmo)) if virtual is None else tuple(virtual),
    )
