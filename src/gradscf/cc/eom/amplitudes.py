"""Static restricted EE singlet and charged doublet amplitude coordinates."""

from dataclasses import dataclass
import jax.numpy as jnp
from ..amplitudes import AmplitudeSpace


@dataclass(frozen=True)
class EOMAmplitudeSpace:
    nocc: int
    nvir: int
    sector: str

    def __post_init__(self):
        if self.sector not in {"ee", "ip", "ea"}:
            raise ValueError("EOM sector must be ee, ip or ea")
        if any(
            not isinstance(n, int) or isinstance(n, bool) or n < 0
            for n in (self.nocc, self.nvir)
        ):
            raise ValueError("Orbital counts must be nonnegative integers")

    @property
    def shapes(self):
        no, nv = self.nocc, self.nvir
        if self.sector == "ee":
            return (no, nv), (no, no, nv, nv)
        if self.sector == "ip":
            return (no,), (no, no, nv)
        return (nv,), (no, nv, nv)

    @property
    def size(self):
        no, nv = self.nocc, self.nvir
        if self.sector == "ee":
            k = no * nv
            return k + k * (k + 1) // 2
        return no + no * no * nv if self.sector == "ip" else nv + no * nv * nv

    def pack(self, r1, r2):
        r1, r2 = jnp.asarray(r1), jnp.asarray(r2)
        if (r1.shape, r2.shape) != self.shapes:
            raise ValueError("EOM amplitudes do not match the active space")
        if jnp.iscomplexobj(r1) or jnp.iscomplexobj(r2):
            raise NotImplementedError("EOM amplitudes currently require real inputs")
        if self.sector == "ee":
            return AmplitudeSpace(self.nocc, self.nvir).pack(r1, r2)
        return jnp.concatenate([r1.ravel(), r2.ravel()])

    def unpack(self, vector):
        vector = jnp.asarray(vector)
        if jnp.iscomplexobj(vector):
            raise NotImplementedError("EOM amplitudes currently require real inputs")
        if vector.shape != (self.size,):
            raise ValueError("EOM vector size mismatch")
        if self.sector == "ee":
            return AmplitudeSpace(self.nocc, self.nvir).unpack(vector)
        nr = self.nocc if self.sector == "ip" else self.nvir
        return vector[:nr], vector[nr:].reshape(self.shapes[1])
