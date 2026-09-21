"""Independent spin-conserving coordinates for antisymmetric CC amplitudes."""
from dataclasses import dataclass
from itertools import combinations
from functools import cached_property
import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class SpinAmplitudeSpace:
    nocc: tuple[int, int]
    nvir: tuple[int, int]
    method: str = "ccsd"

    @cached_property
    def indices(self):
        so = [0]*self.nocc[0] + [1]*self.nocc[1]
        sv = [0]*self.nvir[0] + [1]*self.nvir[1]
        singles = [(i, a) for i in range(len(so)) for a in range(len(sv))
                   if so[i] == sv[a] and self.method != "ccd"]
        doubles = [(i, j, a, b) for i, j in combinations(range(len(so)), 2)
                   for a, b in combinations(range(len(sv)), 2)
                   if so[i]+so[j] == sv[a]+sv[b]]
        return (np.asarray(singles, dtype=np.int32).reshape(-1, 2).T,
                np.asarray(doubles, dtype=np.int32).reshape(-1, 4).T)

    def pack(self, t1, t2):
        """Norm-preserving coordinates: each independent double occurs four times."""
        singles, doubles = self.indices
        return jnp.concatenate((t1[tuple(singles)], 2*t2[tuple(doubles)]))

    def unpack(self, vector):
        no, nv = sum(self.nocc), sum(self.nvir)
        singles, doubles = self.indices
        ns = singles.shape[1]
        t1 = jnp.zeros((no, nv), vector.dtype).at[tuple(singles)].set(vector[:ns])
        i, j, a, b = doubles
        v = vector[ns:]/2
        t2 = jnp.zeros((no, no, nv, nv), vector.dtype)
        t2 = t2.at[i, j, a, b].set(v).at[j, i, a, b].set(-v)
        return t1, t2.at[i, j, b, a].set(-v).at[j, i, b, a].set(v)

    def to_blocks(self, t1, t2):
        oa, va = self.nocc[0], self.nvir[0]
        return ((t1[:oa, :va], t1[oa:, va:]),
                (t2[:oa, :oa, :va, :va], t2[:oa, oa:, :va, va:],
                 t2[oa:, oa:, va:, va:]))

    def from_blocks(self, t1, t2, dtype):
        oa, ob = self.nocc
        va, vb = self.nvir
        shapes1 = ((oa, va), (ob, vb))
        shapes2 = ((oa, oa, va, va), (oa, ob, va, vb), (ob, ob, vb, vb))
        if len(t1) != 2 or len(t2) != 3:
            raise ValueError("UCC amplitudes require two singles and three doubles blocks")
        blocks = tuple(jnp.asarray(a) for a in (*t1, *t2))
        if any(jnp.iscomplexobj(a) for a in blocks):
            raise NotImplementedError("UCC amplitudes must be real")
        if any(a.shape != s for a, s in zip(blocks, shapes1+shapes2)):
            raise ValueError("Initial UCC amplitude shapes do not match the active space")
        dtype = jnp.result_type(dtype, *blocks)
        ta, tb, taa, tab, tbb = (a.astype(dtype) for a in blocks)
        # Project same-spin restart amplitudes onto their antisymmetric subspace.
        project = lambda t: .25*(t-t.swapaxes(0, 1)-t.swapaxes(2, 3)+t.transpose(1, 0, 3, 2))
        s = jnp.zeros((oa+ob, va+vb), dtype).at[:oa, :va].set(ta).at[oa:, va:].set(tb)
        d = jnp.zeros((oa+ob, oa+ob, va+vb, va+vb), dtype)
        d = d.at[:oa, :oa, :va, :va].set(project(taa)).at[oa:, oa:, va:, va:].set(project(tbb))
        d = d.at[:oa, oa:, :va, va:].set(tab).at[oa:, :oa, :va, va:].set(-tab.swapaxes(0, 1))
        d = d.at[:oa, oa:, va:, :va].set(-tab.swapaxes(2, 3))
        return s, d.at[oa:, :oa, va:, :va].set(tab.transpose(1, 0, 3, 2))
