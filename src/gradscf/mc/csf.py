"""Fixed-spin configuration state functions and direct graph contractions.

A distinct-row table (DRT) stores the allowed U(n) branching paths. Each
spin-free generator is evaluated by contracting the spin-coupling segment
between its two orbital indices. Only nonzero CSF links are retained; no
determinant expansion, determinant CI storage, or square Hamiltonian is used
in these contractions.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gradscf._memory import DEFAULT_MAX_MEMORY_MB, workspace_bytes


# Order: empty, single increasing 2S, single decreasing 2S, double.
_ELECTRONS = (0, 1, 1, 2)
_SPIN_STEP = (0, 1, -1, 0)
_LOCAL_STATES = ((0,), (1, 2), (1, 2), (3,))


class _Arc(NamedTuple):
    step: int
    end: tuple[int, int]
    offset: int
    count: int


@dataclass(frozen=True, eq=False)
class CSFGraph:
    """Numbered DRT paths in a fixed-(N,S,M_S) sector."""

    norb: int
    nelec: tuple[int, int]
    spin2: int
    dimension: int
    _layers: tuple[Mapping[tuple[int, int], tuple[_Arc, ...]], ...]

    @classmethod
    def build(cls, norb, nelec, spin2):
        """Construct a DRT using suffix path counts, without listing paths."""
        if not isinstance(norb, Integral) or isinstance(norb, bool) or norb < 0:
            raise ValueError("norb must be a nonnegative integer")
        if not isinstance(spin2, Integral) or isinstance(spin2, bool) or spin2 < 0:
            raise ValueError("spin2 must be a nonnegative integer")
        norb, spin2 = int(norb), int(spin2)
        if isinstance(nelec, Integral) and not isinstance(nelec, bool):
            total = int(nelec)
            if (total + spin2) % 2:
                raise ValueError("nelec and spin2 must have matching parity")
            na, nb = (total + spin2) // 2, (total - spin2) // 2
        else:
            if (not isinstance(nelec, (tuple, list)) or len(nelec) != 2
                    or any(not isinstance(n, Integral) or isinstance(n, bool) for n in nelec)):
                raise ValueError("nelec must be an integer or a pair (nalpha, nbeta)")
            na, nb = map(int, nelec)
            total = na + nb
        if not 0 <= na <= norb or not 0 <= nb <= norb:
            raise ValueError("nelec does not fit the active orbital space")
        if (abs(na - nb) > spin2 or spin2 > min(total, 2 * norb - total)
                or (total + spin2) % 2):
            raise ValueError("spin2 is incompatible with nelec and active orbitals")

        layers = [{} for _ in range(norb + 1)]

        @lru_cache(maxsize=None)
        def count_paths(level, n, spin):
            remaining = norb - level
            if not 0 <= total - n <= 2 * remaining or abs(spin2 - spin) > remaining:
                return 0
            if level == norb:
                layers[level][n, spin] = ()
                return 1
            arcs = []
            offset = 0
            for step, (electrons, delta) in enumerate(zip(_ELECTRONS, _SPIN_STEP)):
                end = n + electrons, spin + delta
                if end[1] >= 0:
                    count = count_paths(level + 1, *end)
                    if count:
                        arcs.append(_Arc(step, end, offset, count))
                        offset += count
            if arcs:
                layers[level][n, spin] = tuple(arcs)
            return offset

        dimension = count_paths(0, 0, 0)
        count_paths.cache_clear()
        return cls(norb, (na, nb), spin2, dimension, tuple(map(MappingProxyType, layers)))

    @property
    def node_count(self):
        return sum(map(len, self._layers))

    @property
    def arc_count(self):
        return sum(len(arcs) for layer in self._layers for arcs in layer.values())

    def rank(self, path):
        """Return the zero-based address of a path in DRT order."""
        path = tuple(path)
        if len(path) != self.norb:
            raise ValueError("CSF path length does not match norb")
        index, row = 0, (0, 0)
        for layer, step in zip(self._layers, path):
            for arc in layer[row]:
                if arc.step == step:
                    index += arc.offset
                    row = arc.end
                    break
            else:
                raise ValueError("CSF path leaves the requested DRT")
        return index

    def unrank(self, index):
        """Return the branching steps of a zero-based CSF address."""
        if not isinstance(index, Integral) or not 0 <= index < self.dimension:
            raise IndexError("CSF index is out of range")
        row = (0, 0)
        path = []
        for layer in self._layers[:-1]:
            for arc in layer[row]:
                if index < arc.offset + arc.count:
                    path.append(arc.step)
                    index -= arc.offset
                    row = arc.end
                    break
        return tuple(path)


@lru_cache(maxsize=8192)
def _cg(spin, projection, step, local_state):
    """Sequential coupling coefficient in the Condon–Shortley convention."""
    if not -spin <= projection <= spin:
        return 0.0
    if step in (0, 3):
        return 1.0
    alpha = local_state == 1
    magnetic = projection if alpha else -projection
    denom = 2 * (spin + 1)
    if step == 1:
        return sqrt((spin + magnetic + 2) / denom)
    magnitude = sqrt((spin - magnetic) / denom)
    return -magnitude if alpha else magnitude


def _local_hop(state, p, q, site, spin):
    """Local bit state and local α-before-β fermionic sign."""
    if site == q:
        if not state & spin:
            return 0, 0
        sign = -1 if spin == 2 and state & 1 else 1
        if p == q:
            return state, 1
        return state ^ spin, sign
    if site == p:
        if state & spin:
            return 0, 0
        return state | spin, -1 if spin == 2 and state & 1 else 1
    return state, 1


_MAGNETIC = (0, 1, -1, 0)


@lru_cache(maxsize=131072)
def _segment(bra, ket, left_spin, right_m, p, q, spin):
    """Double-path transfer over the allowed prefix spin projections.

    Only one projection needs to be stored. The other differs by the
    deterministic spin carried between the two hops.
    """
    amplitude = {m: 1.0 for m in range(-left_spin, left_spin + 1, 2)}
    bra_spin = ket_spin = left_spin
    delta = 0
    for site, (bra_step, ket_step) in enumerate(zip(bra, ket)):
        next_amplitude = {}
        next_delta = delta
        bra_states = _LOCAL_STATES[bra_step]
        for local in _LOCAL_STATES[ket_step]:
            changed, sign = _local_hop(local, p, q, site, spin)
            if not sign or changed not in bra_states:
                continue
            next_delta = delta + _MAGNETIC[changed] - _MAGNETIC[local]
            for m, weight in amplitude.items():
                factor = _cg(ket_spin, m, ket_step, local)
                factor *= _cg(bra_spin, m + delta, bra_step, changed) * sign
                if factor:
                    next_m = m + _MAGNETIC[local]
                    next_amplitude[next_m] = next_amplitude.get(next_m, 0.0) + weight * factor
        if not next_amplitude:
            return 0.0
        amplitude = next_amplitude
        delta = next_delta
        bra_spin += _SPIN_STEP[bra_step]
        ket_spin += _SPIN_STEP[ket_step]
    return amplitude.get(right_m, 0.0)


class _Links(NamedTuple):
    pair: np.ndarray
    target: np.ndarray
    source: np.ndarray
    weight: np.ndarray


def _readonly(array):
    array.setflags(write=False)
    return array


def _links(edges):
    pair, target, source, weight = edges
    return _Links(*(_readonly(np.asarray(a, dtype=np.int32)) for a in (pair, target, source)),
                  _readonly(np.asarray(weight, dtype=np.float64)))


def _build_links(graph, occupation_groups, paths, spins, edge_limit, *, alpha_only=False):
    """Join occupied-pattern groups only across a valid graph hop."""
    n = graph.norb
    edges = [], [], [], []
    pair, target, source, weight = edges
    offsets = [0]
    for p in range(n):
        for q in range(p + 1):
            for occupation, sources in occupation_groups.items():
                if p == q:
                    if not alpha_only or occupation[p] == 0:
                        continue
                elif occupation[q] == 0 or occupation[p] == 2:
                    continue

                if p == q:
                    targets = sources
                    sign = 1
                else:
                    changed = list(occupation)
                    changed[p] += 1
                    changed[q] -= 1
                    targets = occupation_groups.get(tuple(changed), ())
                    sign = (-1) ** (sum(occupation[q:p]) - 1)

                stop = p + 1
                for i in sources:
                    ket = paths[i]
                    for j in targets:
                        bra = paths[j]
                        if bra[:q] != ket[:q]:
                            continue
                        left_spin = spins[i][q]
                        if alpha_only:
                            # The α generator can also change the intermediate
                            # spin after its last orbital, so retain the suffix.
                            value = _segment(bra[q:], ket[q:], left_spin,
                                             graph.nelec[0] - graph.nelec[1], p - q, 0, 1)
                        else:
                            # A scalar generator commutes with the total spin
                            # of the prefix that contains its two endpoints.
                            right_spin = spins[i][stop]
                            if spins[j][stop] != right_spin or bra[stop:] != ket[stop:]:
                                continue
                            args = bra[q:stop], ket[q:stop], left_spin, right_spin, p - q, 0
                            value = _segment(*args, 1) + _segment(*args, 2)

                        value *= sign
                        if abs(value) <= 1e-15:
                            continue
                        if len(pair) + 1 + (p != q) > edge_limit:
                            raise ValueError("CSF sparse links exceed the host memory budget")
                        pair.append(p * n + q)
                        target.append(j)
                        source.append(i)
                        weight.append(value)
                        if p != q:
                            pair.append(q * n + p)
                            target.append(i)
                            source.append(j)
                            weight.append(value)
            offsets.append(len(pair))
    return _links(edges), _readonly(np.asarray(offsets, dtype=np.int32))


def _pair_symmetrized_eri(eri, p, q, r, s):
    """Average permutations within each real orbital pair."""
    return .25 * (eri[p, q, r, s] + eri[q, p, r, s]
                  + eri[p, q, s, r] + eri[q, p, s, r])


def _path_data(graph):
    paths, spins = [], []
    groups = {}
    occupation = np.empty((graph.norb, graph.dimension), dtype=np.int8)
    for i in range(graph.dimension):
        path = graph.unrank(i)
        numbers = tuple(_ELECTRONS[step] for step in path)
        paths.append(path)
        cumulative = [0]
        for step in path:
            cumulative.append(cumulative[-1] + _SPIN_STEP[step])
        spins.append(cumulative)
        occupation[:, i] = numbers
        groups.setdefault(numbers, []).append(i)
    return _readonly(occupation), groups, paths, spins


@dataclass(frozen=True, eq=False)
class CSFOperators:
    """Reusable direct CI and density operators on one static CSF topology."""

    graph: CSFGraph
    _occupation: np.ndarray
    _links: _Links
    _pair_offsets: np.ndarray
    _alpha: _Links | None

    @classmethod
    def build(cls, graph, *, max_memory=DEFAULT_MAX_MEMORY_MB):
        """Plan host construction before listing paths or retaining sparse links.

        The host budget is in decimal MB. It includes estimates for the Python
        path tables and edge assembly. A CI matvec uses O(norb² * dimension)
        floating-point work arrays alongside the sparse graph links. Per-device
        eigensolver budgets also need to reserve these arrays.
        """
        n, d = graph.norb, graph.dimension
        if d > np.iinfo(np.int32).max:
            raise ValueError("CSF action indices exceed the int32 address range")
        available = workspace_bytes(max_memory, device="cpu")
        # Conservative host estimates for Python path and occupation lookup
        # entries, and for staging three FP64 pair-by-CSF action buffers.
        planned = d * (384 + 112 * n) + 24 * n * n * d
        if planned > available:
            raise ValueError("CSF path and action arrays exceed the host memory budget")
        # Each edge is assembled in four Python lists before packing to NumPy.
        edge_limit = min((available - planned) // 192, np.iinfo(np.int32).max)
        occupation, groups, paths, spins = _path_data(graph)
        try:
            links, pair_offsets = _build_links(graph, groups, paths, spins, edge_limit)
            alpha = None
            if graph.nelec[0] != graph.nelec[1]:
                alpha, _ = _build_links(graph, groups, paths, spins,
                                         edge_limit - len(links.pair), alpha_only=True)
        finally:
            _segment.cache_clear()
        return cls(graph, occupation, links, pair_offsets, alpha)

    @property
    def norb(self):
        return self.graph.norb

    @property
    def nelec(self):
        return self.graph.nelec

    @property
    def spin2(self):
        return self.graph.spin2

    @property
    def size(self):
        return self.graph.dimension

    @property
    def storage_bytes(self):
        """Byte count of resident occupation and sparse link arrays."""
        alpha = () if self._alpha is None else self._alpha
        return self._occupation.nbytes + self._pair_offsets.nbytes + sum(
            x.nbytes for x in (*self._links, *alpha)
        )

    def _pair_tiles(self, width):
        starts = self._pair_offsets[:-1:width].copy()
        counts = np.diff(starts, append=self._pair_offsets[-1])
        capacity = int(counts.max(initial=0))
        return starts, capacity

    def action_workspace_bytes(self, dtype, columns=1, *, symmetric=False, pair_block_size=None):
        """Estimate ordered or real symmetric orbital-pair action workspace."""
        n = self.norb
        pairs = n * (n + 1) // 2 if symmetric else n**2
        if pair_block_size is not None and (not symmetric or pair_block_size < 1):
            raise ValueError("Pair tiles require a positive symmetric pair block size")
        width = pairs if pair_block_size is None else min(pair_block_size, pairs)
        edge_scratch = 3 * len(self._links.pair)
        if width < pairs:
            starts, capacity = self._pair_tiles(width)
            # The input and output can each slice up to this many edges.
            edge_scratch = 6 * capacity + len(starts)
        per_column = 4 * width * self.size + edge_scratch + 2 * self.size
        # The ERI block and trace are common to all CI columns.
        common = pairs + (2 * width**2 + n**3 if symmetric else 0)
        return np.dtype(dtype).itemsize * (columns * per_column + common)

    def symmetric_pair_block_size(self, dtype, available_bytes, *, columns=1):
        """Choose a static pair tile from a caller's remaining action budget."""
        if self.action_workspace_bytes(dtype, columns, symmetric=True) <= available_bytes:
            return None
        pair_count = self.norb * (self.norb + 1) // 2
        for width in range(pair_count - 1, 0, -1):
            if self.action_workspace_bytes(
                dtype, columns, symmetric=True, pair_block_size=width
            ) <= available_bytes:
                return width
        raise ValueError("CSF orbital pair action exceeds the working-memory budget")

    def _vector(self, c):
        c = jnp.asarray(c)
        if c.shape != (self.size,) or jnp.iscomplexobj(c):
            raise ValueError("c must be a real vector in CSF coordinates")
        return c.astype(jnp.result_type(c, 1.0))

    def _integrals(self, h, eri):
        h, eri = jnp.asarray(h), jnp.asarray(eri)
        n = self.norb
        if h.shape != (n, n) or eri.shape != (n,) * 4:
            raise ValueError("h and eri must have active (norb,norb) and (norb,)*4 shapes")
        if jnp.iscomplexobj(h) or jnp.iscomplexobj(eri):
            raise ValueError("CSF operators require real integrals")
        dtype = jnp.result_type(h, eri, 1.0)
        return h.astype(dtype), eri.astype(dtype)

    def _generator_vectors(self, c, pair, diagonal, pair_count):
        _, target, source, weight = self._links
        values = jnp.zeros((pair_count, self.size), c.dtype)
        values = values.at[diagonal, :].set(
            self._occupation * c, indices_are_sorted=True, unique_indices=True
        )
        return values.at[pair, target].add(jnp.asarray(weight, c.dtype) * c[source])

    def one_body_vectors(self, c):
        """All spin-free generator actions E_pq c, indexed by (p, q, CSF)."""
        c = self._vector(c)
        n = self.norb
        diagonal = jnp.arange(n) * (n + 1)
        values = self._generator_vectors(c, self._links.pair, diagonal, n * n)
        return values.reshape(n, n, self.size)

    def _transpose_generators(self, values, pair, diagonal):
        _, target, source, weight = self._links
        result = jnp.einsum("pd,pd->d", self._occupation, values[diagonal])
        update = values[pair, target]
        return result.at[source].add(jnp.asarray(weight, values.dtype) * update)

    def apply_one_body(self, h, c):
        """Apply sum_pq h[p,q] E_pq."""
        return jnp.einsum("pq,pqd->d", h, self.one_body_vectors(c))

    def matvec(self, h1, eri, c):
        """Apply the physical spin-free Hamiltonian with chemists' ERIs."""
        h1, eri = self._integrals(h1, eri)
        if not self.norb:
            return jnp.zeros_like(self._vector(c), dtype=jnp.result_type(h1, c))
        actions = self.one_body_vectors(c)
        effective = h1 - 0.5 * jnp.einsum("pqqs->ps", eri)
        weighted = 0.5 * jnp.einsum("pqrs,rsd->qpd", eri, actions)
        n = self.norb
        diagonal = jnp.arange(n) * (n + 1)
        back = self._transpose_generators(weighted.reshape(n * n, self.size),
                                          self._links.pair, diagonal)
        return jnp.einsum("pq,pqd->d", effective, actions) + back

    def matvec_symmetric(self, h1, eri, c, *, pair_block_size=None):
        """Apply the real Hermitian Hamiltonian in triangular orbital pairs.

        Off-diagonal pairs use S_pq = E_pq + E_qp. The four ERI permutations
        and pair exchange are projected explicitly so derivatives with respect
        to full integrals are independent of the chosen triangle. Small static
        tiles bound the pair-by-CSF intermediates and recompute them in reverse
        differentiation rather than retaining a stack.
        """
        action = self._prepare_symmetric_action(h1, eri, pair_block_size=pair_block_size)
        return action(c)

    def _prepare_symmetric_action(self, h1, eri, *, pair_block_size=None):
        """Stage the differentiable, CI-vector-independent work for one Hamiltonian."""
        h1, eri = self._integrals(h1, eri)
        n = self.norb
        if pair_block_size is not None and (
            isinstance(pair_block_size, bool) or not isinstance(pair_block_size, Integral)
            or pair_block_size < 1
        ):
            raise ValueError("Symmetric pair block size must be a positive integer")
        if not n:
            def empty(c):
                c = self._vector(c)
                return jnp.zeros_like(c, dtype=jnp.result_type(h1, c))

            return empty
        p, q = np.tril_indices(n)
        orbital_pair = np.empty((n, n), dtype=np.int32)
        orbital_pair[p, q] = orbital_pair[q, p] = np.arange(p.size, dtype=np.int32)
        pair_lookup = orbital_pair.reshape(-1)
        if pair_block_size is not None and pair_block_size < p.size:
            # Keep the ERI projection tiled. Only the trace is needed outside
            # the action, so no full pair-by-pair array is made in this branch.
            orbital = jnp.arange(n)
            row, middle, column = orbital[:, None, None], orbital[None, :, None], orbital[None, None, :]
            trace = _pair_symmetrized_eri(eri, row, middle, middle, column).sum(axis=1)
            effective = .5 * (h1 + h1.T) - .25 * (trace + trace.T)

            def apply_tiled(c):
                return self._matvec_pair_tiles(
                    effective, eri, self._vector(c), p, q, pair_lookup, pair_block_size
                )

            return apply_tiled

        pair = pair_lookup[self._links.pair]
        integrals = _pair_symmetrized_eri(eri, p[:, None], q[:, None], p, q)
        integrals = .5 * (integrals + integrals.T)
        diagonal = np.diag(orbital_pair)
        trace = integrals[orbital_pair[:, :, None], orbital_pair[None, :, :]].sum(axis=1)
        effective = (.5 * (h1 + h1.T - trace))[p, q]

        def apply(c):
            actions = self._generator_vectors(self._vector(c), pair, diagonal, p.size)
            weighted = .5 * (integrals @ actions)
            return effective @ actions + self._transpose_generators(weighted, pair, diagonal)

        return apply

    def _matvec_pair_tiles(self, effective, eri, c, p, q, pair_lookup, width):
        dtype = jnp.result_type(effective, eri, c)
        links = _Links(*map(jnp.asarray, self._links))
        occupation = jnp.asarray(self._occupation)
        pair_lookup = jnp.asarray(pair_lookup)
        starts, capacity = self._pair_tiles(width)
        starts = jnp.asarray(starts)
        pair_count = len(p)
        p, q = jnp.asarray(p), jnp.asarray(q)
        offsets = jnp.arange(width)

        def tile_pairs(index):
            ids = index * width + offsets
            clipped = jnp.minimum(ids, pair_count - 1)
            return p[clipped], q[clipped], ids < pair_count

        def tile_edges(index):
            # Dynamic slices have a fixed capacity. A short final tile can
            # overlap its predecessor; the local pair check masks padding.
            fetch = jnp.minimum(starts[index], len(self._links.pair) - capacity)
            pair, target, source, weight = (
                jax.lax.dynamic_slice_in_dim(value, fetch, capacity) for value in links
            )
            local = pair_lookup[pair] - index * width
            inside = (0 <= local) & (local < width)
            return local, inside, target, source, weight.astype(dtype)

        @jax.checkpoint
        def accumulate(index, sigma):
            out_p, out_q, valid = tile_pairs(index)
            left_p, left_q = out_p[:, None], out_q[:, None]

            @jax.checkpoint
            def contract(index, weighted):
                in_p, in_q, valid_in = tile_pairs(index)
                local, inside, target, source, weight = tile_edges(index)
                diag_in = valid_in & (in_p == in_q)
                actions = jnp.where(diag_in[:, None], occupation[in_p] * c, 0.).astype(dtype)
                actions = actions.at[jnp.where(inside, local, width), target].add(
                    weight * c[source], mode="drop"
                )
                forward = _pair_symmetrized_eri(eri, left_p, left_q, in_p, in_q)
                exchange = _pair_symmetrized_eri(eri, in_p, in_q, left_p, left_q)
                return weighted + .25 * ((forward + exchange) @ actions)

            seed = effective[out_p, out_q, None] * c
            weighted = jax.lax.fori_loop(0, len(starts), contract, seed.astype(dtype))
            diagonal = jnp.where((valid & (out_p == out_q))[:, None], weighted, 0.)
            sigma = sigma + jnp.einsum("pd,pd->d", occupation[out_p], diagonal)
            local, inside, target, source, weight = tile_edges(index)
            update = weighted[jnp.clip(local, 0, width - 1), target]
            return sigma.at[source].add(jnp.where(inside, weight * update, 0.))

        return jax.lax.fori_loop(0, len(starts), accumulate, jnp.zeros(self.size, dtype=dtype))

    def matmat(self, h1, eri, coefficients):
        """Apply the Hamiltonian to column vectors."""
        return jax.vmap(lambda c: self.matvec(h1, eri, c), in_axes=1, out_axes=1)(coefficients)

    def diagonal(self, h1, eri):
        """Exact Davidson diagonal from projected occupation and hop norms."""
        h1, eri = self._integrals(h1, eri)
        dtype = jnp.result_type(h1, eri)
        if not self.norb:
            return jnp.zeros(self.size, dtype)
        occ = self._occupation
        effective = h1 - 0.5 * jnp.einsum("pqqs->ps", eri)
        diag = jnp.einsum("pp,pd->d", effective, occ)
        diag += 0.5 * jnp.einsum("pprr,pd,rd->d", eri, occ, occ)
        pair, _, source, weight = self._links
        p, q = divmod(pair, self.norb)
        update = 0.5 * eri[q, p, p, q] * jnp.square(jnp.asarray(weight, dtype))
        return diag.at[source].add(update)

    def rdm12(self, c):
        """Spin-summed 1-RDM and ordered 2-RDM in E_pq/e_pq,rs convention."""
        c = self._vector(c)
        actions = self.one_body_vectors(c)
        rdm1 = jnp.einsum("d,pqd->pq", c, actions)
        rdm2 = jnp.einsum("qpd,rsd->pqrs", actions, actions)
        rdm2 -= jnp.einsum("qr,ps->pqrs", jnp.eye(self.norb, dtype=c.dtype), rdm1)
        return rdm1, rdm2

    def rdm1s(self, c):
        """Alpha and beta 1-RDMs in the graph's chosen M_S sector."""
        c = self._vector(c)
        total = jnp.einsum("d,pqd->pq", c, self.one_body_vectors(c))
        if self._alpha is None:
            alpha = 0.5 * total
        else:
            pair, target, source, weight = self._alpha
            update = jnp.asarray(weight, c.dtype) * c[target] * c[source]
            alpha = jnp.zeros(self.norb**2, c.dtype).at[pair].add(update)
            alpha = alpha.reshape(self.norb, self.norb)
        return alpha, total - alpha


def make_csf_space(norb, nelecas, spin2, *, max_memory=DEFAULT_MAX_MEMORY_MB):
    """Build the fixed-spin DRT and its static spin-coupled operator links."""
    return CSFOperators.build(CSFGraph.build(norb, nelecas, spin2), max_memory=max_memory)
