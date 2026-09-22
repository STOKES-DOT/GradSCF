# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Canonical real UCCSD(T), adapted from PySCF 2.9.0 gccsd_t.kernel.

Spin-orbital formula: J. Chem. Phys. 98, 8718 (1993), doi:10.1063/1.464480.
Virtual triples are streamed, retaining only occupied-cube intermediates.
See NOTICE.md for software attribution; no PySCF runtime is used.
"""
from itertools import combinations
from math import isfinite

import jax
import jax.numpy as jnp

from .uccsd import prepare_ucc_integrals
from .spin_amplitudes import SpinAmplitudeSpace
from ._spin_equations import residual
from .ground import METHODS
from .types import TriplesResult


def evaluate_ucc_triples(h1, eri, result, *, nocc, frozen=None,
                         denominator_tol=1e-10, max_virtual_triples=20000,
                         variant="ccsd(t)", residual_tol=1e-8, canonical_tol=1e-7):
    if variant != "ccsd(t)":
        raise NotImplementedError("Unrestricted triples currently implement only CCSD(T)")
    if any(not isfinite(x) or x <= 0 for x in (denominator_tol, residual_tol, canonical_tol)):
        raise ValueError("Triples tolerances must be finite and positive")
    if not isinstance(max_virtual_triples, int) or max_virtual_triples < 1:
        raise ValueError("max_virtual_triples must be a positive integer")
    ints, occupied, virtual = prepare_ucc_integrals(h1, eri, nocc=nocc, frozen=frozen)
    space = SpinAmplitudeSpace(occupied, virtual)
    t1, t2 = space.from_blocks(result.t1, result.t2, ints.fock.dtype)
    symmetry_error = jnp.asarray(0., dtype=t1.dtype)
    for block in (result.t2[0], result.t2[2]):
        block = jnp.asarray(block)
        symmetry_error = jnp.maximum(symmetry_error, jnp.max(jnp.abs(block+block.swapaxes(0, 1)), initial=0.))
        symmetry_error = jnp.maximum(symmetry_error, jnp.max(jnp.abs(block+block.swapaxes(2, 3)), initial=0.))
    norm = jnp.max(jnp.abs(space.pack(*residual(t1, t2, ints))), initial=0.)
    eps = jnp.diag(ints.fock)
    canonical_error = jnp.max(jnp.abs(ints.fock-jnp.diag(eps)), initial=0.)
    state_valid = (result.converged & (result.method_id == METHODS.index("ccsd"))
                   & (norm <= residual_tol) & (symmetry_error <= residual_tol)
                   & (canonical_error <= canonical_tol))

    def assemble(connected, singles, minimum):
        valid = state_valid & (minimum > denominator_tol) & jnp.isfinite(connected+singles)
        return TriplesResult(jnp.where(valid, connected+singles, jnp.nan),
            jnp.where(valid, connected, jnp.nan), jnp.where(valid, singles, jnp.nan),
            minimum, norm, canonical_error, valid, jnp.asarray(1, dtype=jnp.int32))

    no, nv = t1.shape
    zero = jnp.asarray(0., dtype=t1.dtype)
    if no < 3 or nv < 3:
        return assemble(zero, zero, jnp.asarray(jnp.inf, dtype=t1.dtype))
    if nv*(nv-1)*(nv-2)//6 > max_virtual_triples:
        raise ValueError("Virtual triples exceed max_virtual_triples")
    combos = jnp.asarray([(a, b, c) for c, b, a in combinations(range(nv), 3)])
    so = jnp.asarray([0]*occupied[0]+[1]*occupied[1])
    sv = jnp.asarray([0]*virtual[0]+[1]*virtual[1])
    indices = jnp.arange(no)
    distinct = ((indices[:, None, None] != indices[None, :, None])
                & (indices[:, None, None] != indices[None, None, :])
                & (indices[None, :, None] != indices[None, None, :]))
    spin_sum = so[:, None, None]+so[None, :, None]+so[None, None, :]
    eo, ev = eps[:no], eps[no:]
    eijk = eo[:, None, None]+eo[None, :, None]+eo[None, None, :]
    bcei = ints.ovvv.transpose(3, 2, 1, 0)
    majk = ints.ooov.transpose(2, 3, 0, 1)
    bcjk = ints.oovv.transpose(2, 3, 0, 1)
    doubles, singles = t2.transpose(2, 3, 0, 1), t1.T
    fvo = ints.fock[no:, :no]

    def wv(a, b, c):
        w = (jnp.einsum("ejk,ei->ijk", doubles[a], bcei[b, c])
             -jnp.einsum("im,mjk->ijk", doubles[b, c], majk[:, a]))
        v = (jnp.einsum("i,jk->ijk", singles[a], bcjk[b, c])
             +jnp.einsum("i,jk->ijk", fvo[a], doubles[b, c]))
        return w+w.transpose(2, 0, 1)+w.transpose(1, 2, 0), w, v

    def body(index, carry):
        connected, disconnected, minimum = carry
        a, b, c = combos[index]
        w0, v0, s0 = wv(a, b, c)
        w1, v1, s1 = wv(c, a, b)
        w2, v2, s2 = wv(b, a, c)
        d = eijk-ev[a]-ev[b]-ev[c]
        allowed = distinct & (spin_sum == sv[a]+sv[b]+sv[c])
        minimum = jnp.minimum(minimum, jnp.min(jnp.where(allowed, jnp.abs(d), jnp.inf)))
        w = jnp.where(allowed, w0+w1-w2, 0.) / jnp.where(jnp.abs(d) > denominator_tol, d, 1.)
        return (connected+.5*jnp.sum(w*(v0+v1-v2)),
                disconnected+.5*jnp.sum(w*(s0+s1-s2)), minimum)

    return assemble(*jax.lax.fori_loop(0, len(combos), body,
                                      (zero, zero, jnp.asarray(jnp.inf, dtype=t1.dtype))))
