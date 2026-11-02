"""Paired fixed-pool execution timings; no SCF or optimizer policy changes."""
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.data.molecule import atomic_number
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.config import MODEL_CONFIG
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.runtime import atomic_json, cache_location
from gradscf.model.nnao.training.objective import BasisRHF
from gradscf.model.nnao.training.trainer import make_mace_predictor

root = Path(__file__).resolve().parents[2]
source = root/'artifacts/fixed-pool200-20261012'
out = Path(__file__).resolve().parent
rows = json.loads((source/'references.json').read_text())['structures']
candidate = json.loads((source/'candidate-exponents.json').read_text())
theta, predict, backward = make_mace_predictor(0, MODEL_CONFIG)
records = []
for identifier in ('S22/08a', 'S66/18', 'MCONF/44'):
    row = next(r for r in rows if r['structure_id'] == identifier)
    with jax.default_device(jax.devices('cpu')[0]):
        layout = prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
                                      cart=False, basis_family='szp663_direct', core_primitives=6)
        layout = bind_pool_candidate(layout, candidate)
    exp, manifest = prepare_fixed_pool(row, layout, cache_location(source/'cache', row), storage='mmap')
    raw = exp.raw
    graph = build_graph([atomic_number(s) for s in row['symbols']], row['coords_angstrom'],
                        element_order=MODEL_CONFIG['elements'])

    def composed(c, s, h, b):
        return exp.value(exp.project(c, s, h, b), exp.enuc)
    fused = jax.jit(jax.value_and_grad(composed, argnums=0, has_aux=True))

    def evaluate(variant):
        c = predict(theta, graph)
        if variant == 'original':
            exp.raw = raw
            value, dc, info = exp.evaluate(c)
        elif variant == 'reuse_device':
            exp.raw = jax.device_put(raw)
            value, dc, info = exp.evaluate(c)
        else:
            arrays = resident if variant == 'fused_resident' else jax.device_put(raw)
            (value, info), dc = fused(c, *arrays)
            info = BasisRHF._checked_info(value, info)
            value = float(value)
        gradient = np.asarray(backward(theta, graph, dc))
        return value, gradient, info

    reference = None
    for variant in ('original', 'reuse_device', 'fused', 'fused_resident'):
        if variant == 'fused_resident':
            resident = jax.block_until_ready(jax.device_put(raw))
        times = []
        for iteration in range(4):
            start = time.perf_counter()
            value, grad, info = evaluate(variant)
            elapsed = time.perf_counter()-start
            if reference is None:
                reference = value, grad
            np.testing.assert_allclose(value, reference[0], atol=1e-8, rtol=0)
            np.testing.assert_allclose(grad, reference[1], atol=1e-6, rtol=1e-5)
            times.append(elapsed)
            print(identifier, variant, iteration, elapsed, flush=True)
        records.append(dict(structure_id=identifier, variant=variant,
            cache_gib=manifest['bytes']/2**30, cold_seconds=times[0], hot_seconds=times[1:],
            hot_median_seconds=float(np.median(times[1:])), energy=value,
            gradient_max_error=float(np.max(np.abs(grad-reference[1])))))
        atomic_json(out/'probe-results.json', dict(records=records,
            devices=[d.device_kind for d in jax.devices()], jax=jax.__version__))
    del resident, exp, raw, fused
    jax.clear_caches()
