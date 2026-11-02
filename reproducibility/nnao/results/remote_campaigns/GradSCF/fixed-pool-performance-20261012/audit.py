"""Ablate transfer/fusion and equivalent analytic stability curvature."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np
from gradscf.data.molecule import atomic_number
from gradscf.scf import stability
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.config import MODEL_CONFIG
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.runtime import atomic_json, cache_location
from gradscf.model.nnao.training.objective import _df_rhf_solver
from gradscf.model.nnao.training.trainer import make_mace_predictor

out = Path(__file__).resolve().parent
source = out.parent/'fixed-pool200-20261012'
def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, out/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
old_stability = load('gradscf.scf.stability_baseline', 'stability_baseline.py')
old_pool = load('gradscf.model.nnao.training.pool_baseline', 'pool_baseline.py')
optimized_stability = stability.stabilize_df_rhf
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
    graph = build_graph([atomic_number(s) for s in row['symbols']], row['coords_angstrom'],
                        element_order=MODEL_CONFIG['elements'])
    reference = None
    for variant in ('baseline', 'fusion_only', 'optimized'):
        stability.stabilize_df_rhf = (optimized_stability if variant == 'optimized'
                                      else old_stability.stabilize_df_rhf)
        _df_rhf_solver.cache_clear()
        exp, manifest = prepare_fixed_pool(row, layout, cache_location(source/'cache', row), storage='mmap')
        if variant == 'baseline':
            exp = old_pool.FreeContractionRHF(layout, exp.raw, exp.enuc, exp.value,
                exp.gradient, identifier, exp.diagnostics)
        times = []
        for iteration in range(4):
            start = time.perf_counter()
            c = predict(theta, graph)
            evaluate = exp.evaluate if variant == 'baseline' else exp.evaluate_device
            value, dc, info = evaluate(c)
            gradient = np.asarray(backward(theta, graph, dc))
            elapsed = time.perf_counter()-start
            if reference is None:
                reference = value, gradient, info['minimum_curvature']
            np.testing.assert_allclose(value, reference[0], atol=1e-8, rtol=0)
            np.testing.assert_allclose(gradient, reference[1], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(info['minimum_curvature'], reference[2], atol=1e-7, rtol=1e-6)
            times.append(elapsed)
            print(identifier, variant, iteration, elapsed, flush=True)
        records.append(dict(structure_id=identifier, variant=variant, energy=value,
            gradient_max_error=float(np.max(np.abs(gradient-reference[1]))),
            curvature_error=abs(info['minimum_curvature']-reference[2]),
            cold_seconds=times[0], hot_seconds=times[1:],
            hot_median_seconds=float(np.median(times[1:]))))
        atomic_json(out/os.environ.get('AUDIT_RESULTS', 'audit-results.json'), dict(records=records,
            devices=[d.device_kind for d in jax.devices()], jax=jax.__version__))
    del exp
    jax.clear_caches()
