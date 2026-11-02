"""Separate parameter quantization from SCF and finite-difference truncation."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from flax import nnx
import numpy as np

from gradscf.data.molecule import atomic_number
from gradscf.model.nnao import MACEBasisModel, build_graph, prepare_direct_basis
from gradscf.model.nnao.config import MODEL_CONFIG
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.runtime import atomic_json, cache_location, verify_mace_source

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
parent = root/'artifacts/fixed-pool200-restart-20261012/archive/last.npz'
with np.load(parent, allow_pickle=False) as z:
    theta = jnp.asarray(z['parameters'])
    metadata = json.loads(str(z['metadata'].item()))
identity = metadata['identity']
verify_mace_source()
model = MACEBasisModel(**MODEL_CONFIG, rngs=nnx.Rngs(0))
definition, params, other = nnx.split(model, nnx.Param, ...)
initial, old_unravel = ravel_pytree(params)
promoted = jax.tree.map(lambda x: x.astype(jnp.float64), params)
initial64, new_unravel = ravel_pytree(promoted)
np.testing.assert_array_equal(initial, initial64)
leaves = jax.tree_util.tree_flatten_with_path(params)[0]
report = dict(parent_epoch=metadata['epoch'], flat_dtype=str(initial.dtype),
    parameter_leaf_dtypes=dict(Counter(str(x.dtype) for _, x in leaves)),
    float32_parameters=[dict(path=str(path), shape=list(x.shape), count=int(x.size))
                        for path, x in leaves if x.dtype == jnp.float32],
    initial_flat_values_unchanged=True,
    initial_sha256=hashlib.sha256(np.asarray(initial).tobytes()).hexdigest(), records=[])
atomic_json(out/'diagnosis.json', report)
print('DTYPES', report['parameter_leaf_dtypes'], flush=True)
direction = np.random.default_rng(0).normal(size=theta.shape)
direction /= np.linalg.norm(direction)
predictors = {name: jax.jit(lambda x, graph, unravel=unravel:
    nnx.merge(definition, unravel(x), other)(graph))
    for name, unravel in [('mixed', old_unravel), ('float64', new_unravel)]}
for identifier in ['INV24/Ether', 'PX13/hf_4_ts', 'S22/08a']:
    row = next(r for r in identity['references']['structures'] if r['structure_id'] == identifier)
    with jax.default_device(jax.devices('cpu')[0]):
        layout = prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
            cart=False, basis_family='szp663_direct', core_primitives=6)
        layout = bind_pool_candidate(layout, identity['candidate'])
    experiment, _ = prepare_fixed_pool(row, layout,
        cache_location(root/'artifacts/fixed-pool200-20261012/cache', row), storage='mmap')
    graph = build_graph([atomic_number(s) for s in row['symbols']], row['coords_angstrom'],
                        element_order=MODEL_CONFIG['elements'])
    reference = None
    for name, predict in predictors.items():
        coefficients, tangent = jax.jvp(lambda x: predict(x, graph), (theta,), (jnp.asarray(direction),))
        energy, dc, info = experiment.evaluate_device(coefficients)
        ad = float(jnp.vdot(dc, tangent))
        record = dict(structure_id=identifier, parameterization=name, energy=energy,
                      analytic=ad, checks=[])
        if reference is None:
            reference = energy, coefficients
        record['energy_change_from_mixed'] = energy-reference[0]
        record['max_coefficient_change_from_mixed'] = float(jnp.max(jnp.abs(coefficients-reference[1])))
        for step in [1e-3, 3e-4, 1e-4, 3e-5]:
            cp, cm = predict(theta+step*direction, graph), predict(theta-step*direction, graph)
            mace_fd = float(jnp.vdot(dc, (cp-cm)/(2*step)))
            ep = experiment.evaluate_device(cp)[0]
            em = experiment.evaluate_device(cm)[0]
            full_fd = (ep-em)/(2*step)
            record['checks'].append(dict(step=step, mace_fd=mace_fd, full_fd=full_fd,
                mace_error=abs(mace_fd-ad), full_error=abs(full_fd-ad)))
        step = 1e-4
        ep = experiment.evaluate_device(coefficients+step*tangent)[0]
        em = experiment.evaluate_device(coefficients-step*tangent)[0]
        record['coefficient_only_fd'] = (ep-em)/(2*step)
        record['coefficient_only_error'] = abs(record['coefficient_only_fd']-ad)
        report['records'].append(record)
        atomic_json(out/'diagnosis.json', report)
        print(json.dumps(record), flush=True)
