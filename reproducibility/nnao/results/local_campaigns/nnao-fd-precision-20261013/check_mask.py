"""Verify immutable MACE activation scales explain the FD discrepancy."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
from flax import nnx
from jax.flatten_util import ravel_pytree
import numpy as np
from gradscf.data.molecule import atomic_number
from gradscf.model.nnao import MACEBasisModel, build_graph, prepare_direct_basis
from gradscf.model.nnao.config import MODEL_CONFIG
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.runtime import atomic_json, cache_location
from gradscf.model.nnao.training.alternating import trainable_parameter_mask

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
with np.load(root/'artifacts/fixed-pool200-restart-20261012/archive/last.npz', allow_pickle=False) as z:
    theta, accepted_gradient = jnp.asarray(z['parameters']), np.array(z['gradient'])
    metadata = json.loads(str(z['metadata'].item()))
model = MACEBasisModel(**MODEL_CONFIG, rngs=nnx.Rngs(0))
definition, params, other = nnx.split(model, nnx.Param, ...)
initial, unravel = ravel_pytree(params)
mask = trainable_parameter_mask(params)
predict = jax.jit(lambda x, graph: nnx.merge(definition, unravel(x), other)(graph))
direction = np.random.default_rng(0).normal(size=theta.shape)
direction /= np.linalg.norm(direction)
report = dict(frozen_indices=np.flatnonzero(~mask).tolist(),
    frozen_metadata=[dict(path=str(path), is_mutable=False)
        for path, variable in params.flat_state() if not variable.get_metadata().get('is_mutable', True)],
    frozen_checkpoint_gradient=accepted_gradient[~mask].tolist(), records=[])
print('FROZEN', report['frozen_indices'], report['frozen_checkpoint_gradient'], flush=True)
identity = metadata['identity']
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
    coefficients = predict(theta, graph)
    energy, dc, _ = experiment.evaluate_device(coefficients)
    for label, d in [('all', direction), ('trainable', direction*mask), ('immutable', direction*(~mask))]:
        tangent = jax.jvp(lambda x: predict(x, graph), (theta,), (jnp.asarray(d),))[1]
        ad = float(jnp.vdot(dc, tangent))
        h = 1e-4
        plus = experiment.evaluate_device(predict(theta+h*d, graph))[0]
        minus = experiment.evaluate_device(predict(theta-h*d, graph))[0]
        fd = (plus-minus)/(2*h)
        record = dict(structure_id=identifier, direction=label, ad=ad, fd=fd,
            absolute_error=abs(ad-fd), step=h, passed=bool(np.isclose(ad, fd, atol=2e-6, rtol=2e-5)))
        report['records'].append(record)
        print(json.dumps(record), flush=True)
    atomic_json(out/'mask-diagnosis.json', report)
