"""CUDA and MACE forward probe, to be run inside the pilot Slurm allocation."""
import json
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from gradscf.model.nnao import MACEBasisModel, build_graph
import gradscf
import mace_jax

source_root = Path(__file__).resolve().parents[2] / 'src'
assert source_root.resolve() in Path(gradscf.__file__).resolve().parents, gradscf.__file__
assert (source_root / 'gradscf/model/nnao').resolve() in Path(mace_jax.__file__).resolve().parents, mace_jax.__file__

jax.config.update('jax_enable_x64', True)
devices = jax.devices()
assert devices and all(d.platform == 'gpu' for d in devices), devices
x = jnp.eye(32, dtype=jnp.float64)
y = jax.jit(lambda a: a @ a)(x)
y.block_until_ready()
assert y.dtype == jnp.float64
model = MACEBasisModel(elements=(1,6,7,8,9), channels=8, num_interactions=2,
    max_ell=1, correlation=2, zero_init=True, basis_family='szp663_direct', rngs=nnx.Rngs(0))
graph = build_graph([1,1], [[0.,0.,0.],[0.,0.,.74]], element_order=model.elements)
outputs = model(graph)
outputs.block_until_ready()
assert np.isfinite(np.asarray(outputs)).all()
report = dict(jax_version=jax.__version__, devices=[str(d) for d in devices],
    device_kinds=[d.device_kind for d in devices], dtype=str(y.dtype),
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    slurm_job_id=os.environ.get('SLURM_JOB_ID'), mace_output_shape=list(outputs.shape),
    mace_module=mace_jax.__file__, gradscf_module=gradscf.__file__,
    pythonpath=os.environ.get('PYTHONPATH'))
Path(sys.argv[1]).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
