"""Record GPU allocator limits inside a Slurm allocation without large inputs."""
import json
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
jnp.ones((2, 2), dtype=jnp.float64).block_until_ready()
device = jax.devices()[0]
report = dict(backend=jax.default_backend(), device=str(device), kind=device.device_kind,
    jax_version=jax.__version__, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
    environment={name: os.environ.get(name) for name in (
        'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_CLIENT_MEM_FRACTION',
        'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_ALLOCATOR', 'TF_GPU_ALLOCATOR')},
    memory_stats=device.memory_stats())
Path(sys.argv[1]).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
