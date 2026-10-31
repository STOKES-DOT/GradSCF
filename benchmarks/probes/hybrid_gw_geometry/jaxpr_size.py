"""Metadata-only shared-Jaxpr probe; this runs no electronic-structure kernel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--depth", type=int, default=16)
args = parser.parse_args()
assert 0 <= args.depth <= 20
os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
import jax
import jax.numpy as jnp
from gradscf import _memory

jax.config.update("jax_enable_x64", True)
function = jax.jit(lambda x: jnp.sin(x)+x)
def lift(previous):
    return jax.jit(lambda x: previous(x)+previous(-x))
for _ in range(args.depth):
    function = lift(function)
program = jax.make_jaxpr(function)(jax.ShapeDtypeStruct((3,), jnp.float64))
samples, counts = [], []
for _ in range(3):
    started = perf_counter()
    counts.append(_memory.jaxpr_temporary_bytes(program))
    samples.append(perf_counter()-started)
assert len(set(counts)) == 1
source = Path(_memory.__file__).resolve()
payload = dict(scope=__doc__, depth=args.depth, jax_version=jax.__version__,
    dtype="float64", vector_shape=[3], samples_seconds=samples,
    declared_bytes=counts[0], source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    affinity=sorted(os.sched_getaffinity(0)),
    notes="Three independent size estimates of one immutable JAX graph; no cross-call cache.")
args.output.write_text(json.dumps(payload, indent=2)+"\n")
print(json.dumps(payload, indent=2), flush=True)
