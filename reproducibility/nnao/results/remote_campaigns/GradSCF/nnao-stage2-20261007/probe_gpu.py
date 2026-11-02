"""Measure representative stage2 RHF/implicit evaluation using packed DF."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import resource

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from validate_nnao_pilot import atomic_json, cache_location, file_hash, numerical_runtime, read_structures

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--structures', type=Path, required=True)
parser.add_argument('--cache-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--id', required=True)
args = parser.parse_args()
jax, experiment_class = numerical_runtime('cuda')
row = read_structures(args.structures, ids=[args.id])[0]
path = cache_location(args.cache_dir, row)
manifest = json.loads(path.with_suffix('.json').read_text())
assert file_hash(path) == manifest['cache_sha256']
experiment = experiment_class(geometry=row, basis_family='szp663_direct',
    core_primitives=6, jk_backend='df', integral_cache=path,
    implicit_tolerance=1e-8, cache_storage='host')
outputs = experiment.layout.reference_outputs()
timings = []
for iteration in range(3):
    tick = time.perf_counter()
    energy, gradient, info = experiment.evaluate(outputs)
    timings.append(time.perf_counter() - tick)
report = dict(structure_id=args.id, devices=[str(d) for d in jax.devices()],
    device_kinds=[d.device_kind for d in jax.devices()], dtype=str(gradient.dtype),
    energy_hartree=energy, diagnostics=info, elapsed_seconds=timings,
    primitive_nao=experiment.primitive_topology.nao,
    nao=experiment.layout.topology.nao, df_shape=list(experiment.rep.shape),
    packed_df_bytes=int(experiment.rep.nbytes), cache_storage='host',
    implicit_tolerance=experiment.implicit_tolerance,
    host_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    device_memory_stats=jax.devices()[0].memory_stats(),
    cache_sha256=manifest['cache_sha256'])
atomic_json(args.output, report)
print(json.dumps(report), flush=True)
