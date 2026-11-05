"""Shared-input RKS XC macro probe; no molecular SCF/ERI/RI construction.
Slurm only. Raw/cached cold+3warm use one native atom-guess density. The
600 s guard includes imports/preparation; cold action includes JIT compile.
Cache preparation is timed separately. This is not a converged workflow.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, help='Existing frozen RKS/PBE config; avoids catalog dependencies')
    parser.add_argument('--system', default='beta-carotene')
    parser.add_argument('--basis', default='def2-svp')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--reserved-factor-bytes', type=int, default=19_279_000_000)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('This numerical probe requires Slurm')
    if args.reserved_factor_bytes < 0:
        parser.error('Reserved capacity must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=False)
    started, events, lock = time.perf_counter(), [], threading.Lock()
    def emit(stage, **values):
        row = dict(stage=stage, elapsed_seconds=time.perf_counter()-started,
            ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024), **values)
        with lock:
            events.append(row)
            (args.output/'events.json').write_text(json.dumps(events, indent=2, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)
    def timeout():
        emit('timeout', scope='whole probe including imports', budget_seconds=600)
        os._exit(124)
    timer = threading.Timer(600, timeout)
    timer.daemon = True
    timer.start()
    frozen = args.config or ROOT/'reports/scf_100/20261105-40a569bf/plan-v2/configs'/f'{args.system}-{args.basis}-rks-pbe-df-gradscf.json'
    if args.config is not None and not frozen.exists():
        parser.error('Explicit frozen config does not exist')
    if frozen.exists():
        cfg = json.loads(frozen.read_text())
    else:
        from benchmarks.probes.scf_100.make_plan import build_cases
        cfg = next(c for c in build_cases(ROOT, args.system, args.basis, ('gradscf',))[0]
                   if c['method']=='RKS' and c['xc']=='pbe' and c['representation']=='df')
    if (cfg['method'], cfg['xc'], cfg['engine']) != ('RKS', 'pbe', 'gradscf'):
        parser.error('This shared-input probe requires an RKS/PBE configuration')
    threads = cfg['threads'] if args.threads is None else args.threads
    cpus = sorted(os.sched_getaffinity(0))
    if not 1 <= threads <= min(64, len(cpus)):
        parser.error('Requested threads exceed the allocation')
    os.sched_setaffinity(0, cpus[:threads])
    os.environ.update(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1', JAX_ENABLE_COMPILATION_CACHE='false',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', GRADSCF_NATIVE_THREADS=str(threads))
    import jax
    import jax.numpy as jnp
    import gradscf
    from gradscf import gto
    from gradscf._memory import tree_storage_bytes
    from gradscf.integrals.basis import basis_from_molecule_spec
    from gradscf.integrals.grids import build_molecular_grid_from_spec
    from gradscf.integrals.grids.blocks import AOGrid
    from gradscf.scf.init_guess import restricted_initial_guess
    from gradscf.scf.inputs.grid import prepare_scf_grid
    from gradscf.scf.rks import RKSConfig, _restricted_xc_energy_and_matrix
    jax.config.update('jax_enable_x64', True)
    device = jax.devices('cpu')[0]
    sync = lambda x: jax.tree.map(lambda a: a.block_until_ready() if hasattr(a, 'block_until_ready') else a, x)
    def allocator():
        try: stats = device.memory_stats() or {}
        except (RuntimeError, NotImplementedError): stats = {}
        return {key: stats.get(key) for key in ('bytes_in_use', 'peak_bytes_in_use', 'bytes_limit')}
    t = time.perf_counter()
    spec = gto.M(atom=cfg['atom'], basis=cfg['basis'], unit=cfg['unit'], charge=cfg['charge'], spin=cfg['spin'], cart=True).to_spec()
    basis = basis_from_molecule_spec(spec, basis=cfg['basis'])
    coords, weights = build_molecular_grid_from_spec(spec, level=cfg['grid_level'])
    raw = AOGrid.from_basis(basis, coords, weights, block_size=4096, adaptive=True)
    density = restricted_initial_guess(init_guess='atom', dtype=jnp.float64, spec=spec,
        basis=cfg['basis'], max_memory=cfg['max_memory'], resident_bytes=tree_storage_bytes(raw)).density
    sync((raw, density))
    runtime = dict(scope='same-input builtin RKS/PBE XC only; no molecular SCF, ERI or RI factors',
        scientific_pass=False, source_config=str(frozen) if frozen.exists() else 'catalog smoke', config=cfg,
        job=os.environ['SLURM_JOB_ID'], backend=device.platform, affinity=sorted(os.sched_getaffinity(0)), jax=jax.__version__, host=platform.node(),
        nao=raw.nao, ngrid=raw.ngrids, dtype=str(density.dtype), density_shape=list(density.shape), density_source='native isolated-atom UHF superposition',
        max_memory_decimal_mb=cfg['max_memory'], reserved_factor_bytes=args.reserved_factor_bytes, factor_actually_held=False,
        source_module=gradscf.__file__, source_snapshot=os.getenv('GRADSCF_SOURCE_SNAPSHOT'),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), allocator=allocator())
    runtime['source_sha256'] = {name: hashlib.sha256((Path(gradscf.__file__).parent/name).read_bytes()).hexdigest()
        for name in ('scf/rks.py', 'scf/inputs/grid.py', 'integrals/grids/blocks.py', 'scf/init_guess.py')}
    (args.output/'runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
    emit('basis_grid_native_atom_guess', seconds=time.perf_counter()-t, allocator=allocator())
    controls = RKSConfig(xc_spec=cfg['xc'], max_memory=cfg['max_memory'])
    @jax.jit
    def action(dm, provider):
        return _restricted_xc_energy_and_matrix(density=dm, ao=jnp.zeros((0, provider.nao), dtype=dm.dtype),
            ao_deriv1=jnp.zeros((4, 0, provider.nao), dtype=dm.dtype), weights=provider.weights,
            cfg=controls, xc_kind='GGA', grid_ao=provider)
    def measure(label, provider):
        for repeat in range(4):
            t = time.perf_counter()
            result = sync(action(density, provider))
            emit(label, repeat=repeat, cache_state='cold compile+action' if repeat==0 else 'warm action',
                seconds=time.perf_counter()-t, energy_hartree=float(result[0]), block_size=provider.block_size, allocator=allocator())
        return result
    reference = measure('raw_xc', raw)
    reserved = args.reserved_factor_bytes+tree_storage_bytes(density)
    t = time.perf_counter()
    cached = sync(prepare_scf_grid(raw, max_memory=cfg['max_memory'], resident_bytes=reserved, deriv=1))
    emit('cache_preparation', seconds=time.perf_counter()-t, reserved_bytes=reserved,
        admitted=hasattr(cached, 'values'), factor_actually_held=False, allocator=allocator())
    result = measure('cached_xc' if hasattr(cached, 'values') else 'cache_not_admitted_xc', cached)
    emit('comparison', energy_abs_difference_hartree=float(jnp.abs(result[0]-reference[0])),
        potential_max_abs_difference_hartree=float(jnp.max(jnp.abs(result[1]-reference[1]))),
        scientific_pass=False, note='Cache preparation is separate; RSS is cumulative process high-water, not isolated warm peak')
    timer.cancel()

if __name__ == '__main__':
    main()
