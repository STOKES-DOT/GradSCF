"""CPU abstract compilation of the current RHF implicit evaluator's memory.

Only small basis/layout arrays are materialized. S, H and primitive DF inputs
are ShapeDtypeStruct placeholders; the compiled SCF is never executed.
Synthetic compositions/geometries are size fixtures, not molecular results.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import runpy
import time
import traceback

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import jax.numpy as jnp
import numpy as np
from gradscf import integrals
from gradscf.integrals.contraction import primitive_basis
from gradscf.model.nnao import prepare_direct_basis

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def cases(selection):
    rows = {row['structure_id']: row for row in
            (json.loads(line) for line in Path(selection).read_text().splitlines())}
    chosen = []
    for name in ('S22/07', 'ADIM6/AD6', 'MCONF/21'):
        row = dict(rows[name]); row['kind'] = 'actual_selection_geometry'
        chosen.append(row)
    for composition in (dict(C=16,H=34), dict(C=19,H=40),
                        dict(C=15,H=16,N=10,O=4), dict(C=17,H=18,N=12,O=3),
                        dict(C=26,H=54)):
        symbols = [symbol for symbol,count in composition.items() for _ in range(count)]
        # Deliberately nonphysical, separated coordinates: only layout sizes
        # and composition determine this fixture's abstract compilation.
        coords = [[3. * (i % 10), 3. * ((i // 10) % 10), 3. * (i // 100)]
                  for i in range(len(symbols))]
        formula = ''.join(symbol + str(count) for symbol,count in composition.items())
        chosen.append(dict(structure_id=f'fixture_{formula}_{len(symbols)}atoms',
            symbols=symbols, coords_angstrom=coords, charge=0, spin=0,
            kind='synthetic_composition_separated_coordinate_fixture'))
    row = rows['S22/07']; coords = np.asarray(row['coords_angstrom'])
    chosen.append(dict(structure_id='fixture_2xS22_07_60atoms',
        symbols=row['symbols'] * 2, charge=0, spin=0,
        coords_angstrom=np.concatenate([coords, coords + [30., 0., 0.]]).tolist(),
        kind='two_translated_copies_size_fixture_not_a_physical_calculation'))
    return chosen


def estimate(row, experiment_class):
    started = time.perf_counter()
    record = dict(structure_id=row['structure_id'], kind=row['kind'],
        natoms=len(row['symbols']), composition=dict(Counter(row['symbols'])),
        status='compiling', native_integral_evaluations=0, scf_executions=0)
    try:
        atom = list(zip(row['symbols'], row['coords_angstrom']))
        ex = experiment_class.__new__(experiment_class)
        ex.layout = prepare_direct_basis(atom, unit='Angstrom', basis_family='szp663_direct', core_primitives=6)
        primitive_topology, _ = primitive_basis(ex.layout.topology, ex.layout.parameters)
        auxiliary, _ = integrals.prepare_basis(atom, 'def2-universal-jkfit', cart=False)
        ex.jk_backend = 'df'; ex.implicit_tolerance = 1e-8
        ex.scf_rescue_level_shift = .2
        ex.nelectron = sum(ex.layout.topology.nuclear_charges)
        # A scalar placeholder preserves the real evaluator signature. Its
        # value has no bearing on workspace sizes; no energy is calculated.
        ex.enuc = jnp.asarray(0., dtype=jnp.float64)
        primitive, nao, rank = primitive_topology.nao, ex.layout.topology.nao, auxiliary.nao
        if row['kind'] == 'actual_selection_geometry':
            expected = row['size_estimate']
            assert (primitive, nao, rank) == (expected['primitive_nao'], expected['nao'], expected['naux'])
        packed = primitive * (primitive + 1) // 2
        shapes = (jax.ShapeDtypeStruct(ex.layout.reference_outputs().shape, jnp.float64),
                  jax.ShapeDtypeStruct((primitive, primitive), jnp.float64),
                  jax.ShapeDtypeStruct((primitive, primitive), jnp.float64),
                  jax.ShapeDtypeStruct((rank, packed), jnp.float64))
        evaluator = jax.jit(jax.value_and_grad(ex._implicit_value, argnums=0, has_aux=True))
        compiled = evaluator.lower(*shapes).compile()
        memory = compiled.memory_analysis()
        values = {name: int(getattr(memory, name)) for name in (
            'argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes',
            'alias_size_in_bytes', 'generated_code_size_in_bytes')}
        total = values['argument_size_in_bytes'] + values['output_size_in_bytes'] + values['temp_size_in_bytes'] - values['alias_size_in_bytes']
        record.update(status='compiled', primitive_nao=primitive, contracted_nao=nao,
            auxiliary_rank_assuming_full_metric_rank=rank, nelectron=ex.nelectron,
            packed_df_shape=[rank, packed], packed_df_bytes=rank * packed * 8,
            memory_analysis_bytes=values, arguments_plus_temporary_minus_alias_bytes=total,
            estimated_arguments_plus_temporary_gib=total / 1024**3)
        del compiled, evaluator, ex, shapes
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        jax.clear_caches(); gc.collect()
        record['compile_elapsed_seconds'] = time.perf_counter() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record['process_peak_rss_gib'] = rss / (1024**3 if platform.system() == 'Darwin' else 1024**2)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=ROOT / 'artifacts/nnao-stage2-20261007/selection/all.jsonl')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results.json'))
    parser.add_argument('--cases', nargs='+', help='Restrict by exact structure/fixture ID.')
    parser.add_argument('--append', action='store_true', help='Add cases to compatible existing results.')
    args = parser.parse_args()
    rows = cases(args.selection)
    if args.cases:
        unknown = set(args.cases) - {row['structure_id'] for row in rows}
        if unknown: parser.error(f'Unknown cases: {sorted(unknown)}')
        rows = [row for row in rows if row['structure_id'] in args.cases]
    source = ROOT / 'tools/optimize_methane_nnao.py'
    experiment_class = runpy.run_path(str(source))['MethaneRHF']
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        jax_version=jax.__version__, backend=jax.default_backend(),
        devices=[str(device) for device in jax.devices()], python_version=platform.python_version(),
        platform=platform.platform(), dtype='float64', basis='szp663_direct/core6 spherical',
        method='RHF-DF implicit value_and_grad with auxiliary diagnostics',
        implicit_tolerance=1e-8, scf_rescue_level_shift=.2,
        source_sha256={str(path.relative_to(ROOT)):digest(path) for path in (
            source, ROOT / 'src/gradscf/integrals/density_fitting.py', ROOT / 'src/gradscf/solvers/nonlinear/fixed_point.py')},
        selection_sha256=digest(args.selection),
        limitations=['CPU XLA abstract compilation; not GPU peak measurement.',
            'No large numerical integral inputs and no compiled SCF execution.',
            'Excludes CUDA runtime, allocator fragmentation, MACE/optimizer, compiled other cases and OS I/O buffers.',
            'Full auxiliary metric rank assumed; no native 2c/3c computation.',
            'Synthetic fixture geometries/compositions do not represent computed physical systems.',
            'Input + temporary memory is an indicative lower-level allocation estimate, not an A100 fit guarantee.'],
        results=[])
    if args.append and args.output.exists():
        previous=json.loads(args.output.read_text())
        for key in ('jax_version','backend','dtype','method','implicit_tolerance',
                    'scf_rescue_level_shift','source_sha256','selection_sha256'):
            if previous.get(key)!=report.get(key):
                raise ValueError(f'Cannot append estimates after {key} changed.')
        report=previous
        completed={row['structure_id'] for row in report['results'] if row['status']=='compiled'}
        rows=[row for row in rows if row['structure_id'] not in completed]
    report.setdefault('compile_runs',[]).append(dict(
        started_utc=datetime.now(timezone.utc).isoformat(), script_sha256=digest(__file__),
        cases=[row['structure_id'] for row in rows]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        print(f"Compiling abstract {row['structure_id']} ({len(row['symbols'])} atoms)", flush=True)
        report['results'].append(estimate(row, experiment_class))
        report['updated_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.output, report)
        last = report['results'][-1]
        print(json.dumps({key:last.get(key) for key in ('structure_id', 'status',
            'primitive_nao', 'contracted_nao', 'estimated_arguments_plus_temporary_gib',
            'compile_elapsed_seconds', 'process_peak_rss_gib', 'error')}, sort_keys=True), flush=True)
    return int(any(row['status'] == 'failed' for row in report['results']))


if __name__ == '__main__':
    raise SystemExit(main())
