"""Instrument the existing RHF scan without changing its iteration algorithm.

Run inside a cluster allocation when using CUDA. Existing CPU-prepared packed
DF caches may be reused; --prepare-cache prepares one new cache on CPU only.
"""
import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform as host_platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from validate_nnao_pilot import atomic_json, file_hash, numerical_runtime, read_structures

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--structures', type=Path,
    default=ROOT / 'artifacts/nnao-stage2-20261007/selection/all.jsonl')
parser.add_argument('--id', default='S66/33')
parser.add_argument('--cache', type=Path)
parser.add_argument('--prepare-cache', type=Path)
parser.add_argument('--platform', choices=('cpu', 'cuda'), default='cpu')
parser.add_argument('--damping', type=float, default=0.)
parser.add_argument('--level-shift', type=float, default=0.)
parser.add_argument('--restart-from', type=Path)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
if args.platform == 'cuda' and args.cache is None:
    parser.error('CUDA diagnostics require a prepared --cache; native preparation is CPU-only.')
if args.cache is not None and args.prepare_cache is not None:
    parser.error('Choose --cache or --prepare-cache, not both.')
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(variable, '8')
jax, experiment_class = numerical_runtime(args.platform)
import jax.numpy as jnp
import numpy as np
from gradscf.df import build_jk_from_df
from gradscf.integrals.density_fitting import project_factors
from gradscf.scf import rks

start = time.perf_counter()
row = read_structures(args.structures, ids=[args.id])[0]
kwargs = dict(geometry=row, basis_family='szp663_direct', core_primitives=6,
              jk_backend='df', implicit_tolerance=1e-8)
if args.cache is not None:
    kwargs.update(integral_cache=args.cache, cache_storage='mmap')
experiment = experiment_class(**kwargs)
cache = args.cache
if args.prepare_cache is not None:
    if args.prepare_cache.exists():
        raise FileExistsError(f'Refusing to overwrite cache {args.prepare_cache}')
    cache = experiment.write_integral_cache(args.prepare_cache)
setup_seconds = time.perf_counter() - start
outputs = experiment.layout.reference_outputs()
transform, overlap, hcore = experiment._contracted_one_electron(outputs, experiment.ps, experiment.ph)
factors = project_factors(experiment.rep, transform)
factors.block_until_ready()
cfg = rks.RKSConfig(xc_spec='hf', jk_backend='df', max_cycle=150,
    conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9,
    damping=args.damping, level_shift=args.level_shift)
initial_density = None
initial_orbitals = initial_eigenvalues = None
if args.restart_from is not None:
    with np.load(args.restart_from, allow_pickle=False) as saved:
        if str(saved['integral_signature'].item()) != experiment.integral_signature:
            raise ValueError('Restart density geometry/basis differs from this experiment.')
        initial_density = jnp.asarray(saved['density'])
        initial_orbitals = jnp.asarray(saved['mo_coeff'])
        initial_eigenvalues = jnp.asarray(saved['mo_energy'])

diis_records = []
original_solve = rks._diis_solve


def record_diis(gram, count):
    count = int(count); gram = np.asarray(gram)[:count, :count]
    scale = max(float(np.max(np.abs(gram))), np.finfo(float).tiny)
    gram = gram / scale
    regularized = gram + np.eye(count) * np.finfo(float).eps * 50.
    matrix = np.zeros((count + 1, count + 1))
    matrix[:count, :count] = regularized
    matrix[:count, count] = matrix[count, :count] = -1.
    rhs = np.zeros(count + 1); rhs[count] = -1.
    coefficients = np.linalg.solve(matrix, rhs)[:count]
    diis_records.append(dict(history_count=count,
        regularized_gram_condition=float(np.linalg.cond(regularized)),
        maximum_absolute_coefficient=float(np.max(np.abs(coefficients))),
        coefficient_norm=float(np.linalg.norm(coefficients))))


def traced_solve(fock_history, error_history, count):
    jax.debug.callback(record_diis, error_history @ error_history.T, count, ordered=True)
    return original_solve(fock_history, error_history, count)


@jax.jit
def solve(s, h, df):
    captured = {}
    original_loop = rks._run_scf_iterations_lax_core

    def traced_loop(**kwargs):
        values = original_loop(**kwargs)
        captured['history'] = values[10:14]
        return values

    rks._run_scf_iterations_lax_core = traced_loop
    rks._diis_solve = traced_solve
    try:
        result = rks.run_rks_from_integrals_traceable(overlap=s, hcore=h, eri=None,
            df_factors=df, nelectron=experiment.nelectron, nuclear_repulsion=experiment.enuc,
            ao=jnp.zeros((0, s.shape[0])), ao_deriv1=jnp.zeros((4, 0, s.shape[0])),
            grid_weights=jnp.zeros(0), config=cfg, init_density=initial_density,
            init_mo_coeff=initial_orbitals, init_mo_energy=initial_eigenvalues)
    finally:
        rks._run_scf_iterations_lax_core = original_loop
        rks._diis_solve = original_solve
    return result, captured['history']


tick = time.perf_counter()
result, history = solve(overlap, hcore, factors)
result.total_energy.block_until_ready()
solve_seconds = time.perf_counter() - tick
occupations = result.mo_occ
nocc = experiment.nelectron // 2
first_fragment_nao = 80 if args.id == 'S66/33' else 0


@jax.jit
def history_metrics(s, h, df, histories):
    def step(_, data):
        density, coeff, effective_energies, density_rms = data
        j, k = build_jk_from_df(df, density)
        fock = h + j - .5 * k
        energy = jnp.sum(density * h) + .5 * jnp.sum(density * j) - .25 * jnp.sum(density * k) + experiment.enuc
        raw_expectations = jnp.real(jnp.diag(coeff.T @ fock @ coeff))
        values = jnp.asarray([energy, jnp.linalg.norm(fock @ density @ s - s @ density @ fock),
            rks._mo_residual_norm(fock, coeff, occupations), density_rms,
            effective_energies[nocc] - effective_energies[nocc - 1],
            raw_expectations[nocc] - raw_expectations[nocc - 1],
            jnp.sum(jnp.diag(density @ s)[:first_fragment_nao])])
        return None, values
    return jax.lax.scan(step, None, histories)[1]


metrics = np.asarray(history_metrics(overlap, hcore, factors, history))[:int(result.cycles)]
names = ('energy_hartree', 'ao_commutator_norm', 'mo_gradient_norm', 'density_rms',
         'effective_gap_hartree', 'raw_expectation_gap_hartree', 'fragment_A_mulliken_electrons')
records = []
for i, values in enumerate(metrics):
    record = dict(cycle=i + 1, **dict(zip(names, map(float, values))))
    record['delta_energy_hartree'] = None if i == 0 else record['energy_hartree'] - records[-1]['energy_hartree']
    # First two cycles precede the first DIIS extrapolation (history count>=2).
    if i >= 2 and i - 2 < len(diis_records):
        record.update(diis_records[i - 2])
    records.append(record)
output = args.output_dir
output.mkdir(parents=True, exist_ok=True)
np.savez(output / 'final_state.npz', density=np.asarray(result.density_matrix),
    mo_coeff=np.asarray(result.mo_coeff), mo_energy=np.asarray(result.mo_energy),
    integral_signature=np.asarray(experiment.integral_signature))
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
report = dict(structure_id=args.id, symbols=row['symbols'], nelectron=experiment.nelectron,
    config=asdict(cfg), converged=bool(result.converged), cycles=int(result.cycles),
    total_energy_hartree=float(result.total_energy), final=records[-1], trace=records,
    minimum_overlap_eigenvalue=float(jnp.linalg.eigvalsh(overlap)[0]),
    primitive_nao=experiment.primitive_topology.nao, nao=experiment.layout.topology.nao,
    df_shape=list(experiment.rep.shape), projected_df_shape=list(factors.shape),
    cache_path=None if cache is None else str(cache), cache_sha256=None if cache is None else file_hash(cache),
    integral_signature=experiment.integral_signature, source_sha256=file_hash(__file__),
    restarted_from=None if args.restart_from is None else str(args.restart_from),
    rks_source_sha256=file_hash(rks.__file__), experiment_source_sha256=file_hash(ROOT / 'tools/optimize_methane_nnao.py'),
    cache_setup_seconds=setup_seconds, scf_compile_and_solve_seconds=solve_seconds,
    elapsed_seconds=time.perf_counter() - start,
    peak_rss_bytes=int(rss if host_platform.system() == 'Darwin' else rss * 1024),
    devices=[str(device) for device in jax.devices()], device_kinds=[device.device_kind for device in jax.devices()],
    device_memory_stats=jax.devices()[0].memory_stats(), dtype=str(factors.dtype),
    diis_capture='Host reconstruction of normalized Gram coefficients; diagnostic only.',
    fragment_population_note='S66/33: first11 nuclei=pyridine(42e),last6=ethene(16e); Mulliken is a diagnostic.'
        if first_fragment_nao else None)
atomic_json(output / 'diagnosis.json', report)
with (output / 'iterations.csv').open('w', newline='') as stream:
    fields = list(dict.fromkeys(key for record in records for key in record))
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)
print(json.dumps({key: report[key] for key in ('structure_id', 'converged', 'cycles',
    'total_energy_hartree', 'final', 'minimum_overlap_eigenvalue', 'scf_compile_and_solve_seconds',
    'elapsed_seconds', 'peak_rss_bytes')}, indent=2), flush=True)
