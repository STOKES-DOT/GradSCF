"""Frozen MACE basis checkpoints: native contracted DF integrals and RHF forward.

Run inside a Slurm allocation. Each SCF starts from hcore, including warm
timing repeats. No primitive cache, four-index ERI, gradients or optimizer.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from validate_nnao_pilot import atomic_json, file_hash, read_structures, record_directory, scientific_source_hash, verify_source_imports

CHECKPOINT_HASH = '7e7f66fd900fe50cf9925f6d875fd566a953eb5a67600d67c53c518254084df2'
PARENT_SOURCE = '7f066083658982a552fda47e0144e762796f69788c6a1f1c7e70c77fbfec403a'
CURRENT_SOURCE = '87f14784cbc038349f65f311c4b664809925932abd2f0cccea834c5400102b30'
NATIVE_HASH = 'b0e65d075c537a3e56e499d7e8df85b8b493ed0e71b6f86ab82cda196a426fd4'
MODEL_CONFIG = dict(elements=(1, 6, 7, 8, 9), channels=8, num_interactions=2,
    max_ell=1, correlation=2, zero_init=True, basis_family='szp663_direct')
PROFILES = {
    'stage2-epoch41': dict(epoch=41, checkpoint_sha256=CHECKPOINT_HASH,
        source_sha256=PARENT_SOURCE, label='NNAO-epoch41', training_count=200),
    'pilot-best975': dict(epoch=975,
        checkpoint_sha256='ba93fb40efca3268962c7501f82627e188ed9509b6a73bd27a2df2170a3495de',
        source_sha256='7df11aecdbf295818de02c0db67da3de8d3db5b1ccef29798114308e28d08ebb',
        label='NNAO-pilot-best975', training_count=24),
}


def read_checkpoint(path, *, profile='stage2-epoch41', expected_hash=None):
    selected = PROFILES[profile]
    expected_hash = selected['checkpoint_sha256'] if expected_hash is None else expected_hash
    if file_hash(path) != expected_hash:
        raise ValueError('Original frozen checkpoint bytes changed.')
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata'].item()))
        vector = np.array(saved['parameters'])
    identity = metadata['identity']
    if (metadata['epoch'] != selected['epoch'] or identity['seed'] != 0 or identity['dtype'] != 'float64'
            or identity['scientific_source_sha256'] != selected['source_sha256']
            or identity['model_config'] != json.loads(json.dumps(MODEL_CONFIG))
            or vector.shape != (7171,) or vector.dtype != np.float64
            or not np.isfinite(vector).all()):
        raise ValueError('Checkpoint epoch/source/model must match the selected finite 7,171 float64 parameter profile.')
    vector.setflags(write=False)
    return metadata, vector


def manifest_source_hash(files, prefix=''):
    digest = hashlib.sha256()
    for name, sha in sorted(files.items()):
        if name.startswith(prefix) and Path(name).suffix in {'.py', '.json', '.dat'}:
            digest.update(name[len(prefix):].encode()); digest.update(sha.encode())
    return digest.hexdigest()


def verify_legacy_model(path, identity, *, root=ROOT):
    legacy = json.loads(path.read_text()); files = legacy['remote_sources']
    source = PROFILES['pilot-best975']['source_sha256']
    if (identity['scientific_source_sha256'] != source or len(files) != 700
            or legacy['measured_remote_source_sha256'] != source
            or manifest_source_hash(files) != source):
        raise RuntimeError('Legacy inventory does not reconstruct the frozen 700-file pilot source.')
    model = {name: sha for name, sha in files.items() if name.startswith('model/nnao/')}
    if len(model) != 127:
        raise RuntimeError('Legacy inventory must contain all 127 NNAO/model files.')
    package = (root / 'src/gradscf').resolve()
    for name, expected in model.items():
        actual = (package / name).resolve()
        if not actual.is_relative_to(package) or not actual.is_file() or file_hash(actual) != expected:
            raise RuntimeError(f'Legacy NNAO model mapping changed: {name}')
    return dict(legacy_inventory_sha256=file_hash(path),
        old_scientific_source_sha256=source, old_scientific_file_count=700,
        verified_old_model_file_count=127, model_files_unchanged=True)


def verify_current_sources(path, *, root=ROOT):
    manifest = json.loads(path.read_text())
    if (manifest['scientific_source_sha256'] != CURRENT_SOURCE or len(manifest['files']) != 764
            or manifest_source_hash(manifest['files'], 'src/gradscf/') != CURRENT_SOURCE):
        raise RuntimeError('Current manifest does not describe the frozen 764-file repaired source.')
    for name, expected in manifest['files'].items():
        actual = (root / name).resolve()
        if not actual.is_relative_to(root.resolve()) or not actual.is_file() or file_hash(actual) != expected:
            raise RuntimeError(f'Live current source differs from its manifest: {name}')
    if scientific_source_hash(root) != CURRENT_SOURCE:
        raise RuntimeError('Current scientific source inventory changed.')
    return dict(repaired_manifest_sha256=file_hash(path),
        repaired_scientific_source_sha256=CURRENT_SOURCE, verified_file_count=764)


def reference_baseline(path, structures_sha256, identifiers):
    expected_hash = 'b6e236e2b5766dc760d6a3b322b00880c7058d77f4c7c1b417b393a1cba09ce6'
    if file_hash(path) != expected_hash:
        raise ValueError('Previously validated job-95106 reference report bytes changed.')
    report = json.loads(path.read_text())
    if (report['status'] != 'complete' or report['failed'] != 0 or report['passed'] != 20
            or report['structures_sha256'] != structures_sha256
            or report['provenance']['repaired_scientific_source_sha256'] != CURRENT_SOURCE
            or report['model_config'] != json.loads(json.dumps(MODEL_CONFIG))
            or report['checkpoint_sha256'] != CHECKPOINT_HASH):
        raise ValueError('Reference baseline input/model/source identity is incompatible.')
    rows = {row['structure_id']: row for row in report['results'] if row['basis'] == 'NNAO-reference'}
    if any(identifier not in rows or rows[identifier]['status'] != 'passed'
           or not np.isfinite(rows[identifier]['energy_hartree']) for identifier in identifiers):
        raise ValueError('Reference baseline does not contain all selected finite, passed cases.')
    return dict(path=str(path.resolve()), sha256=expected_hash, slurm_job_id='95106',
        reference_recomputed=False, native_source_sha256=CURRENT_SOURCE,
        structures_sha256=structures_sha256,
        reference_energies_hartree={identifier: rows[identifier]['energy_hartree'] for identifier in identifiers})


def frozen_record(row, variant, shells, *, nao):
    labels = [f'{symbol}{index}' for index, symbol in enumerate(row['symbols'])]
    return dict(structure_id=row['structure_id'], basis=variant,
        symbols=list(row['symbols']), coords_angstrom=row['coords_angstrom'],
        atom_labels=labels, atom_shells=shells,
        atom=list(zip(labels, row['coords_angstrom'])), basis_by_atom=dict(zip(labels, shells)),
        unit='Angstrom', cart=False, charge=row['charge'], spin=row['spin'], nao=int(nao))


def atomic_jsonl(path, records):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n'
                                 for row in records))
    os.replace(temporary, path)


def place_parameters(parameters, device):
    """Move every registered BasisParameters leaf to the native FFI device."""
    import jax
    return jax.device_put(parameters, device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-profile', choices=tuple(PROFILES), default='stage2-epoch41')
    parser.add_argument('--variants', choices=('all', 'trained'), default='all')
    parser.add_argument('--output-dir', type=Path, required=True)
    manifest_dir = ROOT / 'artifacts/nnao-stage2-20261009-adim6-fix'
    parser.add_argument('--parent-manifest', type=Path, default=manifest_dir / 'parent-sources.json')
    parser.add_argument('--new-manifest', type=Path, default=manifest_dir / 'new-sources.json')
    parser.add_argument('--legacy-manifest', type=Path,
        default=Path(__file__).with_name('pilot-source-inventory.json'))
    parser.add_argument('--reference-report', type=Path,
        default=ROOT / 'artifacts/nnao-heldout-benchmark-20261010/job-95106/native/forward.json')
    parser.add_argument('--platform', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    selected = PROFILES[args.checkpoint_profile]
    checkpoint_hash = selected['checkpoint_sha256']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'forward.json').exists():
        parser.error('Use a fresh output directory to preserve existing benchmark evidence.')
    started = time.perf_counter()
    report = dict(status='running', slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        structures_sha256=file_hash(args.structures), checkpoint_sha256=checkpoint_hash,
        checkpoint_profile=args.checkpoint_profile, checkpoint_epoch=selected['epoch'],
        model_training_structure_count=selected['training_count'], variants=args.variants,
        seed=0, model_config=MODEL_CONFIG, basis_family='szp663_direct', core_primitives=6,
        method='RHF', auxbasis='def2-universal-jkfit', cart=False, dtype='float64',
        performed_gradients=False, performed_optimizer_updates=False, primitive_cache_used=False,
        requested_four_center_eri=False, pyscf_called=False,
        cache_policy='clear JAX caches before each variant; independent hcore SCF starts for all repeats',
        scf_config=dict(max_cycle=200, conv_tol=1e-12, conv_tol_density=1e-10,
            conv_tol_grad=1e-9, scf_rescue_level_shift=.2, initial_guess='hcore'),
        passed=0, failed=0, results=[])
    exports = []

    def persist():
        report['elapsed_seconds'] = time.perf_counter() - started
        atomic_json(args.output_dir / 'forward.json', report)
        atomic_jsonl(args.output_dir / 'frozen-bases.jsonl', exports)

    persist()
    try:
        os.environ['JAX_PLATFORMS'] = 'cuda,cpu' if args.platform == 'cuda' else 'cpu'
        os.environ['JAX_ENABLE_X64'] = '1'
        threads = os.environ.get('SLURM_CPUS_PER_TASK', '16')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ[name] = threads
        metadata, vector = read_checkpoint(args.checkpoint, profile=args.checkpoint_profile)
        rows = read_structures(args.structures, limit=args.limit)
        identity = metadata['identity']
        if len(identity['structure_ids']) != selected['training_count']:
            raise ValueError('Checkpoint training split differs from the selected model profile.')
        if set(row['structure_id'] for row in rows) & set(identity['structure_ids'] + identity.get('heldout_structure_ids', [])):
            raise ValueError('Benchmark structures overlap the training or model-selection validation IDs.')
        if args.checkpoint_profile == 'pilot-best975' and args.variants == 'trained':
            report['reference_baseline'] = reference_baseline(args.reference_report,
                report['structures_sha256'], [row['structure_id'] for row in rows])
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from jax.flatten_util import ravel_pytree
        from gradscf import integrals, scf
        from gradscf.data.molecule import atomic_number
        from gradscf.df import build_jk_from_df
        from gradscf.integrals.density_fitting import make_auxiliary_plan, unpack_factors
        from gradscf.model.nnao import MACEBasisModel, build_graph, prepare_direct_basis
        from gradscf.scf.core import _orthogonalizer, _diagonalize_fock, _build_density_from_occ
        from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable
        from optimize_methane_nnao import MethaneRHF, _run_scf_with_rescue
        jax.config.update('jax_enable_x64', True)
        if jax.default_backend() != ('gpu' if args.platform == 'cuda' else 'cpu'):
            raise RuntimeError('Requested execution platform is not available.')
        integral_device = jax.devices('cpu')[0]
        scf_device = jax.devices('gpu' if args.platform == 'cuda' else 'cpu')[0]
        if args.checkpoint_profile == 'stage2-epoch41':
            spec = importlib.util.spec_from_file_location('nnao_frozen_source_audit', manifest_dir / 'regress.py')
            audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
            provenance = audit.verify_sources(args.parent_manifest, args.new_manifest, identity)
        else:
            provenance = dict(verify_current_sources(args.new_manifest),
                legacy_model_compatibility=verify_legacy_model(args.legacy_manifest, identity),
                note='Frozen pilot weights are evaluated with the unchanged model mapping and independently audited current forward source; no training identity is rewritten.')
        if provenance['repaired_scientific_source_sha256'] != CURRENT_SOURCE:
            raise ValueError('Forward benchmark must use the frozen repaired source.')
        from gradscf.integrals import _native
        native_library = Path(_native.__file__).parent / 'lib' / 'libgradscf_integrals.so'
        if file_hash(native_library) != NATIVE_HASH:
            raise ValueError('Native library differs from the verified cluster build.')
        import mace_jax
        vendor = (ROOT / 'src/gradscf/model/nnao').resolve()
        if not Path(mace_jax.__file__).resolve().is_relative_to(vendor):
            raise RuntimeError('MACE imported from an installed or setup copy.')
        report.update(provenance=provenance, imported_sources=verify_source_imports(ROOT),
            mace_source=str(Path(mace_jax.__file__).resolve()), native_library=str(native_library.resolve()),
            native_library_sha256=NATIVE_HASH, parameter_count=7171,
            python_version=platform.python_version(),
            versions={name: version(name) for name in ('jax', 'jaxlib', 'flax', 'mace-jax', 'e3nn-jax')},
            devices=[str(device) for device in jax.devices()],
            device_kinds=[device.device_kind for device in jax.devices()],
            integral_device=str(integral_device), scf_device=str(scf_device),
            timing_note='Integral CPU time excludes explicit CPU-to-SCF transfer; transfer time is recorded separately.',
            cpu_threads=int(threads), script_sha256=file_hash(__file__), structure_count=len(rows))
        model = MACEBasisModel(**MODEL_CONFIG, rngs=nnx.Rngs(0))
        definition, parameters, other = nnx.split(model, nnx.Param, ...)
        initial, unravel = ravel_pytree(parameters)
        if initial.shape != vector.shape or str(initial.dtype) != 'float64':
            raise RuntimeError('MACE parameter tree shape/dtype differs from the checkpoint.')

        @jax.jit
        def predict(parameters, graph):
            return nnx.merge(definition, unravel(parameters), other)(graph)

        cfg = RKSConfig(xc_spec='hf', jk_backend='df', max_cycle=200,
            conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9)

        def solve(s, h, factors, nuclear_energy, nelectron):
            n = s.shape[0]
            kwargs = dict(overlap=s, hcore=h, eri=None, df_factors=factors,
                nelectron=nelectron, nuclear_repulsion=nuclear_energy,
                ao=jnp.zeros((0, n)), ao_deriv1=jnp.zeros((4, 0, n)), grid_weights=jnp.zeros(0))
            result, rescued, total_cycles = _run_scf_with_rescue(
                run_rks_from_integrals_traceable, kwargs, cfg, rescue_level_shift=.2)
            density = result.density_matrix
            j, k = build_jk_from_df(factors, density)
            fock = h + j - .5 * k
            energy = (jnp.sum(density * h) + .5 * jnp.sum(density * j)
                      - .25 * jnp.sum(density * k) + nuclear_energy)
            _, coeff = _diagonalize_fock(fock, _orthogonalizer(s, cfg.orthogonalization_eps))
            details = dict(converged=result.converged, scf_cycles=result.cycles,
                rescue_used=rescued, total_scf_cycles=total_cycles,
                orbital_residual=jnp.linalg.norm(fock @ density @ s - s @ density @ fock),
                fixed_point_residual=jnp.linalg.norm(_build_density_from_occ(coeff, result.mo_occ) - density),
                min_overlap_eigenvalue=jnp.linalg.eigvalsh(s)[0],
                reconstruction_error=jnp.abs(energy - result.total_energy))
            return energy, details

        solve = jax.jit(solve, static_argnums=(4,))
        persist()
        for row in rows:
            atom = list(zip(row['symbols'], row['coords_angstrom']))
            layout = prepare_direct_basis(atom, unit='Angstrom', cart=False,
                basis_family='szp663_direct', core_primitives=6)
            graph = build_graph([atomic_number(symbol) for symbol in row['symbols']],
                row['coords_angstrom'], element_order=model.elements)
            primitive_nao = sum((2 * l + 1) * count for l, count in
                zip(layout.topology.angular_momenta, layout.topology.primitive_counts))
            variants = [(selected['label'], vector)]
            if args.variants == 'all':
                variants.insert(0, ('NNAO-reference', initial))
            for variant, weights in variants:
                jax.clear_caches(); gc.collect()
                case_start = time.perf_counter()
                case = dict(structure_id=row['structure_id'], basis=variant, status='running',
                    natoms=len(row['symbols']), nao=layout.topology.nao, primitive_nao=primitive_nao,
                    fock_shape=[layout.topology.nao] * 2, nelectron=layout.nelectron,
                    checkpoint_profile='reference' if variant == 'NNAO-reference' else args.checkpoint_profile,
                    checkpoint_epoch=0 if variant == 'NNAO-reference' else selected['epoch'],
                    model_training_structure_count=0 if variant == 'NNAO-reference' else selected['training_count'],
                    nn_forward_timings=[], scf_timings=[], native_source_sha256=CURRENT_SOURCE)
                directory = record_directory(args.output_dir / 'cases', row['structure_id']) / variant
                path = directory / 'result.json'
                atomic_json(path, case)
                print(f'BEGIN {row["structure_id"]} {variant}', flush=True)
                try:
                    for index in range(3):
                        tick = time.perf_counter()
                        outputs = predict(jnp.asarray(weights), graph)
                        outputs.block_until_ready()
                        case['nn_forward_timings'].append(dict(index=index,
                            phase='cold' if index == 0 else 'warm',
                            elapsed_seconds=time.perf_counter() - tick))
                    if variant == 'NNAO-reference':
                        np.testing.assert_allclose(np.asarray(outputs), np.asarray(layout.reference_outputs()),
                                                   atol=1e-12, rtol=1e-12)
                        outputs = layout.reference_outputs()
                    tick = time.perf_counter()
                    bound = place_parameters(layout.bind(outputs), integral_device)
                    jax.block_until_ready(bound)
                    case['basis_binding_and_to_cpu_seconds'] = time.perf_counter() - tick
                    frozen = frozen_record(row, variant, layout.atom_shells(bound), nao=layout.topology.nao)
                    tick = time.perf_counter()
                    with jax.default_device(integral_device):
                        plan = integrals.make_plan(layout.topology, backend='native')
                        overlap = plan.evaluate('overlap', bound)
                        hcore = plan.evaluate('kinetic', bound) + plan.evaluate('nuclear', bound)
                        aux_top, aux_parameters = integrals.prepare_basis(atom, 'def2-universal-jkfit',
                            unit='Angstrom', cart=False)
                        aux_parameters = place_parameters(aux_parameters, integral_device)
                        packed_df = make_auxiliary_plan(layout.topology, aux_top).factors(bound, aux_parameters)
                        factors = unpack_factors(packed_df, layout.topology.nao)
                        enuc = scf.nuclear_repulsion_energy(bound.nuclear_coords,
                            jnp.asarray(layout.topology.nuclear_charges))
                        jax.block_until_ready((overlap, hcore, packed_df, factors, enuc))
                    case.update(integral_generation_seconds=time.perf_counter() - tick,
                        integral_kernel='native CPU; CPU DF whitening/unpacking included; CPU-to-SCF transfer excluded',
                        integral_device=str(integral_device), scf_device=str(scf_device),
                        auxiliary_nao=aux_top.nao, actual_auxiliary_rank=int(packed_df.shape[0]),
                        contracted_packed_shape=list(packed_df.shape), contracted_dense_shape=list(factors.shape),
                        contracted_packed_bytes=int(packed_df.size * packed_df.dtype.itemsize),
                        contracted_dense_bytes=int(factors.size * factors.dtype.itemsize),
                        coefficient_source='seed-zero MACE/reference' if variant == 'NNAO-reference' else f'frozen {args.checkpoint_profile} MACE',
                        prediction_shape=list(outputs.shape))
                    tick = time.perf_counter()
                    overlap, hcore, packed_df, factors, enuc = jax.device_put(
                        (overlap, hcore, packed_df, factors, enuc), scf_device)
                    jax.block_until_ready((overlap, hcore, packed_df, factors, enuc))
                    case['host_device_transfer_seconds'] = time.perf_counter() - tick
                    case['scf_input_devices'] = dict(overlap=str(overlap.device), hcore=str(hcore.device),
                        df_factors=str(factors.device), nuclear_energy=str(enuc.device))
                    atomic_json(path, case)
                    energies = []
                    for index in range(3):
                        tick = time.perf_counter()
                        energy, details = solve(overlap, hcore, factors, enuc, layout.nelectron)
                        jax.block_until_ready((energy, details))
                        elapsed = time.perf_counter() - tick
                        checked = MethaneRHF._checked_info(energy, details)
                        energies.append(float(energy))
                        case['scf_timings'].append(dict(index=index,
                            phase='cold' if index == 0 else 'warm', elapsed_seconds=elapsed,
                            energy_hartree=float(energy), scf=checked))
                        atomic_json(path, case)
                    np.testing.assert_allclose(energies, energies[0], atol=1e-9, rtol=0)
                    case.update(status='passed', energy_hartree=energies[-1],
                        elapsed_seconds=time.perf_counter() - case_start)
                    atomic_json(path, case)
                    frozen.update(native_energy_hartree=energies[-1], native_source_sha256=CURRENT_SOURCE,
                        native_result_sha256=file_hash(path), checkpoint_sha256=checkpoint_hash,
                        checkpoint_profile='reference' if variant == 'NNAO-reference' else args.checkpoint_profile,
                        checkpoint_epoch=0 if variant == 'NNAO-reference' else selected['epoch'],
                        model_training_structure_count=0 if variant == 'NNAO-reference' else selected['training_count'],
                        auxiliary_basis='def2-universal-jkfit')
                    exports.append(frozen); report['passed'] += 1
                    del overlap, hcore, packed_df, factors, enuc
                except Exception as error:
                    case.update(status='failed', error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - case_start)
                    report['failed'] += 1
                    atomic_json(path, case)
                report['results'].append(case); persist()
                print(f'END {row["structure_id"]} {variant} {case["status"]}', flush=True)
        if file_hash(args.checkpoint) != checkpoint_hash:
            raise ValueError('Parent checkpoint bytes changed during the benchmark.')
        report['status'] = 'complete' if report['failed'] == 0 else 'failed'
        report['original_checkpoint_bytes_unchanged'] = True
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc())
    persist()
    return int(report['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
