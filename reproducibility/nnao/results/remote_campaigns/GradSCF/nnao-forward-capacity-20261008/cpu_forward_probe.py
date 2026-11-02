"""Actual MACE and contracted-native RHF forward inference, without AD or ERI.

Run on a CPU Slurm allocation. No primitive cache is built or loaded. The
default S22/07 prediction matches the zero-init 7,171-parameter GPU-93809 model.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from validate_nnao_pilot import atomic_json, file_hash, read_structures, verify_source_imports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structures', type=Path,
        default=ROOT / 'artifacts/nnao-stage2-20261007/selection/all.jsonl')
    parser.add_argument('--id', default='S22/07')
    parser.add_argument('--gpu-reference', type=Path,
        default=ROOT / 'artifacts/nnao-capacity-20261008/gpu-93809.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    report = dict(status='running', structure_id=args.id,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), seed=0,
        basis='szp663_direct', core_primitives=6, cartesian=False,
        auxbasis='def2-universal-jkfit', primitive_cache_used=False,
        performed_gradients=False, requested_four_center_eri=False, pyscf_called=False,
        scf_timings=[], nn_forward_timings=[])
    atomic_json(args.output, report)
    try:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        os.environ['JAX_ENABLE_X64'] = '1'
        threads = os.environ.get('SLURM_CPUS_PER_TASK', '8')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ[name] = threads
        import jax
        import jax.numpy as jnp
        import numpy as np
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
        if any(device.platform != 'cpu' for device in jax.devices()):
            raise RuntimeError('CPU forward probe must use CPU devices only.')
        report.update(imported_sources=verify_source_imports(ROOT),
            python_version=platform.python_version(), jax_version=jax.__version__,
            versions={name: version(name) for name in ('jaxlib', 'flax', 'mace-jax', 'e3nn-jax')},
            devices=[str(device) for device in jax.devices()],
            device_kinds=[device.device_kind for device in jax.devices()],
            cpu_threads=threads,
            affinity_count=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            structures_sha256=file_hash(args.structures))
        import mace_jax
        vendor = ROOT / 'src/gradscf/model/nnao'
        if not Path(mace_jax.__file__).resolve().is_relative_to(vendor.resolve()):
            raise RuntimeError(f'MACE source mismatch: {mace_jax.__file__}')
        report['mace_source'] = str(Path(mace_jax.__file__).resolve())
        row = read_structures(args.structures, ids=[args.id])[0]
        atom = list(zip(row['symbols'], row['coords_angstrom']))
        model_config = dict(elements=(1, 6, 7, 8, 9), channels=8,
            num_interactions=2, max_ell=1, correlation=2, zero_init=True,
            basis_family='szp663_direct')
        layout = prepare_direct_basis(atom, unit='Angstrom', cart=False,
            basis_family='szp663_direct', core_primitives=6)
        model = MACEBasisModel(**model_config, rngs=nnx.Rngs(0))
        graph = build_graph([atomic_number(symbol) for symbol in row['symbols']],
            row['coords_angstrom'], element_order=model.elements)
        definition, parameters, other = nnx.split(model, nnx.Param, ...)
        initial, unravel = ravel_pytree(parameters)
        if int(initial.size) != 7171 or str(initial.dtype) != 'float64':
            raise RuntimeError('Model must match the float64 7,171-parameter GPU probe.')
        report.update(model_config=model_config, parameter_count=int(initial.size),
            natoms=len(row['symbols']), formula=row.get('formula_group'))

        @jax.jit
        def predict(vector, graph):
            return nnx.merge(definition, unravel(vector), other)(graph)

        for index in range(3):
            tick = time.perf_counter()
            outputs = predict(initial, graph)
            outputs.block_until_ready()
            report['nn_forward_timings'].append(dict(index=index,
                elapsed_seconds=time.perf_counter() - tick))
        np.testing.assert_allclose(np.asarray(outputs), np.asarray(layout.reference_outputs()),
                                   atol=1e-12, rtol=1e-12)
        report['coefficient_source'] = 'Actual zero-init MACE prediction; weights and output frozen.'
        atomic_json(args.output, report)

        tick = time.perf_counter()
        bound = layout.bind(outputs)
        plan = integrals.make_plan(layout.topology, backend='native')
        overlap = plan.evaluate('overlap', bound)
        hcore = plan.evaluate('kinetic', bound) + plan.evaluate('nuclear', bound)
        aux_top, aux_parameters = integrals.prepare_basis(atom, 'def2-universal-jkfit',
            unit='Angstrom', cart=False)
        packed_df = make_auxiliary_plan(layout.topology, aux_top).factors(bound, aux_parameters)
        df = unpack_factors(packed_df, layout.topology.nao)
        enuc = scf.nuclear_repulsion_energy(bound.nuclear_coords,
                                          jnp.asarray(layout.topology.nuclear_charges))
        jax.block_until_ready((overlap, hcore, packed_df, df, enuc))
        report.update(integral_generation_seconds=time.perf_counter() - tick,
            nao=layout.topology.nao, auxiliary_nao=aux_top.nao,
            actual_auxiliary_rank=int(packed_df.shape[0]), dtype=str(df.dtype),
            contracted_packed_shape=list(packed_df.shape), contracted_dense_shape=list(df.shape),
            contracted_packed_bytes=int(packed_df.size * packed_df.dtype.itemsize),
            contracted_dense_bytes=int(df.size * df.dtype.itemsize))
        atomic_json(args.output, report)
        cfg = RKSConfig(xc_spec='hf', jk_backend='df', max_cycle=150,
            conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9)
        report.update(scf_rescue_level_shift=.2, scf_config=dict(max_cycle=150,
            conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9))

        @jax.jit
        def solve(s, h, factors, nuclear_energy):
            n = s.shape[0]
            kwargs = dict(overlap=s, hcore=h, eri=None, df_factors=factors,
                nelectron=layout.nelectron, nuclear_repulsion=nuclear_energy,
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

        # Large DF factors are explicit JIT inputs; never captured as constants.
        reference_energy = -914.2778951467535 if args.id == 'S22/07' else None
        if args.gpu_reference.is_file() and args.id == 'S22/07':
            reference = json.loads(args.gpu_reference.read_text())
            if reference['model_config'] != json.loads(json.dumps(model_config)):
                raise ValueError('GPU reference model configuration differs.')
            reference_energy = reference['timings'][0]['energy_hartree']
            report['gpu_reference_sha256'] = file_hash(args.gpu_reference)
        report.update(reference_gpu_job='93809' if reference_energy is not None else None,
            reference_energy_hartree=reference_energy, reference_atol_hartree=1e-8)
        for index in range(3):
            tick = time.perf_counter()
            energy, details = solve(overlap, hcore, df, enuc)
            jax.block_until_ready((energy, details))
            elapsed = time.perf_counter() - tick
            checked = MethaneRHF._checked_info(energy, details)
            value = float(energy)
            if reference_energy is not None:
                np.testing.assert_allclose(value, reference_energy, atol=1e-8, rtol=0.)
            report['scf_timings'].append(dict(index=index, elapsed_seconds=elapsed,
                energy_hartree=value, scf=checked,
                reference_error_hartree=None if reference_energy is None else abs(value - reference_energy)))
            atomic_json(args.output, report)
        from gradscf.integrals import _native
        suffix = '.dylib' if platform.system() == 'Darwin' else '.so'
        native_library = Path(_native.__file__).parent / 'lib' / ('libgradscf_integrals' + suffix)
        report.update(status='passed', native_library_path=str(native_library.resolve()),
            native_library_sha256=file_hash(native_library))
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc())
    finally:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = int(rss if platform.system() == 'Darwin' else rss * 1024)
        report.update(host_max_rss_raw=rss, host_max_rss_raw_unit='bytes' if platform.system() == 'Darwin' else 'KiB',
            host_max_rss_bytes=rss_bytes, host_max_rss_gib=rss_bytes / 2**30,
            elapsed_seconds=time.perf_counter() - start, script_sha256=file_hash(__file__))
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
