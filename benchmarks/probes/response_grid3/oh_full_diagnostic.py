"""Bounded baseline OH full-TDDFT diagnostic; not a performance benchmark.

Rebuild the exact failed DFT reference, retain the public iterative result, and
materialize the ORIGINAL gen_tdhf_vind action in batches of at most 16 rows.
No changed thresholds, response equations, roots or stabilizing modifications.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline-runtime', type=Path, required=True)
    parser.add_argument('--block-size', type=int, default=16)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        parser.error('Numerical diagnostics require the allocated remote Slurm step.')
    if not 1 <= args.block_size <= 16:
        parser.error('Operator batches are restricted to 1 through 16 rows.')
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        parser.error('Use new output paths; existing evidence is immutable.')
    cfg = json.loads(args.config.read_text())
    baseline = json.loads(args.baseline_runtime.read_text())
    normalized_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if normalized_hash != baseline['input_sha256']:
        parser.error('Input must match the normalized failed-run configuration exactly.')
    if (cfg['ks'] != 'uks' or cfg['xc'] != 'pbe' or not cfg['full']
            or cfg['basis'].lower() != 'def2-svp' or cfg['grid_level'] != 3
            or cfg['representation'] != 'full' or cfg['device'] != 'cpu'
            or cfg['charge'] != 0 or cfg['spin'] != 1):
        parser.error('This diagnostic is bounded to the recorded OH/UKS/PBE/def2-SVP/grid3 full-integral case.')
    allowed = sorted(os.sched_getaffinity(0))
    if cfg['threads'] > len(allowed):
        parser.error('Requested CPUs exceed the allocated affinity.')
    os.sched_setaffinity(0, allowed[:cfg['threads']])
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(cfg['blas_threads'])
    os.environ['GRADSCF_NATIVE_THREADS'] = str(cfg['threads'])
    os.environ.setdefault('JAX_ENABLE_COMPILATION_CACHE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    import numpy as np
    import gradscf
    from gradscf import dft, gto, tdscf
    from gradscf.integrals._native import register_integrals
    from gradscf.tddft.defaults import TD_POSITIVE_EIG_THRESHOLD

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='started', diagnostic_only=True, eligible_for_speed_ratio=False,
                  config=cfg, input_sha256=normalized_hash, probe_sha256=digest(__file__),
                  baseline_runtime_sha256=digest(args.baseline_runtime),
                  units='Hartree, Angstrom input, seconds, bytes',
                  interpretation='Forward operator and spectrum evidence; no AD or external-reference certification.',
                  versions=dict(jax=jax.__version__, numpy=np.__version__),
                  python=sys.executable, source=gradscf.__file__,
                  slurm={key: os.getenv(key) for key in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURMD_NODENAME')},
                  affinity=sorted(os.sched_getaffinity(0)), stages=[])

    def safe(value):
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return safe(value.tolist())
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def save():
        args.output.write_text(json.dumps(safe(report), indent=2, allow_nan=False) + '\n')

    def event(stage, **values):
        row = dict(stage=stage, **values)
        report['stages'].append(row)
        print(json.dumps(safe(row), allow_nan=False), flush=True)
        save()

    def maxabs(array):
        return float(np.max(np.abs(array)))

    try:
        source_root = Path(gradscf.__file__).resolve().parent
        hashes = {name: digest(source_root / name) for name in baseline['source_files']}
        report['source_sha256'] = hashes
        mismatches = [name for name, value in hashes.items()
                      if value != baseline['source_files'][name]['sha256']]
        library = Path(register_integrals()._name).resolve()
        report.update(native_path=str(library), native_sha256=digest(library))
        if mismatches or report['native_sha256'] != baseline['native_sha256']:
            raise ValueError(f'Diagnostic source/native must match the failed baseline: {mismatches}')
        event('source_checked', production_source_matches_failed_run=True)
        mol = gto.M(atom=cfg['atom'], basis=cfg['basis'], unit=cfg['unit'],
                    charge=cfg['charge'], spin=cfg['spin'], cart=True)
        mf = dft.UKS(mol, xc=cfg['xc'], grids_level=cfg['grid_level'], execution_device='cpu',
                     conv_tol=cfg['scf_tol'], conv_tol_density=cfg['density_tol'],
                     conv_tol_grad=cfg['gradient_tol'], max_cycle=cfg['max_cycle'],
                     max_memory=cfg['max_memory'], grid_block_size=cfg['grid_block_size'],
                     init_guess='hcore')
        start = time.perf_counter()
        mf.run()
        jax.block_until_ready(mf.reference)
        event('scf', diagnostic_seconds=time.perf_counter()-start,
              converged=bool(mf.converged), energy_hartree=float(mf.e_tot), cycles=int(mf.cycles))
        if not mf.converged:
            raise ArithmeticError('New SCF did not converge; no response diagnosis is made.')
        td = tdscf.TDDFT(mf, nstates=cfg['roots'], eigensolver='davidson',
                         davidson_tol=cfg['td_tol'], davidson_max_iter=cfg['td_max_cycle'],
                         response_kernel_options=dict(two_electron_mode='auto',
                             max_memory=cfg['max_memory'], direct_workers=cfg['threads'],
                             direct_tol=cfg['direct_tol'], integral_cache=cfg['integral_cache'],
                             grid_block_size=cfg['response_grid_block_size']))
        ref = td.molecule
        jax.block_until_ready(ref)
        occupations = np.asarray(ref.mo_occ)
        nocc = np.count_nonzero(occupations, axis=1)
        nmo = int(ref.mo_coeff.shape[-1])
        dimension = int(np.sum(nocc*(nmo-nocc)))
        if nmo != 20 or tuple(nocc) != (5, 4) or dimension != 139:
            raise ValueError(f'Expected 20 MOs and (5,4) occupations / 139 transitions, got {nmo}, {nocc}, {dimension}.')
        size = 2*dimension
        threshold = float(TD_POSITIVE_EIG_THRESHOLD)
        report.update(nao=int(ref.mo_coeff.shape[-2]), nmo=nmo, nocc=nocc,
                      ngrid=int(ref.grid.weights.size), transition_dimension=dimension,
                      operator_dimension=size, existing_positive_threshold_hartree=threshold,
                      configured_residual_tolerance_hartree=cfg['td_tol'])
        start = time.perf_counter()
        iterative = None
        try:
            iterative = jax.block_until_ready(td.kernel())
            event('public_full_tddft', diagnostic_seconds=time.perf_counter()-start,
                  converged=bool(np.all(np.asarray(td.converged))),
                  excitation_energies_hartree=np.asarray(td.e))
        except Exception as exc:
            event('public_full_tddft', diagnostic_seconds=time.perf_counter()-start,
                  error_type=type(exc).__name__, error=str(exc))
        solver = td._solver if td._solver is not None else td._build_solver()
        start = time.perf_counter()
        vind = solver.gen_tdhf_vind()
        # gen_tdhf_vind consumes row vectors: applying identity rows returns
        # columns of L as rows. Transpose ONCE to recover L, not L.T.
        columns_as_rows = np.empty((size, size), dtype=np.float64)
        for offset in range(0, size, args.block_size):
            width = min(args.block_size, size-offset)
            basis_rows = np.zeros((width, size), dtype=np.float64)
            basis_rows[np.arange(width), offset+np.arange(width)] = 1.
            acted = np.asarray(jax.block_until_ready(vind(jax.numpy.asarray(basis_rows))))
            if acted.shape != basis_rows.shape:
                raise ValueError(f'Unexpected row action shape {acted.shape}.')
            columns_as_rows[offset:offset+width] = acted
        matrix = columns_as_rows.T.copy()
        if not np.isfinite(matrix).all():
            raise ArithmeticError('Original response operator contains non-finite entries.')
        # An independent, non-basis row verifies the transpose convention.
        probe = np.sin(np.arange(size, dtype=float)+.3)[None, :]
        probe_action = np.asarray(jax.block_until_ready(vind(jax.numpy.asarray(probe))))
        row_error = maxabs(probe_action-probe@matrix.T)
        a, b = matrix[:dimension, :dimension], matrix[:dimension, dimension:]
        structure = dict(a_symmetry=maxabs(a-a.T), b_symmetry=maxabs(b-b.T),
                         lower_left_plus_b=maxabs(matrix[dimension:, :dimension]+b),
                         lower_right_plus_a=maxabs(matrix[dimension:, dimension:]+a),
                         independent_row_action_error=row_error)
        event('original_operator', diagnostic_seconds=time.perf_counter()-start,
              input_row_block_limit=args.block_size, max_abs_matrix_element=maxabs(matrix),
              structure_errors_hartree=structure)
        # Report symmetrization explicitly; the full spectrum below uses the
        # unmodified L. Never replace physical L by a symmetrized/stabilized one.
        minus, plus = a-b, a+b
        stability_minus = np.linalg.eigvalsh(.5*(minus+minus.T))
        stability_plus = np.linalg.eigvalsh(.5*(plus+plus.T))
        values, vectors = np.linalg.eig(matrix)
        order = np.lexsort((values.imag, values.real))
        values, vectors = values[order], vectors[:, order]
        residuals = np.linalg.norm(matrix@vectors-vectors*values[None, :], axis=0)
        residuals /= np.linalg.norm(vectors, axis=0)
        metric_norms = (np.sum(np.abs(vectors[:dimension])**2, axis=0)
                        - np.sum(np.abs(vectors[dimension:])**2, axis=0))
        nearly_real = np.abs(values.imag) <= cfg['td_tol']
        positive = nearly_real & (values.real > threshold)
        small_real = nearly_real & (np.abs(values.real) <= threshold)
        report['dense_spectrum'] = dict(
            eigenvalue_real_hartree=values.real, eigenvalue_imaginary_hartree=values.imag,
            original_operator_residual_hartree=residuals, metric_norms=metric_norms,
            imaginary_classification_tolerance_hartree=cfg['td_tol'],
            positive_real_above_existing_threshold_hartree=values.real[positive],
            positive_real_metric_norms=metric_norms[positive],
            small_real_within_existing_threshold_hartree=values.real[small_real],
            complex_count=int(np.count_nonzero(~nearly_real)),
            min_eigenvalue_symmetrized_a_minus_b_hartree=float(stability_minus[0]),
            min_eigenvalue_symmetrized_a_plus_b_hartree=float(stability_plus[0]),
            eigenvalues_symmetrized_a_minus_b_hartree=stability_minus,
            eigenvalues_symmetrized_a_plus_b_hartree=stability_plus,
            note='A±B eigenvalues use their Hermitian parts; interpret with the separately reported structure errors. Full L eigenvalues and residuals use the unmodified original operator.')
        selected_vectors = np.zeros((0, size))
        if iterative is not None:
            energies = np.asarray(iterative.excitation_energies)
            names = ('x_amplitudes_alpha', 'x_amplitudes_beta',
                     'y_amplitudes_alpha', 'y_amplitudes_beta')
            selected_vectors = np.concatenate([np.asarray(getattr(iterative, name)).reshape(len(energies), -1)
                                               for name in names], axis=1)
            selected_action = np.asarray(jax.block_until_ready(vind(jax.numpy.asarray(selected_vectors))))
            selected_residuals = np.linalg.norm(selected_action-energies[:, None]*selected_vectors, axis=1)
            norms = np.linalg.norm(selected_vectors, axis=1)
            nearest = np.argmin(np.abs(energies[:, None]-values[None, :]), axis=1)
            report['iterative_comparison'] = dict(
                public_converged=np.asarray(iterative.converged), excitation_energies_hartree=energies,
                original_action_residual_absolute_hartree=selected_residuals,
                original_action_residual_relative_hartree=selected_residuals/norms,
                euclidean_vector_norms=norms,
                metric_norms=np.sum(selected_vectors[:, :dimension]**2, axis=1)
                    - np.sum(selected_vectors[:, dimension:]**2, axis=1),
                action_vs_dense_matrix_error_hartree=maxabs(selected_action-selected_vectors@matrix.T),
                nearest_full_eigenvalue_real_hartree=values.real[nearest],
                nearest_full_eigenvalue_imaginary_hartree=values.imag[nearest],
                distance_to_nearest_full_eigenvalue_hartree=np.abs(energies-values[nearest]))
        arrays = args.output.with_suffix('.npz')
        np.savez(arrays, response_matrix=matrix, a=a, b=b, eigenvalues=values,
                 eigenvectors=vectors, selected_vectors=selected_vectors,
                 mo_coeff=np.asarray(ref.mo_coeff), mo_energy=np.asarray(ref.mo_energy),
                 mo_occ=occupations, density=np.asarray(ref.rdm1))
        report.update(status='diagnostic_complete', arrays_path=str(arrays), arrays_sha256=digest(arrays),
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        event('complete', diagnostic_complete=True, public_convergence_is_not_overridden=True)
        return 0
    except Exception as exc:
        report.update(status='error', error_type=type(exc).__name__, error=str(exc))
        save()
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
