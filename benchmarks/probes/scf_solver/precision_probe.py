"""Compare orbital frames on native S/h only; no SCF, grid or four-center ERI.

Inside remote Slurm: python precision_probe.py case.json output.json --threads 8
The Cholesky candidate is diagnostic until separately accepted in production.
Both variants use the same float64 overlap/hcore and fixed spin occupations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Numerical runs require remote Slurm")
    affinity = sorted(os.sched_getaffinity(0))
    if not 1 <= args.threads <= min(64, len(affinity)):
        parser.error("Require 1--64 allocated logical CPUs")
    if args.output.exists():
        parser.error("Refusing to overwrite raw probe output")
    os.sched_setaffinity(0, affinity[:args.threads])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
        JAX_ENABLE_COMPILATION_CACHE="false", GRADSCF_NATIVE_THREADS=str(args.threads))
    cfg = json.loads(args.config.read_text())
    if not cfg.get("cart", True):
        parser.error("This probe preserves the archived Cartesian AO setting")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.scipy.linalg import solve_triangular
    from scipy.linalg import eigh
    import gradscf
    from gradscf import gto
    from gradscf.integrals import basis_from_molecule_spec, overlap_matrix, build_hcore
    from gradscf.integrals._native import register_integrals
    from gradscf.solvers.eigen.spectral import inverse_sqrt
    jax.config.update("jax_enable_x64", True)

    def timed(function, *values):
        start = time.perf_counter()
        result = function(*values)
        jax.block_until_ready(result)
        return result, time.perf_counter()-start

    mol = gto.M(atom=cfg["atom"], basis=cfg["basis"], unit=cfg.get("unit", "Angstrom"),
                charge=cfg.get("charge", 0), spin=cfg.get("spin", 0), cart=True)
    start = time.perf_counter()
    basis = basis_from_molecule_spec(mol.to_spec(), basis=mol.basis)
    jax.block_until_ready((basis.ao_centers, basis.ao_exponents_padded,
        basis.ao_coefficients_padded, basis.shell_centers, basis.shell_exponents_padded,
        basis.shell_coefficients_padded))
    basis_seconds = time.perf_counter()-start
    overlap, overlap_seconds = timed(overlap_matrix, basis)
    hcore, hcore_seconds = timed(build_hcore, basis)
    s_host, h_host = np.asarray(overlap), np.asarray(hcore)
    spectrum = np.linalg.eigvalsh(s_host)
    eps = 1e-10
    if spectrum[0] <= eps:
        raise ValueError("The archived case is outside the unregularized SPD comparison")
    reference_values = eigh(h_host, s_host, eigvals_only=True)
    na = (mol.nelectron+mol.spin)//2
    counts = (na, mol.nelectron-na)
    occupation = sum(np.arange(basis.nao) < count for count in counts)
    reference_energy = float(np.dot(occupation, reference_values))

    def cholesky_frame(s):
        lower = jnp.linalg.cholesky(s)
        return solve_triangular(lower.T, jnp.eye(s.shape[0], dtype=s.dtype), lower=False)

    def diagonalize(x, h):
        transformed = x.T @ h @ x
        values, vectors = jnp.linalg.eigh(.5*(transformed+transformed.T))
        return values, x @ vectors

    spectrum_program = jax.jit(diagonalize)
    variants = []
    for name, transform in (("symmetric_inverse_sqrt", lambda s: inverse_sqrt(s, eps)),
                            ("cholesky_frame", cholesky_frame)):
        start = time.perf_counter()
        executable = jax.jit(transform).lower(overlap).compile()
        compile_seconds = time.perf_counter()-start
        x, first_seconds = timed(executable, overlap)
        warm_seconds = [timed(executable, overlap)[1] for _ in range(3)]
        (values, coeff), diagonalization_seconds = timed(spectrum_program, x, hcore)
        x_host, values_host, coeff_host = map(np.asarray, (x, values, coeff))
        metric = x_host.T @ s_host @ x_host-np.eye(basis.nao)
        residual = h_host @ coeff_host-s_host @ (coeff_host*values_host)
        occupied_energy = float(np.dot(occupation, values_host))
        density = (coeff_host*occupation) @ coeff_host.T
        spin_mo_norms, spin_commutators = [], []
        for count in counts:
            co, cv = coeff_host[:, :count], coeff_host[:, count:]
            spin_mo_norms.append(float(np.linalg.norm(cv.T @ (h_host @ co))))
            dm = co @ co.T
            spin_commutators.append(x_host.T @ (h_host @ dm @ s_host-s_host @ dm @ h_host) @ x_host)
        variants.append(dict(name=name, transform_compile_seconds=compile_seconds,
            transform_first_execution_seconds=first_seconds, transform_warm_seconds=warm_seconds,
            diagonalization_seconds=diagonalization_seconds,
            diagonalization_scope="first call includes JIT compile; second variant reuses the same signature",
            metric_error_frobenius=float(np.linalg.norm(metric)), metric_error_max=float(np.max(np.abs(metric))),
            coeff_metric_error_frobenius=float(np.linalg.norm(coeff_host.T @ s_host @ coeff_host-np.eye(basis.nao))),
            generalized_eigen_residual_frobenius=float(np.linalg.norm(residual)),
            occupied_generalized_residual_frobenius=float(np.linalg.norm(residual[:, :max(counts)])),
            occupied_one_electron_eigenvalue_sum_hartree=occupied_energy,
            eigenvalue_sum_error_vs_scipy_hartree=occupied_energy-reference_energy,
            density_contraction_error_vs_scipy_hartree=float(np.einsum('ij,ji->', density, h_host))-reference_energy,
            maximum_eigenvalue_error_hartree=float(np.max(np.abs(values_host-reference_values))),
            occupied_mo_residual_joint=float(np.linalg.norm(spin_mo_norms)),
            occupied_ao_commutator_residual_joint=float(np.linalg.norm(spin_commutators)/np.sqrt(2.))))
    native = Path(register_integrals()._name).resolve()
    result = dict(schema_version=1, case_id=cfg.get("case_id"), config_sha256=digest(args.config),
        probe_sha256=digest(__file__), nao=basis.nao, occupied_per_spin=counts,
        input_arrays_sha256={name:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                             for name, value in (("overlap",s_host),("hcore",h_host))},
        overlap_eigenvalue_min=float(spectrum[0]), overlap_condition_number=float(spectrum[-1]/spectrum[0]),
        cutoff=eps, cholesky_eligible=True, variants=variants,
        preparation_seconds=dict(basis=basis_seconds, overlap=overlap_seconds, hcore=hcore_seconds),
        reference_one_electron_energy_hartree=reference_energy,
        scope="Native S/h only, fixed occupations; no SCF, XC, RI or four-center ERIs. No end-to-end speed claim.",
        cholesky_timing_note="Raw SPD Cholesky transform: cutoff eligibility check is outside timing; production guard cost not measured.",
        ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        runtime=dict(python=sys.executable, gradscf_module=gradscf.__file__, jax=jax.__version__, numpy=np.__version__,
            native_path=str(native), native_sha256=digest(native), slurm_job_id=os.environ["SLURM_JOB_ID"],
            affinity=sorted(os.sched_getaffinity(0)), device="cpu", dtype="float64",
            thread_environment={key:os.environ.get(key) for key in
                ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","GRADSCF_NATIVE_THREADS")}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output), metric_errors={v['name']:v['metric_error_frobenius'] for v in variants})))


if __name__ == "__main__":
    main()
