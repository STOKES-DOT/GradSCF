"""Replay the exact failed Kr3 coefficients; compare orbital-Hessian solvers."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time
import resource

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.data.molecule import atomic_number
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.runtime import atomic_json, cache_location, file_hash
from gradscf.model.nnao.training.trainer import make_mace_predictor
from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable
from gradscf.scf.stability import _df_rhf_orbital_curvature
from gradscf.solvers import EigenSolverConfig, LinearOperator, solve_hermitian


ROOT = Path("/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021")
OUTPUT = ROOT / "repairs/kr3-20261024"


def main():
    start = time.perf_counter()
    data = json.loads((ROOT / "inputs/references.json").read_text())
    row = next(row for row in data["structures"] if row["structure_id"] == "GMTKN55-v2/RG18/kr3")
    candidate = json.loads((ROOT / "inputs/s-candidate.json").read_text())
    config = dict(json.loads((ROOT / "inputs/model-config.json").read_text()), basis_family="nnao_dzp_s")
    _, predict, _ = make_mace_predictor(0, config)
    checkpoint = ROOT / "runs/nnao_dzp_s/train/last.npz"
    failed = checkpoint.with_name("failed-update-000471.npz")
    with np.load(checkpoint, allow_pickle=False) as saved:
        vector = np.array(saved["parameters"])
    with np.load(failed, allow_pickle=False) as saved:
        np.testing.assert_array_equal(vector, saved["parameters"])
    graph = build_graph([atomic_number(s) for s in row["symbols"]], row["coords_angstrom"],
                        element_order=config["elements"], cutoff=config.get("cutoff", 5.))
    outputs = jax.block_until_ready(predict(jnp.asarray(vector), graph))
    with jax.default_device(jax.devices("cpu")[0]):
        layout = prepare_direct_basis(list(zip(row["symbols"], row["coords_angstrom"])),
                                      basis_family="nnao_dzp_s", core_primitives=data["core_primitives"])
        layout = bind_pool_candidate(layout, candidate)
    problem, manifest = prepare_fixed_pool(row, layout, cache_location(ROOT / "cache/s", row), storage="mmap")
    report = dict(structure=row, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  devices=[str(d) for d in jax.devices()], device_kinds=[d.device_kind for d in jax.devices()], jax=jax.__version__,
                  checkpoint_sha256=file_hash(checkpoint), fixed_exponents=True,
                  finite_differences=False, exact_failed_parameters=True, tests=[])
    atomic_json(OUTPUT / "reproduction.json", report)
    # The saved failure was in the fused value/gradient path; preserve that path.
    (energy, info), gradient = problem._value_grad(outputs, *problem.raw)
    jax.block_until_ready((energy, info, gradient))
    report["original_fused"] = {key: np.asarray(value).tolist() for key, value in info.items()}
    report["original_energy_hartree"] = float(energy)
    report["original_gradient_finite"] = bool(jnp.all(jnp.isfinite(gradient)))
    atomic_json(OUTPUT / "reproduction.json", report)
    physical = jax.jit(problem.project)(outputs, *problem.raw)
    s, h, b = physical
    n, no = s.shape[0], problem.layout.nelectron // 2
    cfg = RKSConfig(xc_spec="hf", jk_backend="df", max_cycle=150,
                    conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9)
    inputs = dict(overlap=s, hcore=h, df_factors=b, eri=None, nelectron=2*no,
                  nuclear_repulsion=problem.enuc, ao=jnp.zeros((0, n)),
                  ao_deriv1=jnp.zeros((4, 0, n)), grid_weights=jnp.zeros(0), config=cfg)
    scf = jax.jit(lambda: run_rks_from_integrals_traceable(**inputs))()
    jax.block_until_ready(scf)
    gradient, apply = _df_rhf_orbital_curvature(h, b, scf.mo_coeff, no)
    eps = jnp.diag(scf.mo_coeff.T @ scf.fock_matrix @ scf.mo_coeff)
    diagonal = (4*(eps[None, no:] - eps[:no, None])).reshape(-1)
    dimension = int(diagonal.size)
    report.update(nao=int(n), nocc=int(no), dimension=dimension,
                  raw_scf_energy_hartree=float(scf.total_energy),
                  orbital_gradient_norm=float(jnp.linalg.norm(gradient)))
    operator = LinearOperator((dimension, dimension), s.dtype, apply, diagonal=diagonal,
                              matmat=jax.vmap(apply, in_axes=1, out_axes=1))
    choices = [("initial", 100, 40, 8), ("current_retry", 500, 160, 16),
               ("expanded_probe", 700, 240, 24)]
    for label, iterations, width, guesses in choices:
        tick = time.perf_counter()
        eigen_config = EigenSolverConfig(method="davidson", nroots=3, atol=1e-7,
            maxiter=iterations, max_subspace=min(width, dimension), initial_guess_count=guesses)
        solved = jax.jit(lambda: solve_hermitian(operator, config=eigen_config))()
        jax.block_until_ready(solved)
        record = dict(label=label, seconds=time.perf_counter()-tick,
                      values=np.asarray(solved.values).tolist(),
                      residuals=np.asarray(solved.residual_norms).tolist(),
                      converged=np.asarray(solved.converged).tolist())
        report["tests"].append(record)
        atomic_json(OUTPUT / "reproduction.json", report)
        print(json.dumps(record), flush=True)
    if dimension <= 4096:
        tick = time.perf_counter()
        # Only a two-index orbital Hessian; no four-index ERI is constructed.
        matrix = jax.jit(lambda: jax.lax.map(apply, jnp.eye(dimension, dtype=s.dtype)).T)()
        jax.block_until_ready(matrix)
        np.savez_compressed(OUTPUT / "orbital-hessian.npz", hessian=np.asarray(matrix),
                            diagonal=np.asarray(diagonal), coefficients=np.asarray(outputs))
        eigen_config = EigenSolverConfig(method="dense", nroots=3, atol=1e-7, max_dense=4096)
        solved = jax.jit(lambda: solve_hermitian(matrix, config=eigen_config))()
        jax.block_until_ready(solved)
        report["dense"] = dict(seconds=time.perf_counter()-tick,
            symmetry_error=float(jnp.linalg.norm(matrix-matrix.T)),
            values=np.asarray(solved.values).tolist(), residuals=np.asarray(solved.residual_norms).tolist(),
            converged=np.asarray(solved.converged).tolist())
    report.update(status="complete", wall_seconds=time.perf_counter()-start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    atomic_json(OUTPUT / "reproduction.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
