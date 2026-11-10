"""Re-evaluate an archived UKS/RI state without another SCF iteration.

Inside Slurm: python scf_residual_certificate.py case.json scf_state.npz output.json
The raw Fock and energy use the production builder at the saved density. This
diagnostic compares two equivalent stationarity measures; it does not change
the original acceptance thresholds or certify stability of the determinant.
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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("state", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Numerical runs require Slurm")
    cfg = json.loads(args.config.read_text())
    archived_input = args.state.with_name("input.json")
    if archived_input.exists() and json.loads(archived_input.read_text()) != cfg:
        parser.error("The state directory input.json differs from the requested configuration")
    if cfg["ks"] != "uks" or cfg["representation"] != "df" or not cfg["auxbasis"]:
        parser.error("This certificate requires the archived explicit-auxiliary UKS/RI case")
    threads = int(cfg["threads"])
    affinity = sorted(os.sched_getaffinity(0))
    if threads < 1 or threads > 64 or len(affinity) < threads:
        parser.error("Require 1--64 allocated logical CPUs")
    if args.output.exists():
        parser.error("Refusing to overwrite a certificate")
    os.sched_setaffinity(0, affinity[:threads])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(cfg["blas_threads"])
    os.environ["GRADSCF_NATIVE_THREADS"] = str(threads)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    os.environ["JAX_ENABLE_X64"] = "1"

    import jax
    import jax.numpy as jnp
    import numpy as np
    import gradscf
    from gradscf.dft.xc import hybrid_coeff, xc_type
    from gradscf.integrals._native import register_integrals
    from gradscf.scf import uks
    from gradscf.scf.core import _orthogonalizer
    from gradscf.scf.inputs import build_uks_integral_inputs
    from gradscf.scf.rsh import local_xc_spec
    from gradscf.solvers.scf import orbital_gradient_norm

    jax.config.update("jax_enable_x64", True)
    start = time.perf_counter()
    config = uks.UKSConfig(xc_spec=cfg["xc"], jk_backend="df", auxbasis=cfg["auxbasis"],
        max_cycle=cfg["max_cycle"], conv_tol=cfg["scf_tol"],
        conv_tol_density=cfg["density_tol"], conv_tol_grad=cfg["gradient_tol"],
        max_memory=cfg["max_memory"], grid_block_size=cfg["grid_block_size"])
    inputs = build_uks_integral_inputs(atom=cfg["atom"], basis=cfg["basis"], config=config,
        unit=cfg["unit"], charge=cfg["charge"], spin=cfg["spin"], cart=cfg["cart"],
        grids_level=cfg["grid_level"], init_guess="hcore")
    jax.block_until_ready((inputs.overlap, inputs.hcore, inputs.df_factors))
    preparation_seconds = time.perf_counter() - start
    with np.load(args.state, allow_pickle=False) as saved:
        coeff, energies, occupations, density = (np.asarray(saved[name])
            for name in ("mo_coeff", "mo_energy", "mo_occ", "density"))
    nao = inputs.overlap.shape[0]
    if (coeff.shape != (2, nao, nao) or density.shape != coeff.shape
            or occupations.shape != (2, nao) or energies.shape != occupations.shape
            or not all(np.isfinite(x).all() and np.isrealobj(x)
                       for x in (coeff, density, occupations, energies))):
        parser.error("Expected finite real complete UKS orbitals and density of the rebuilt AO dimension")
    if not np.all(np.minimum(np.abs(occupations), np.abs(occupations-1.)) < 1e-12):
        parser.error("The commutator/MO norm identity requires occupations 0 or 1")
    evaluate = jax.jit(uks._raw_fock_and_energy_for_state, static_argnames=("cfg", "xc_kind"))
    t0 = time.perf_counter()
    result = evaluate(density_a=jnp.asarray(density[0]), density_b=jnp.asarray(density[1]),
        mo_coeff_a=jnp.asarray(coeff[0]), mo_coeff_b=jnp.asarray(coeff[1]),
        mo_occ_a=jnp.asarray(occupations[0]), mo_occ_b=jnp.asarray(occupations[1]),
        mo_energy_a=jnp.asarray(energies[0]), mo_energy_b=jnp.asarray(energies[1]),
        ao=inputs.ao, ao_deriv1=inputs.ao_deriv1, weights=inputs.grid_weights,
        h=inputs.hcore, eri=inputs.eri, df_factors=inputs.df_factors,
        enuc=jnp.asarray(inputs.nuclear_repulsion), alpha=jnp.asarray(hybrid_coeff(cfg["xc"])),
        cfg=config, xc_kind=xc_type(local_xc_spec(cfg["xc"])), overlap=inputs.overlap,
        eri_lr=inputs.eri_lr, dispersion_energy=inputs.dispersion_energy, grid_ao=inputs.grid_ao)
    jax.block_until_ready(result)
    evaluation_seconds = time.perf_counter() - t0
    energy, xc_energy, fa, fb = result
    fock = np.asarray(jnp.stack((fa, fb)))
    if not np.isfinite(fock).all() or not np.isfinite(float(energy)):
        parser.error("The rebuilt raw Fock or energy is non-finite; stationarity cannot be certified")
    overlap = np.asarray(inputs.overlap)
    x = _orthogonalizer(inputs.overlap, config.orthogonalization_eps)
    commutators = np.asarray(uks._spin_diis_error(jnp.asarray(fock), jnp.asarray(density),
        inputs.overlap, x)).reshape(2, nao, nao)
    eigenvalues = np.linalg.eigvalsh((overlap + overlap.T) * .5)
    x_host = np.asarray(x)
    per_spin = []
    for spin, (f, c, occ, dm, comm) in enumerate(zip(fock, coeff, occupations, density, commutators)):
        occupied = occ > 1e-8
        co, cv = c[:, occupied], c[:, ~occupied]
        fvo = cv.T @ (f @ co)
        fvo_symmetric = cv.T @ (((f+f.T)*.5) @ co)
        # Independent contraction check on the same float64 state, not a
        # higher-precision SCF or a replacement acceptance criterion.
        fvo_extended = cv.astype(np.longdouble).T @ (
            f.astype(np.longdouble) @ co.astype(np.longdouble))
        reconstructed = (c * occ) @ c.T
        per_spin.append(dict(spin=spin, nocc=int(occupied.sum()), nvir=int((~occupied).sum()),
            population=float(np.trace(dm @ overlap)),
            population_target=int((inputs.nalpha, inputs.nbeta)[spin]),
            orthogonality_error=float(np.linalg.norm(c.T @ overlap @ c - np.eye(nao))),
            density_reconstruction_error=float(np.linalg.norm(dm-reconstructed)),
            density_idempotency_error=float(np.linalg.norm(dm @ overlap @ dm-dm)),
            fock_asymmetry=float(np.linalg.norm(f-f.T)),
            mo_vo_norm=float(np.linalg.norm(fvo)),
            symmetric_fock_mo_vo_norm=float(np.linalg.norm(fvo_symmetric)),
            extended_contraction_mo_vo_norm=float(np.linalg.norm(fvo_extended)),
            mo_vo_max_abs=float(np.max(np.abs(fvo))) if fvo.size else 0.,
            commutator_norm_over_sqrt2=float(np.linalg.norm(comm)/np.sqrt(2.))))
    mo_norm = float(np.sqrt(sum(row["mo_vo_norm"]**2 for row in per_spin)))
    comm_norm = float(np.linalg.norm(commutators)/np.sqrt(2.))
    production_norm = float(orbital_gradient_norm(jnp.asarray(fock), jnp.asarray(coeff),
                                                 jnp.asarray(occupations)))
    native = Path(register_integrals()._name).resolve()
    status_path = args.state.with_name("status.json")
    prior_status = json.loads(status_path.read_text()) if status_path.exists() else None
    report = dict(schema_version=2, config_path=str(args.config.resolve()), state_path=str(args.state.resolve()),
        config_sha256=sha256(args.config), state_sha256=sha256(args.state),
        prior_status=prior_status, energy_hartree=float(energy), xc_energy_hartree=float(xc_energy),
        energy_difference_from_archived_hartree=(None if prior_status is None else
            float(energy)-prior_status["energy_hartree"]),
        nao=nao, naux=int(inputs.df_factors.shape[0]), ngrid=int(inputs.grid_weights.shape[0]),
        overlap_eigenvalue_min=float(eigenvalues[0]), overlap_eigenvalue_max=float(eigenvalues[-1]),
        overlap_condition_number=float(eigenvalues[-1]/eigenvalues[0]),
        overlap_clip_epsilon=config.orthogonalization_eps,
        overlap_clipped_count=int(np.count_nonzero(eigenvalues < config.orthogonalization_eps)),
        orthogonalizer_metric_error=float(np.linalg.norm(x_host.T @ overlap @ x_host-np.eye(nao))),
        per_spin=per_spin, mo_vo_gradient_norm=mo_norm,
        ao_commutator_gradient_norm=comm_norm, production_gradient_norm=production_norm,
        production_gradient_representation="occupation-weighted MO commutator",
        residual_norm_difference=comm_norm-mo_norm, gradient_tolerance=cfg["gradient_tol"],
        mo_gradient_pass=mo_norm < cfg["gradient_tol"],
        production_gradient_pass=production_norm < cfg["gradient_tol"],
        acceptance_note="Fixed-density diagnostic only: no delta-energy/density or stability acceptance inferred. A separate JIT can have different rounding from the original SCF executable.",
        identity_assumptions="Real symmetric F, complete S-orthonormal orbitals, integer 0/1 occupations, unclipped SPD S.",
        extended_contraction_mantissa_bits=int(np.finfo(np.longdouble).nmant+1),
        timings=dict(integral_grid_preparation_seconds=preparation_seconds,
            fixed_state_evaluation_including_compile_seconds=evaluation_seconds,
            total_seconds=time.perf_counter()-start),
        ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        runtime=dict(python=sys.executable, gradscf_module=gradscf.__file__, jax=jax.__version__,
            numpy=np.__version__, native_path=str(native), native_sha256=sha256(native),
            source_sha256={name: sha256(Path(uks.__file__).with_name(name))
                           for name in ("uks.py", "rks.py", "iteration.py", "core.py")},
            probe_sha256=sha256(__file__), slurm_job_id=os.environ["SLURM_JOB_ID"],
            affinity=sorted(os.sched_getaffinity(0)),
            thread_environment={key: os.environ.get(key) for key in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GRADSCF_NATIVE_THREADS")},
            device="cpu", dtype="float64"))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), mo_vo_norm=mo_norm, commutator_norm=comm_norm)))


if __name__ == "__main__":
    main()
