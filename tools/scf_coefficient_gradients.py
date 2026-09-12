"""Print dE_k/d(raw contraction coefficient) along an H2 RHF trajectory.

Uses the existing GradSCF SCF scan and JAX reference integrals. Coefficients
are held fixed across SCF iterations, while differentiation includes the
normalization, integrals, hcore initial guess, and all preceding DIIS steps.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import integrals, scf
from gradscf.scf.core import _build_density_from_occ, _diagonalize_fock, _orthogonalizer
from gradscf.scf.rks import RKSConfig, _build_jk, _run_scf_iterations_lax_core


def make_experiment(*, steps=12, bond=.74, basis="3-21g"):
    if steps < 1:
        raise ValueError("steps must be positive")
    topology, params = integrals.prepare_basis(
        atom=f"H 0 0 0; H 0 0 {bond}", basis=basis, unit="Angstrom",
    )
    plan = integrals.make_plan(topology, backend="jax_reference")
    shapes = [c.shape for c in params.coefficients]
    sizes = [c.size for c in params.coefficients]
    offsets = np.cumsum([0] + sizes)
    coefficients = jnp.concatenate([c.ravel() for c in params.coefficients])
    enuc = scf.nuclear_repulsion_energy(params.nuclear_coords, jnp.asarray(topology.nuclear_charges))

    def integral_inputs(vector):
        cs = tuple(vector[offsets[i]:offsets[i+1]].reshape(shape) for i, shape in enumerate(shapes))
        p = replace(params, coefficients=cs)
        s = plan.evaluate("overlap", p)
        h = plan.evaluate("kinetic", p) + plan.evaluate("nuclear", p)
        eri = plan.evaluate("eri", p)
        return s, h, eri

    def energy_trace(vector):
        s, h, eri = integral_inputs(vector)
        x = _orthogonalizer(s, 1e-10)
        eps, coeff = _diagonalize_fock(h, x)
        occ = jnp.zeros(h.shape[0], dtype=h.dtype).at[0].set(2.)
        density = _build_density_from_occ(coeff, occ)

        def energy_and_fock(dm, *_):
            j, k = _build_jk(eri, dm)
            fock = h + j - .5*k
            energy = .5*jnp.einsum("ij,ji->", dm, h+fock) + enuc
            return energy, jnp.zeros_like(energy), fock, j, k

        initial_energy, _, fock, j, k = energy_and_fock(density)
        # A fixed trajectory makes every E_k comparable under FD perturbations:
        # no coefficient-dependent early stopping or extra finalization cycle.
        outputs = _run_scf_iterations_lax_core(
            h=h, s=s, x=x, energy_and_fock_builder=energy_and_fock,
            cfg=RKSConfig(xc_spec="hf", max_cycle=steps, conv_tol=0., damping=0., level_shift=0.),
            mo_occ_fixed=occ, diis_basis=coeff, skip_first_fock_damping=False,
            density=density, mo_coeff=coeff, mo_occ=occ, mo_energy=eps,
            raw_fock=fock, j_mat=j, k_mat=k,
        )
        density_history = outputs[10]
        energies = jax.vmap(lambda dm: energy_and_fock(dm)[0])(density_history)
        return jnp.concatenate([initial_energy[None], energies])

    rows, contraction_indices, shared = [], [], {}
    shell_counts = [0, 0]
    for shell, (shape, a, center) in enumerate(zip(shapes, params.exponents, params.centers)):
        atom = int(np.argmin(np.linalg.norm(np.asarray(params.nuclear_coords-center), axis=1)))
        local_shell = shell_counts[atom]
        shell_counts[atom] += 1
        for ctr in range(shape[1]):
            contraction_indices.append([int(offsets[shell]+p*shape[1]+ctr) for p in range(shape[0])])
        for primitive in range(shape[0]):
            for ctr in range(shape[1]):
                index = int(offsets[shell]+primitive*shape[1]+ctr)
                name = f"H{atom+1}.shell{local_shell+1}.p{primitive+1}.c{ctr+1}"
                rows.append(dict(index=index, label=name, atom=atom+1, shell=local_shell+1,
                                 primitive=primitive+1, contraction=ctr+1,
                                 exponent_bohr_minus2=float(a[primitive]), coefficient=float(coefficients[index])))
                shared.setdefault(f"H.shell{local_shell+1}.p{primitive+1}.c{ctr+1}", []).append(index)
    return dict(energy_trace=energy_trace, integral_inputs=integral_inputs, coefficients=coefficients,
                enuc=enuc, rows=rows, shared=shared, contraction_indices=contraction_indices,
                steps=steps, bond=bond, basis=basis)


def finite_difference_jacobian(fn, coefficients, *, step=1e-4):
    """Four-point central differences for the same fixed-length SCF map."""
    columns = []
    for i in range(coefficients.size):
        d = jnp.zeros_like(coefficients).at[i].set(step)
        columns.append(np.asarray((8*(fn(coefficients+d)-fn(coefficients-d))
                                   - fn(coefficients+2*d)+fn(coefficients-2*d))/(12*step)))
    return np.stack(columns, axis=1)


def reference_rhf_result(experiment):
    s, h, eri = experiment["integral_inputs"](experiment["coefficients"])
    return scf.run_rhf_from_integrals(
        overlap=s, hcore=h, eri=eri, nelectron=2, nuclear_repulsion=experiment["enuc"],
        config=scf.RHFConfig(max_cycle=80, conv_tol=1e-12),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--bond", type=float, default=.74, help="H-H distance in Angstrom")
    parser.add_argument("--basis", default="3-21g")
    parser.add_argument("--fd-step", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/scf-coefficient-gradients"))
    args = parser.parse_args()
    if args.fd_step <= 0 or args.bond <= 0:
        parser.error("fd-step and bond must be positive")
    jax.config.update("jax_enable_x64", True)
    start = time.perf_counter()
    experiment = make_experiment(steps=args.steps, bond=args.bond, basis=args.basis)
    c = experiment["coefficients"]
    fn = jax.jit(experiment["energy_trace"])
    print(f"H2 RHF/{args.basis}; R={args.bond} Angstrom; hcore guess; fixed {args.steps} DIIS iterations", flush=True)
    print("Gradient: total dE_k/d(raw c), including contraction normalization and iteration history", flush=True)
    for row in experiment["rows"]:
        print(f"c{row['index']+1}: {row['label']} alpha={row['exponent_bohr_minus2']:.10g} c={row['coefficient']:.10g}", flush=True)
    energies = np.asarray(fn(c))
    jac = np.asarray(jax.jit(jax.jacrev(experiment["energy_trace"]))(c))
    print("step    E_total/Ha       delta_E/Ha      " + " ".join(f"dE/dc{i+1: <2}" for i in range(c.size)), flush=True)
    for k, (energy, gradient) in enumerate(zip(energies, jac)):
        delta = 0. if k == 0 else energy-energies[k-1]
        print(f"{k:3d}  {energy: .12f}  {delta:+.3e}  " + " ".join(f"{g:+.8e}" for g in gradient), flush=True)
    fd = finite_difference_jacobian(fn, c, step=args.fd_step)
    error = np.max(np.abs(jac-fd), axis=1)
    reference = reference_rhf_result(experiment)
    if not reference.converged:
        raise RuntimeError("Reference RHF run did not converge")
    shared = np.stack([jac[:, indices].sum(axis=1) for indices in experiment["shared"].values()], axis=1)
    print("Shared-H basis gradients: " + " ".join(experiment["shared"]), flush=True)
    for k, row in enumerate(shared):
        print(f"{k:3d}  " + " ".join(f"{g:+.8e}" for g in row), flush=True)
    print(f"Max AD-FD error: {error.max():.3e} Ha / raw coefficient", flush=True)
    print(f"Final energy difference from converged RHF: {abs(energies[-1]-reference.total_energy):.3e} Ha", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/"trace.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "energy_hartree", *[r["label"] for r in experiment["rows"]], "max_ad_fd_error"])
        for k in range(len(energies)):
            writer.writerow([k, energies[k], *jac[k], error[k]])
    summary = dict(molecule="H2", bond_angstrom=args.bond, basis=args.basis, method="RHF",
                   backend="jax_reference", devices=[str(d) for d in jax.devices()], jax_version=jax.__version__,
                   dtype="float64", steps=args.steps, initialization="hcore", coefficients=experiment["rows"],
                   gradient_semantics="total derivative through normalized basis, hcore guess, and fixed SCF/DIIS history",
                   fd_step=args.fd_step, fd_stencil="4-point central", max_ad_fd_error=float(error.max()),
                   reference_energy_hartree=float(reference.total_energy), shared_labels=list(experiment["shared"]),
                   shared_gradients=shared.tolist(), elapsed_seconds=time.perf_counter()-start)
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    np.testing.assert_allclose(jac, fd, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    main()
