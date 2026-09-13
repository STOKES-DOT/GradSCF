"""Gradient optimization of shared H 3-21G contractions for H2 at fixed geometry.

The two inner-shell coefficients are optimized through a fixed-norm angle;
the single-primitive outer-shell coefficient stays 1. Gaussian exponents do
not change. Every optimizer evaluation is checked against a converged RHF run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable
from scf_coefficient_gradients import make_experiment


def optimize_coefficients(*, bond=.74, scf_steps=20, maxiter=40, gtol=1e-8, verbose=True):
    """Minimize E_RHF using JAX gradients and BFGS with a line search."""
    if bond <= 0 or scf_steps < 1 or maxiter < 1 or gtol <= 0:
        raise ValueError("bond, iteration counts, and gradient tolerance must be positive")
    jax.config.update("jax_enable_x64", True)
    started = time.perf_counter()
    experiment = make_experiment(steps=scf_steps, bond=bond, basis="3-21g")
    initial = experiment["coefficients"]
    if initial.shape != (6,):
        raise ValueError("This experiment expects the H 3-21G two-shell contraction layout")
    radius = jnp.linalg.norm(initial[:2])
    theta0 = float(jnp.arctan2(initial[1], initial[0]))

    def coefficients(theta):
        c = radius*jnp.stack([jnp.cos(theta), jnp.sin(theta)])
        one_atom = jnp.concatenate([c, jnp.ones(1, dtype=c.dtype)])
        return jnp.tile(one_atom, 2)

    def energy(theta):
        return experiment["energy_trace"](coefficients(theta))[-1]

    value_and_grad = jax.jit(jax.value_and_grad(energy))
    energy_only = jax.jit(energy)

    @jax.jit
    def converged_energy(theta):
        s, h, eri = experiment["integral_inputs"](coefficients(theta))
        n = s.shape[0]
        result = run_rks_from_integrals_traceable(
            overlap=s, hcore=h, eri=eri, nelectron=2,
            nuclear_repulsion=experiment["enuc"],
            ao=jnp.zeros((0,n)), ao_deriv1=jnp.zeros((4,0,n)), grid_weights=jnp.zeros(0),
            config=RKSConfig(xc_spec="hf", max_cycle=100, conv_tol=1e-12, conv_tol_density=1e-10),
        )
        return result.total_energy, result.converged, result.cycles

    evaluations = {}

    def evaluate(theta):
        theta = float(theta)
        if theta not in evaluations:
            e, grad = value_and_grad(theta)
            exact, converged, cycles = converged_energy(theta)
            e, grad, exact = float(e), float(grad), float(exact)
            if not bool(converged):
                raise RuntimeError(f"SCF did not converge at angle {theta}")
            difference = abs(e-exact)
            if not np.isfinite(e) or not np.isfinite(grad) or difference > 1e-9:
                raise RuntimeError(f"Unrolled/converged SCF mismatch: {difference:.3e} Ha")
            evaluations[theta] = dict(theta=theta, energy_hartree=e, angle_gradient=grad,
                                      scf_converged=True, scf_cycles=int(cycles),
                                      trace_scf_energy_difference=difference,
                                      c1=float(coefficients(theta)[0]), c2=float(coefficients(theta)[1]), c3=1.)
        return evaluations[theta]

    history = []

    def record(theta):
        row = dict(iteration=len(history), **evaluate(theta))
        history.append(row)
        if verbose:
            print(f"{row['iteration']:3d}  {row['energy_hartree']:.12f}  "
                  f"{row['c1']:.10f}  {row['c2']:.10f}  1.0000000000  "
                  f"{row['angle_gradient']:+.6e}  {row['scf_cycles']:3d}", flush=True)

    if verbose:
        print(f"H2 RHF/3-21G; R={bond} Angstrom; shared raw contractions; fixed exponents", flush=True)
        print("BFGS with analytic JAX gradient; c1=r*cos(theta), c2=r*sin(theta), c3=1", flush=True)
        print("iter  E_total/Ha       c1            c2            c3            dE/dtheta     SCF", flush=True)
    record(theta0)

    def objective(x):
        row = evaluate(x[0])
        return row["energy_hartree"], np.array([row["angle_gradient"]])

    result = minimize(objective, [theta0], jac=True, method="BFGS",
                      callback=lambda x: record(x[0]), options={"gtol":gtol, "maxiter":maxiter})
    final_theta = float(result.x[0])
    final = evaluate(final_theta)
    if history[-1]["theta"] != final_theta:
        record(final_theta)
    # Independently check the optimizer gradient using a four-point stencil.
    def fd(theta, step=1e-4):
        return float((8*(energy_only(theta+step)-energy_only(theta-step))
                      - energy_only(theta+2*step)+energy_only(theta-2*step))/(12*step))
    initial_fd, final_fd = fd(theta0), fd(final_theta)
    np.testing.assert_allclose([history[0]["angle_gradient"], final["angle_gradient"]],
                               [initial_fd, final_fd], atol=1e-7, rtol=1e-5)
    final_c = np.asarray(coefficients(final_theta))
    rows = experiment["rows"][:3]
    optimized_basis = {"H": [[0, [rows[0]["exponent_bohr_minus2"], float(final_c[0])],
                                 [rows[1]["exponent_bohr_minus2"], float(final_c[1])]],
                              [0, [rows[2]["exponent_bohr_minus2"], 1.]] ]}
    summary = dict(
        molecule="H2", bond_angstrom=bond, basis_origin="3-21g", method="RHF",
        backend="jax_reference", optimizer="BFGS", parameterization="fixed-norm shared contraction angle",
        scf_steps=scf_steps, gtol=gtol, optimizer_success=bool(result.success), optimizer_message=str(result.message),
        iterations=int(result.nit), evaluations=len(evaluations),
        initial_coefficients=np.asarray(initial).tolist(), final_coefficients=final_c.tolist(),
        fixed_exponents=[row["exponent_bohr_minus2"] for row in rows],
        initial_energy_hartree=history[0]["energy_hartree"], final_energy_hartree=final["energy_hartree"],
        energy_decrease_hartree=history[0]["energy_hartree"]-final["energy_hartree"],
        initial_angle_gradient=history[0]["angle_gradient"], final_angle_gradient=final["angle_gradient"],
        initial_fd_gradient=initial_fd, final_fd_gradient=final_fd,
        max_trace_scf_energy_difference=max(row["trace_scf_energy_difference"] for row in evaluations.values()),
        optimized_basis=optimized_basis, history=history,
        jax_version=jax.__version__, devices=[str(d) for d in jax.devices()], dtype="float64",
        elapsed_seconds=time.perf_counter()-started,
    )
    if verbose:
        print(f"Optimizer: {result.message}", flush=True)
        print(f"Energy decrease: {summary['energy_decrease_hartree']:.12e} Ha", flush=True)
        print(f"Final dE/dtheta: AD={final['angle_gradient']:+.6e}; FD={final_fd:+.6e}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond", type=float, default=.74)
    parser.add_argument("--scf-steps", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/optimized-321g-contractions"))
    args = parser.parse_args()
    result = optimize_coefficients(bond=args.bond, scf_steps=args.scf_steps, maxiter=args.maxiter)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    (args.output_dir/"optimized_basis.json").write_text(json.dumps(result["optimized_basis"], indent=2)+"\n")
    with (args.output_dir/"optimization.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["history"][0]))
        writer.writeheader()
        writer.writerows(result["history"])
    if not result["optimizer_success"]:
        raise RuntimeError(result["optimizer_message"])


if __name__ == "__main__":
    main()
