"""Check total RHF geometry gradients against finite differences and rigid-motion invariance."""
from __future__ import annotations

import argparse
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
from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable

SYSTEMS = {
    # A rotated, translated 0.74 Angstrom bond exercises all Cartesian axes.
    "h2": ("H .1 -.2 .3; H .5933333333333333 -.4466666666666667 .7933333333333333", "3-21g"),
    "water": ("O 0 0 0; H 0 .76 .58; H 0 -.76 .58", "sto-3g"),
}


def check_geometry(molecule="h2", *, steps=30, fd_step=1e-4, backend="native"):
    if steps < 1 or fd_step <= 0:
        raise ValueError("steps and fd_step must be positive")
    jax.config.update("jax_enable_x64", True)
    started = time.perf_counter()
    atom, basis = SYSTEMS[molecule]
    topology, params = integrals.prepare_basis(atom=atom, basis=basis, unit="Angstrom")
    plan = integrals.make_plan(topology, backend=backend)
    shell_atoms = np.argmin(np.linalg.norm(np.asarray(params.centers)[:, None, :]
                                          - np.asarray(params.nuclear_coords)[None, :, :], axis=2), axis=1)
    charges = jnp.asarray(topology.nuclear_charges)
    nelectron = sum(topology.nuclear_charges)

    def energy(coords):
        moving = replace(params, nuclear_coords=coords, centers=coords[shell_atoms])
        s = plan.evaluate("overlap", moving)
        h = plan.evaluate("kinetic", moving)+plan.evaluate("nuclear", moving)
        eri = plan.evaluate("eri", moving)
        enuc = scf.nuclear_repulsion_energy(coords, charges)
        n = s.shape[0]
        # Fixed iteration count avoids coefficient/geometry-dependent stopping.
        result = run_rks_from_integrals_traceable(
            overlap=s, hcore=h, eri=eri, nelectron=nelectron, nuclear_repulsion=enuc,
            ao=jnp.zeros((0,n)), ao_deriv1=jnp.zeros((4,0,n)), grid_weights=jnp.zeros(0),
            config=RKSConfig(xc_spec="hf", max_cycle=steps, conv_tol=0.))
        f, d = result.fock_matrix, result.density_matrix
        residual = jnp.max(jnp.abs(f@d@s-s@d@f))
        return result.total_energy, residual

    coords = params.nuclear_coords
    print(f"{molecule} RHF/{basis}, {topology.nao} AOs, {steps} fixed SCF steps; {backend}, float64", flush=True)
    (total, residual), grad = jax.jit(jax.value_and_grad(energy, has_aux=True))(coords)
    grad = np.asarray(grad)
    np.testing.assert_array_equal(np.isfinite(grad), True)
    print(f"E={float(total):.12f} Ha; SCF commutator={float(residual):.3e}", flush=True)
    forward = jax.jit(energy)
    residuals = [float(residual)]

    def value(r):
        e, error = forward(r)
        residuals.append(float(error))
        return float(e)

    fd = np.zeros_like(grad)
    for i in range(len(coords)):
        for axis in range(3):
            delta = jnp.zeros_like(coords).at[i, axis].set(fd_step)
            fd[i, axis] = (8*(value(coords+delta)-value(coords-delta))
                           - value(coords+2*delta)+value(coords-2*delta))/(12*fd_step)
    fd_error = float(np.max(np.abs(grad-fd)))
    translation = np.sum(grad, axis=0)
    torque = np.sum(np.cross(np.asarray(coords)-np.asarray(coords).mean(axis=0), grad), axis=0)
    summary = dict(molecule=molecule, basis=basis, method="RHF", backend=backend,
                   atom_angstrom=atom, coordinates_bohr=np.asarray(coords).tolist(),
                   gradient_units="Hartree/Bohr", gradient=grad.tolist(), forces=(-grad).tolist(),
                   fd_gradient=fd.tolist(),
                   energy_hartree=float(total),
                   max_ad_fd_error=fd_error,
                   translation_sum=translation.tolist(), torque=torque.tolist(),
                   max_scf_commutator=max(residuals), steps=steps, fd_step_bohr=fd_step,
                   fd_stencil="four-point central", jax_version=jax.__version__,
                   dtype="float64", devices=[str(d) for d in jax.devices()],
                   elapsed_seconds=time.perf_counter()-started)
    np.testing.assert_allclose(max(residuals), 0., atol=1e-8)
    np.testing.assert_allclose(grad, fd, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(translation, 0., atol=1e-8)
    np.testing.assert_allclose(torque, 0., atol=1e-8)
    for i, row in enumerate(grad):
        print(f"atom {i+1}: "+" ".join(f"{g:+.10e}" for g in row)+" Ha/Bohr", flush=True)
    print(f"AD-FD={fd_error:.3e}; "
          f"elapsed={summary['elapsed_seconds']:.2f}s", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=SYSTEMS, default="h2")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--backend", choices=("native", "jax_reference"), default="native")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/geometry-gradients"))
    args = parser.parse_args()
    result = check_geometry(args.molecule, steps=args.steps, backend=args.backend)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/f"{args.molecule}-{args.backend}.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
