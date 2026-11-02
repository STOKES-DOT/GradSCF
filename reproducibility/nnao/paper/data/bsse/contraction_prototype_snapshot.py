#!/usr/bin/env python
"""Differentiate counterpoise BSSE with respect to shared STO-3G contractions.

Fixed water-dimer geometry/exponents/core; shared H 1s and O 2s/2p coefficients.
Native primitive integrals are fixed; JAX differentiates their contractions.
Closed-shell HF energies use the public implicit shared-orbital SCF minimizer.
"""
import json
import os
from functools import lru_cache
from dataclasses import replace
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from gradscf import gto, integrals, scf
from gradscf.data.molecule import parse_molecule_spec
from gradscf.integrals.contraction import (
    primitive_basis, contraction_matrix, contract_integrals,
)

HARTREE_TO_KCAL_MOL = 627.5094740631
CHANNELS = {("H", 0): 0, ("O", 1): 1, ("O", 2): 2}
CHANNEL_NAMES = ("H 1s", "O 2s", "O 2p")


def make_problem(geometry_file=None):
    """Prepare five fixed-geometry components; no native integral AD is needed."""
    path = geometry_file or Path(__file__).with_name("s22_hbond_geometries.json")
    system = json.loads(Path(path).read_text())["systems"]["Water_dimer"]
    atom = list(zip(system["symbols"], system["positions_angstrom"]))
    a, b = system["fragments"]
    states = ((atom, ()), ([atom[i] for i in a], ()), ([atom[i] for i in b], ()),
              (atom, b), (atom, a))
    components, reference = [], [None] * 3
    for atoms, ghosts in states:
        spec = parse_molecule_spec(atoms, unit="Angstrom")
        charges = np.asarray(spec.charges).copy()
        charges[list(ghosts)] = 0
        spec = replace(spec, charges=charges)
        topology, parameters = integrals.prepare_basis(spec, "sto-3g", cart=True)
        owners = np.argmin(np.linalg.norm(
            np.asarray(parameters.centers)[:, None] - np.asarray(spec.coords_bohr)[None],
            axis=2), axis=1)
        shell_counts, channel_indices = {}, []
        for owner, coefficients in zip(owners, parameters.coefficients):
            shell = shell_counts.get(int(owner), 0)
            shell_counts[int(owner)] = shell + 1
            channel = CHANNELS.get((spec.symbols[owner], shell))
            channel_indices.append(channel)
            if channel is not None:
                raw = np.asarray(coefficients[:, 0])
                normalized = raw / np.linalg.norm(raw)
                if reference[channel] is None:
                    reference[channel] = normalized
                else:
                    np.testing.assert_allclose(reference[channel], normalized, atol=1e-14)
        prim_topology, prim_parameters = primitive_basis(topology, parameters)
        plan = integrals.make_plan(prim_topology, backend="native")
        tensors = {
            "overlap": plan.evaluate("overlap", prim_parameters),
            "hcore": plan.evaluate("kinetic", prim_parameters)
                     + plan.evaluate("nuclear", prim_parameters),
            "eri": plan.evaluate("eri", prim_parameters),
        }
        seed = scf.RHF(
            gto.M(atom=spec, basis="sto-3g", cart=True),
            conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9,
            max_cycle=200,
        ).run()
        if not seed.converged:
            raise RuntimeError("Initial RHF did not converge.")
        occupations = np.zeros((2, topology.nao))
        occupations[:, :spec.nelectron // 2] = 1
        components.append(dict(
            topology=topology, parameters=parameters, tensors=tensors,
            channels=channel_indices, seed=jnp.asarray(seed.mo_coeff),
            occupations=occupations, enuc=spec.nuclear_repulsion,
            initial_energy=float(seed.e_tot),
        ))
    reference = jnp.asarray(np.stack(reference))
    # Fix the first coefficient of each unit-scale reference vector to remove
    # its common scale freedom; the other two coefficients are optimization variables.
    x0 = np.asarray(reference[:, 1:]).reshape(-1)

    def coefficients(x):
        raw = jnp.concatenate([reference[:, :1], jnp.asarray(x).reshape(3, 2)], axis=1)
        return raw / jnp.linalg.norm(raw, axis=1, keepdims=True)

    def energies(x):
        shared = coefficients(x)
        values, stationary, norms, min_overlap = [], [], [], []
        for component in components:
            coeffs = tuple(
                c if channel is None else shared[channel, :, None]
                for c, channel in zip(component["parameters"].coefficients,
                                      component["channels"])
            )
            parameters = replace(component["parameters"], coefficients=coeffs)
            transform = contraction_matrix(component["topology"], parameters)
            data = contract_integrals(component["tensors"], transform)
            n = data["overlap"].shape[0]
            result = scf.minimize_roks_from_integrals(
                overlap=data["overlap"], hcore=data["hcore"], eri=data["eri"],
                nuclear_repulsion=component["enuc"],
                ao=jnp.zeros((0, n)), ao_deriv1=jnp.zeros((4, 0, n)),
                grid_weights=jnp.zeros(0), mo_coeff=component["seed"],
                mo_occ=component["occupations"], xc_spec="hf",
                orthonormalize_initial=True, max_iterations=300,
                gradient_tolerance=1e-9,
                differentiation=scf.SCFDifferentiationConfig(
                    mode="implicit", tolerance=1e-9, max_iter=100, restart=40),
            )
            # Equal alpha/beta occupations and shared spatial orbitals give RHF.
            values.append(result.total_energy)
            stationary.append(result.stationary)
            norms.append(result.gradient_norm)
            min_overlap.append(jnp.linalg.eigvalsh(data["overlap"])[0])
        return jnp.stack(values), dict(
            stationary=jnp.stack(stationary), orbital_gradient_norm=jnp.stack(norms),
            min_overlap=jnp.stack(min_overlap),
        )

    return dict(x0=x0, reference=reference, coefficients=coefficients,
                energies=energies, components=components)


def optimize(problem, *, maxiter=80, energy_budget=1e-4, verbose=True):
    """Reduce BSSE while limiting each own-basis fragment energy increase."""
    def energy_with_aux(x):
        energies, info = problem["energies"](x)
        return energies, dict(info, energy=energies)

    derivative = jax.jit(jax.jacrev(energy_with_aux, has_aux=True))

    @lru_cache(maxsize=1)
    def evaluate(x_tuple):
        jacobian, info = derivative(jnp.asarray(x_tuple))
        energies, jacobian = np.asarray(info["energy"]), np.asarray(jacobian)
        if (not np.asarray(info["stationary"]).all()
                or not np.isfinite(energies).all() or not np.isfinite(jacobian).all()
                or np.min(np.asarray(info["min_overlap"])) <= 1e-8):
            raise RuntimeError("Invalid SCF state or implicit contraction gradient.")
        return energies, jacobian, info

    initial, _, _ = evaluate(tuple(problem["x0"]))

    def objective(x):
        energy, jacobian, _ = evaluate(tuple(x))
        weights = np.array([0., 1., 1., -1., -1.])
        return float(weights @ energy * HARTREE_TO_KCAL_MOL), (
            weights @ jacobian * HARTREE_TO_KCAL_MOL)

    constraints = dict(
        type="ineq",
        fun=lambda x: initial[1:3] + energy_budget - evaluate(tuple(x))[0][1:3],
        jac=lambda x: -evaluate(tuple(x))[1][1:3],
    )
    history = []

    def record(x):
        energy, _, info = evaluate(tuple(x))
        row = dict(bsse_kcal_mol=objective(x)[0],
                   max_fragment_energy_increase_hartree=float(
                       np.max(energy[1:3] - initial[1:3])),
                   max_orbital_gradient=float(np.max(info["orbital_gradient_norm"])))
        history.append(row)
        if verbose:
            print(f"{len(history):3d}  BSSE={row['bsse_kcal_mol']:.8f} kcal/mol"
                  f"  fragment increase={row['max_fragment_energy_increase_hartree']:.3e} Ha",
                  flush=True)

    result = minimize(
        objective, problem["x0"], jac=True, method="SLSQP",
        bounds=[(float(x)-.15, float(x)+.15) for x in problem["x0"]],
        constraints=constraints, callback=record,
        options=dict(maxiter=maxiter, ftol=1e-8, disp=verbose),
    )
    final, _, info = evaluate(tuple(result.x))
    if np.max(final[1:3] - initial[1:3]) > energy_budget + 1e-8:
        raise RuntimeError("Final fragment energy budget was not satisfied.")
    return dict(
        success=bool(result.success), message=str(result.message),
        iterations=int(result.nit),
        initial_bsse_kcal_mol=objective(problem["x0"])[0],
        final_bsse_kcal_mol=objective(result.x)[0],
        initial_energies_hartree=initial.tolist(), final_energies_hartree=final.tolist(),
        fragment_energy_budget_hartree=energy_budget,
        initial_coefficients=np.asarray(problem["coefficients"](problem["x0"])).tolist(),
        final_coefficients=np.asarray(problem["coefficients"](result.x)).tolist(),
        final_parameters=result.x.tolist(),
        final_min_overlap_eigenvalue=float(np.min(info["min_overlap"])),
        final_max_orbital_gradient=float(np.max(info["orbital_gradient_norm"])),
        history=history,
    )


if __name__ == "__main__":
    problem = make_problem()
    result = optimize(problem)
    print(json.dumps(result, indent=2))
