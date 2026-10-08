"""Differentiable CIS wavefunction probes for H2O/6-31G*.

The example uses three scalar targets because a vector-valued wavefunction does
not have a unique gradient until a scalar probe is selected:

* a CIS transition-amplitude probe differentiated with respect to nuclear
  coordinates;
* the same probe differentiated with respect to the CIS single-excitation
  amplitudes; and
* the probe differentiated with respect to occupied--virtual rotations of the
  reference orbitals.

The nuclear derivative differentiates through the traceable RHF reference and
the geometry-dependent AO integrals.  The CIS root is therefore evaluated on a
geometry-dependent HF orbital frame, while the scalar probe below removes the
irrelevant global sign of an eigenvector.

Coordinates are in Bohr internally, energies in Hartree, and gradients in the
corresponding Hartree/Bohr or probe units.  No PySCF or command-line interface
is used.
"""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto, integrals
from gradscf.ci.solver import solve_cis
from gradscf.ci.types import CIConfig
from gradscf.scf.rhf import nuclear_repulsion_energy
from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable


BASIS = "6-31g*"
ATOM = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
NROOTS = 1
N_OCC = 5


def _givens_rotation(theta, nmo, pairs):
    """Build an orthogonal orbital rotation from a few Givens angles."""
    rotation = jnp.eye(nmo)
    for angle, (occupied, virtual) in zip(theta, pairs):
        cosine = jnp.cos(angle)
        sine = jnp.sin(angle)
        givens = jnp.eye(nmo)
        givens = givens.at[occupied, occupied].set(cosine)
        givens = givens.at[virtual, virtual].set(cosine)
        givens = givens.at[occupied, virtual].set(sine)
        givens = givens.at[virtual, occupied].set(-sine)
        rotation = rotation @ givens
    return rotation


def main():
    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom")
    hf = dft.RKS(
        molecule,
        xc="hf",
        integral_backend="cpu",
        conv_tol=1.0e-10,
        max_cycle=100,
    ).run()
    if not hf.converged:
        raise RuntimeError("RHF did not converge")

    topology, basis_parameters = integrals.prepare_basis(ATOM, basis=BASIS)
    plan = integrals.make_plan(topology, backend="native")
    coordinates = jnp.asarray(basis_parameters.nuclear_coords)
    centers = jnp.asarray(basis_parameters.centers)
    shell_owners = np.argmin(
        np.linalg.norm(
            np.asarray(centers)[:, None, :] - np.asarray(coordinates)[None, :, :],
            axis=2,
        ),
        axis=1,
    )
    reference_coefficients = jnp.asarray(hf.mo_coeff)
    nmo = int(reference_coefficients.shape[1])

    def ao_integrals(current_coordinates):
        current_parameters = replace(
            basis_parameters,
            centers=current_coordinates[shell_owners],
            nuclear_coords=current_coordinates,
        )
        overlap = plan.evaluate("overlap", current_parameters)
        hcore = plan.evaluate("kinetic", current_parameters)
        hcore = hcore + plan.evaluate("nuclear", current_parameters)
        eri_ao = plan.evaluate("eri", current_parameters)
        return overlap, hcore, eri_ao

    def transform_to_mo(hcore, eri_ao, coefficients):
        h1 = coefficients.T @ hcore @ coefficients
        eri = jnp.einsum(
            "pi,qj,rk,sl,pqrs->ijkl",
            coefficients,
            coefficients,
            coefficients,
            coefficients,
            eri_ao,
            precision=jax.lax.Precision.HIGHEST,
        )
        return h1, eri

    def mo_integrals(current_coordinates, coefficients):
        _, hcore, eri_ao = ao_integrals(current_coordinates)
        return transform_to_mo(hcore, eri_ao, coefficients)

    cis_config = CIConfig(
        nroots=NROOTS,
        solver="dense",
        conv_tol=1.0e-10,
        gradient_mode="implicit_eigenvector",
    )

    _, hcore, eri_ao = ao_integrals(coordinates)
    h1, eri = transform_to_mo(hcore, eri_ao, reference_coefficients)
    cis = solve_cis(h1, eri, nocc=N_OCC, config=cis_config)
    amplitudes = cis.amplitudes[0]
    probe = jnp.linspace(0.15, 1.0, amplitudes.size, dtype=amplitudes.dtype)
    probe = probe.reshape(amplitudes.shape)

    def wavefunction_probe(single_amplitudes):
        """A sign-invariant scalar CIS-state probe."""
        return jnp.sum(probe * single_amplitudes**2) + 0.1 * jnp.sum(
            single_amplitudes**4
        )

    # 1. Nuclear-coordinate derivative of the CIS wavefunction probe.
    def geometry_probe(current_coordinates):
        overlap, hcore, eri_ao = ao_integrals(current_coordinates)
        nao = int(hcore.shape[0])
        scf = run_rks_from_integrals_traceable(
            overlap=overlap,
            hcore=hcore,
            eri=eri_ao,
            nelectron=10,
            nuclear_repulsion=nuclear_repulsion_energy(
                current_coordinates,
                jnp.asarray(topology.nuclear_charges, dtype=current_coordinates.dtype),
            ),
            ao=jnp.zeros((0, nao), dtype=hcore.dtype),
            ao_deriv1=jnp.zeros((4, 0, nao), dtype=hcore.dtype),
            grid_weights=jnp.zeros((0,), dtype=hcore.dtype),
            config=RKSConfig(
                xc_spec="hf",
                max_cycle=40,
                conv_tol=1.0e-10,
                conv_tol_density=1.0e-9,
            ),
        )
        current_h1, current_eri = transform_to_mo(hcore, eri_ao, scf.mo_coeff)
        current_cis = solve_cis(
            current_h1, current_eri, nocc=N_OCC, config=cis_config
        )
        return wavefunction_probe(current_cis.amplitudes[0])

    geometry_value, geometry_gradient = jax.value_and_grad(geometry_probe)(coordinates)

    # 2. Direct derivative with respect to the CIS single-excitation coefficients.
    amplitude_value, amplitude_gradient = jax.value_and_grad(wavefunction_probe)(
        amplitudes
    )

    # 3. Reference-state coefficient derivative represented on the orthogonal
    # occupied--virtual tangent space.  Raw orbital coefficients are constrained
    # by C^T S C = I, so unconstrained elementwise derivatives are gauge dependent.
    pairs = tuple((i, N_OCC + i) for i in range(min(N_OCC, nmo - N_OCC, 3)))
    theta0 = jnp.zeros((len(pairs),), dtype=coordinates.dtype)

    def reference_coefficient_probe(coefficients):
        reference_h1, reference_eri = transform_to_mo(hcore, eri_ao, coefficients)
        reference_cis = solve_cis(
            reference_h1, reference_eri, nocc=N_OCC, config=cis_config
        )
        return wavefunction_probe(reference_cis.amplitudes[0])

    reference_value, reference_coefficient_gradient = jax.value_and_grad(
        reference_coefficient_probe
    )(reference_coefficients)

    def reference_rotation_probe(theta):
        rotation = _givens_rotation(theta, nmo, pairs)
        return reference_coefficient_probe(reference_coefficients @ rotation)

    _, reference_rotation_gradient = jax.value_and_grad(reference_rotation_probe)(theta0)

    # A small finite-difference check for the coordinate derivative.
    step = 1.0e-4
    displacement = jnp.zeros_like(coordinates).at[1, 2].set(step)
    geometry_fd = (
        geometry_probe(coordinates + displacement)
        - geometry_probe(coordinates - displacement)
    ) / (2.0 * step)

    report = {
        "molecule": "H2O",
        "basis": "6-31G*",
        "geometry_unit": "Angstrom input / Bohr differentiation",
        "energy_unit": "Hartree",
        "hf_energy": float(hf.e_tot),
        "nmo": nmo,
        "nocc": N_OCC,
        "cis_excitation_energy": float(cis.excitation_energies[0]),
        "cis_residual_norm": float(cis.residual_norms[0]),
        "cis_converged": bool(cis.converged[0]),
        "wavefunction_probe": float(geometry_value),
        "nuclear_coordinate_gradient": np.asarray(geometry_gradient).tolist(),
        "nuclear_coordinate_gradient_units": "probe / Bohr",
        "coordinate_gradient_fd_abs_error": abs(
            float(geometry_gradient[1, 2] - geometry_fd)
        ),
        "amplitude_shape": list(amplitudes.shape),
        "amplitude_probe": float(amplitude_value),
        "amplitude_gradient": np.asarray(amplitude_gradient).tolist(),
        "reference_coefficient_shape": list(reference_coefficients.shape),
        "reference_coefficient_gradient": np.asarray(
            reference_coefficient_gradient
        ).tolist(),
        "reference_rotation_pairs": [list(pair) for pair in pairs],
        "reference_probe": float(reference_value),
        "reference_rotation_gradient": np.asarray(reference_rotation_gradient).tolist(),
        "reference_rotation_units": "probe / radian",
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()


# Example output (CPU, JAX float64; full coefficient arrays are printed by the
# script):
# {
#   "molecule": "H2O",
#   "basis": "6-31G*",
#   "hf_energy": -76.01050498825722,
#   "nmo": 19,
#   "nocc": 5,
#   "cis_excitation_energy": 0.35252320362118794,
#   "cis_residual_norm": 5.454652003230702e-15,
#   "cis_converged": true,
#   "wavefunction_probe": 0.9376616535544786,
#   "nuclear_coordinate_gradient_units": "probe / Bohr",
#   "nuclear_coordinate_gradient": [
#     [-7.24e-18, -1.07e-14, -2.8691047516571106e-03],
#     [ 4.77e-18, -1.7496012213160670e-03,  1.4345523758246348e-03],
#     [ 2.47e-18,  1.7496012213266948e-03,  1.4345523758328396e-03]
#   ],
#   "coordinate_gradient_fd_abs_error": 1.1817863831628528e-09,
#   "amplitude_shape": [5, 14],
#   "amplitude_probe": 0.9376616535152971,
#   "reference_coefficient_shape": [19, 19],
#   "reference_rotation_pairs": [[0, 5], [1, 6], [2, 7]],
#   "reference_probe": 0.9376616535152971,
#   "reference_rotation_gradient": [
#     -5.843741354601801e-04,
#      5.692958848742402e-15,
#     -2.0729042718354604e-03
#   ],
#   "reference_rotation_units": "probe / radian"
# }
