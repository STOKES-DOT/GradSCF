"""Full-TDDFT gap/dipole AD through native integrals, grids and relaxed SCF.

The signed finite differences independently reconverge SCF at both geometries.
Distances are bohr, energies Hartree and electronic transition dipoles e*bohr.
No external electronic-structure package supplies a value or derivative.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, gto, tdscf
from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule
from gradscf_tools.molecular_ep import NativeRHF
from gradscf_tools.molecular_response import MolecularEnergy


def _mean_field(xc, coordinates, symbols, charge):
    mol = gto.M(atom=list(zip(symbols, np.asarray(coordinates))),
                basis="sto-3g", unit="Bohr", charge=charge, cart=True)
    mf = dft.RKS(mol, xc=xc, grids_level=0, jk_backend="direct",
        direct_scf_tol=0., direct_scf_workers=1, grid_screen_tol=0.,
        conv_tol=1e-13, conv_tol_density=1e-11, conv_tol_grad=1e-10,
        max_cycle=150).run()
    assert mf.converged
    return mf


def _moving_reference(model, coordinates, coefficients):
    ground, inputs = model.solve_with_inputs(coordinates, solution=coefficients)
    moving = replace(model.parameters, centers=coordinates[model.owners],
                     nuclear_coords=coordinates)
    position = model.plan.evaluate("dipole", moving, origin=jnp.zeros(3))
    fock = jnp.mean(ground.fock_matrix, axis=0)
    energies = jnp.diag(ground.mo_coeff.T @ fock @ ground.mo_coeff)
    empty = jnp.zeros((0, 0, 0, 0), dtype=coordinates.dtype)
    reference = RestrictedMolecule(
        ao=inputs["ao"], ao_deriv1=inputs["ao_deriv1"],
        grid_ao=inputs.get("grid_ao"), grid=QuadratureGrid(inputs["grid_weights"]),
        dipole_integrals=position, rep_tensor=empty,
        mo_coeff=jnp.stack((ground.mo_coeff, ground.mo_coeff)),
        mo_occ=ground.mo_occ, mo_energy=jnp.stack((energies, energies)),
        rdm1=ground.density_matrix, h1e=inputs["hcore"],
        nuclear_repulsion=inputs["nuclear_repulsion"], atom_coords=coordinates,
        overlap_matrix=inputs["overlap"], fock_matrix=fock,
        direct_basis=inputs["direct_basis"], mf_energy=ground.total_energy,
        nocc=1, xc_spec=model.xc, exact_exchange_fraction=float(model.xc == "hf"))
    return reference, ground


def _response(model, coordinates, coefficients):
    reference, ground = _moving_reference(model, coordinates, coefficients)
    driver = tdscf.TDDFT(reference, nstates=1, davidson_tol=1e-11,
        davidson_max_iter=100, tda_gradient_mode="implicit_eigenvector",
        eigenvector_adjoint_tol=1e-11, eigenvector_adjoint_max_iter=100,
        response_kernel_options={"max_memory": 256., "integral_cache": "none",
            "direct_tol": 0., "direct_workers": 1, "grid_block_size": 128})
    result = driver.kernel()
    values = jnp.concatenate((result.excitation_energies, driver.transition_dipole().reshape(-1)))
    valid = ground.stationary & jnp.all(result.converged)
    return jnp.where(valid, values, jnp.nan), (
        valid, reference.mo_coeff[0], result.x_amplitudes+result.y_amplitudes,
        ground.density_matrix, ground.gradient_norm)


@pytest.mark.parametrize("xc", ["hf", "pbe", "b3lyp"])
def test_full_tddft_gap_and_signed_transition_dipole_geometry_ad(xc, record_property):
    coordinates = jnp.array([[0., 0., -.8], [0., 0., .8]])
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    # Heteronuclear HeH+ additionally exercises occupied/virtual SCF mixing;
    # H2 stretching alone constrains that response by inversion symmetry.
    symbols, charge = (("He", "H"), 1) if xc == "hf" else (("H", "H"), 0)
    mf = _mean_field(xc, coordinates, symbols, charge)
    model = MolecularEnergy(mf, forward="reference", jk_backend="direct",
                            gradient_tolerance=1e-8)
    coefficients = jnp.asarray(mf.mo_coeff)
    function = lambda q, c: _response(model, coordinates+q*direction, c)
    evaluate = jax.jit(function)
    values, anchor = evaluate(jnp.asarray(0.), coefficients)
    assert bool(anchor[0]) and np.isfinite(values).all()
    assert values.shape == (4,)
    assert abs(float(values[3])) > .1, "The axial transition must exercise a nonzero dipole"

    outputs = lambda q, c: function(q, c)[0]
    _, tangent = jax.jit(lambda q, c: jax.jvp(
        lambda displacement: outputs(displacement, c), (q,), (jnp.ones_like(q),)))(
            jnp.asarray(0.), coefficients)
    cotangent = jnp.array([.31, -.17, .23, .73])
    scalar = lambda q, c: jnp.vdot(cotangent, outputs(q, c))
    value, gradient = jax.jit(jax.value_and_grad(scalar, argnums=0))(
        jnp.asarray(0.), coefficients)
    _, backward = jax.vjp(lambda q: outputs(q, coefficients), jnp.asarray(0.))
    reverse = backward(cotangent)[0]
    np.testing.assert_allclose(value, jnp.vdot(cotangent, values), atol=2e-11, rtol=0.)
    np.testing.assert_allclose(gradient, jnp.vdot(cotangent, tangent), atol=3e-8, rtol=2e-6)
    np.testing.assert_allclose(reverse, gradient, atol=3e-8, rtol=2e-6)
    if xc == "b3lyp":
        np.testing.assert_allclose(jax.jit(lambda pb, weight: pb(weight)[0])(backward, cotangent),
                                   reverse, atol=3e-8, rtol=2e-6)
    assert np.isfinite(tangent).all()

    engine = NativeRHF(list(zip(symbols, np.asarray(coordinates))), "sto-3g", unit="Bohr", charge=charge)
    step = 2e-4
    endpoints, densities = [], []
    for sign in (1, -1):
        moving = coordinates+sign*step*direction
        displaced = _mean_field(xc, moving, symbols, charge)
        observed, info = evaluate(jnp.asarray(sign*step), jnp.asarray(displaced.mo_coeff))
        assert bool(info[0]) and np.isfinite(observed).all()
        # These STO-3G diatomics have only one singlet particle-hole pair, hence no second
        # candidate for the multistate assignment helper. Compare the signed
        # AO transition densities directly, independently of the dipole.
        left, right = np.asarray(anchor[1]), np.asarray(info[1])
        cross = np.asarray(engine.cross_overlap(np.asarray(coordinates), np.asarray(moving)))
        occupied, virtual = left[:, :1].T@cross@right[:, :1], left[:, 1:].T@cross@right[:, 1:]
        previous, current = np.asarray(anchor[2][0]), np.asarray(info[2][0])
        overlap = np.sum(previous*(occupied@current@virtual.T))/(
            np.linalg.norm(previous)*np.linalg.norm(current))
        assert abs(overlap) > .999
        signed = np.asarray(observed).copy()
        signed[1:] *= 1. if overlap > 0 else -1.
        endpoints.append(signed)
        densities.append(np.asarray(info[3]))
    finite = (endpoints[0]-endpoints[1])/(2*step)
    assert np.linalg.norm(densities[0]-densities[1]) > 1e-8
    record_property("gap_directional_error_ha_per_bohr", float(abs(tangent[0]-finite[0])))
    record_property("dipole_directional_error_e", float(np.max(abs(tangent[1:]-finite[1:]))))
    np.testing.assert_allclose(tangent[0], finite[0], atol=4e-6, rtol=4e-4)
    np.testing.assert_allclose(tangent[1:], finite[1:], atol=4e-6, rtol=4e-4)
    np.testing.assert_allclose(gradient, cotangent@finite, atol=5e-6, rtol=4e-4)
