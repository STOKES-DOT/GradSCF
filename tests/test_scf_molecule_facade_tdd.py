"""Count molecule facade work while preserving its solved-state contracts."""
from collections import Counter
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, gto, integrals, scf
from gradscf.scf import autodiff, builders, facade, uks
from gradscf.scf.inputs import assembly
from gradscf.scf.molecules import UnrestrictedMolecule


def _molecule(unrestricted):
    atom = "H 0 0 0" if unrestricted else "H 0 0 0; H 0 0 .74"
    return gto.M(atom=atom, basis="sto-3g", spin=int(unrestricted))


def _count(monkeypatch, owner, name, calls):
    original = getattr(owner, name)

    def counted(*args, **kwargs):
        calls[name] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, counted)


@pytest.mark.parametrize("method,unrestricted", [(scf.RKS, False), (scf.UKS, True)])
def test_kernel_prepares_spec_config_and_orbital_signature_once(monkeypatch, method, unrestricted):
    mf = method(_molecule(unrestricted), xc="hf")
    calls = Counter()
    names = ("_spec", "_config", "_standard_xc_spec", "_orbital_signature")
    for name in names:
        _count(monkeypatch, mf, name, calls)

    assert np.isfinite(mf.kernel())
    assert mf.converged
    assert calls == Counter(dict.fromkeys(names, 1))


@pytest.mark.parametrize("method,unrestricted", [(scf.RKS, False), (scf.UKS, True)])
def test_cold_reference_checks_and_completes_once_without_resolving(monkeypatch, method, unrestricted):
    mf = method(_molecule(unrestricted), xc="hf").run()
    result = mf.scf_result
    source = mf._scf_inputs
    calls = Counter()
    names = ("_spec", "_config", "_check_reference_source", "_check_response_grid",
             "_orbital_signature")
    for name in names:
        _count(monkeypatch, mf, name, calls)
    complete = builders.complete_restricted_response_inputs

    def counted_complete(*args, **kwargs):
        calls["completion"] += 1
        return complete(*args, **kwargs)

    def reject(*args, **kwargs):
        pytest.fail("An existing SCF result must not trigger another solve.")

    monkeypatch.setattr(builders, "complete_restricted_response_inputs", counted_complete)
    monkeypatch.setattr(facade, "complete_restricted_response_inputs", counted_complete, raising=False)
    monkeypatch.setattr(builders, "run_rks_from_integrals", reject)
    monkeypatch.setattr(builders, "run_uks_from_integrals", reject)
    reference = mf.to_reference()
    assert calls == Counter(dict.fromkeys((*names, "completion"), 1))
    assert mf.scf_result is result
    assert mf._scf_inputs.overlap is source.overlap
    assert mf._scf_inputs.hcore is source.hcore
    assert reference.dipole_integrals is mf._scf_inputs.dipole_integrals
    assert reference.grid.weights is mf._scf_inputs.grid_weights
    assert reference.grid.weights.size > 0

    calls.clear()
    assert mf.to_reference() is reference
    assert calls == Counter(dict.fromkeys(names, 1))
    molecule = mf.mol
    changed_atom = "H 0 0 .1" if unrestricted else "H 0 0 0; H 0 0 .75"
    with pytest.raises(RuntimeError, match="kernel"):
        mf.mol = replace(molecule, atom=changed_atom)
        mf.to_reference()
    mf.mol = molecule
    mf.mo_coeff = np.array(mf.mo_coeff)
    mf.mo_coeff.flat[0] += .01
    with pytest.raises(RuntimeError, match="(?i)orbital"):
        mf.to_reference()


def test_uks_hf_kernel_defers_dipole_and_reference():
    mf = scf.UKS(_molecule(True), xc="hf").run()
    assert mf.scf_result is not None and mf.converged
    assert mf.reference is None
    assert mf._scf_inputs.dipole_integrals is None
    ref = mf.to_reference()
    assert ref.dipole_integrals.shape == (3, 1, 1)
    assert mf.reference is ref


@pytest.mark.parametrize("xc", ["pbe", "wb97xd"])
def test_uks_standard_dft_reference_is_completed_only_on_request(monkeypatch, xc):
    mol = _molecule(True)
    # Run once without optional jax_xc; this test observes facade dispatch.
    seed = scf.UKS(mol, xc="hf").run().scf_result
    rsh_coefficients = (.2, 1., .222036) if xc == "wb97xd" else None
    result = replace(seed, range_separated_coefficients=rsh_coefficients)
    mf = scf.UKS(mol, xc=xc)
    calls = Counter()
    _count(monkeypatch, mf, "_build_reference", calls)
    _count(monkeypatch, builders, "build_uks_integral_inputs", calls)

    def solve(**kwargs):
        calls["solve"] += 1
        assert kwargs["config"].xc_spec == xc
        return result

    monkeypatch.setattr(builders, "run_uks_from_integrals", solve)
    assert mf.kernel() == result.total_energy
    assert calls == Counter(solve=1, build_uks_integral_inputs=1)
    assert mf.scf_result is result
    assert mf._scf_inputs.dipole_integrals is None
    if xc == "wb97xd":
        with pytest.raises(NotImplementedError, match="(?i)range.separated"):
            mf.to_reference()
        assert calls["_build_reference"] == 0
    ref = mf.reference
    assert isinstance(ref, UnrestrictedMolecule)
    assert ref.range_separated_coefficients == rsh_coefficients
    assert ref.dipole_integrals.shape == (3, 1, 1)
    assert mf.reference is ref
    if xc != "wb97xd":
        assert mf.to_reference() is ref
    assert calls == Counter(solve=1, build_uks_integral_inputs=1, _build_reference=1)


def test_uks_reference_reuses_stacked_orbitals_and_make_rdm1_tracks_edits():
    mf = scf.UKS(_molecule(True), xc="hf").run()
    ref = mf.to_reference()
    assert all(getattr(ref, field) is getattr(mf, field)
               for field in ("mo_coeff", "mo_occ", "mo_energy"))
    np.testing.assert_allclose(mf.make_rdm1(), ref.rdm1, atol=1e-14)
    mf.mo_coeff = mf.mo_coeff * 2
    np.testing.assert_allclose(mf.make_rdm1(), 4 * ref.rdm1, atol=1e-14)


def test_rks_reference_does_not_stage_electronic_arrays_on_host(monkeypatch):
    mf = scf.RKS(_molecule(False), xc="hf", execution_device="cpu").run()
    result, inputs = mf.scf_result, mf._scf_inputs
    device_values = (result.mo_coeff, result.mo_occ, result.mo_energy,
                     result.density_matrix, inputs.overlap, inputs.hcore)
    device_get = jax.device_get
    copies = []

    def counted(value):
        if any(value is item for item in device_values):
            copies.append(value)
        return device_get(value)

    monkeypatch.setattr(jax, "device_get", counted)
    ref = mf.to_reference()
    assert not copies
    np.testing.assert_array_equal(ref.mo_coeff[0], result.mo_coeff)
    np.testing.assert_array_equal(ref.mo_coeff[1], result.mo_coeff)
    np.testing.assert_array_equal(ref.mo_occ.sum(axis=0), result.mo_occ)
    np.testing.assert_array_equal(ref.rdm1.sum(axis=0), result.density_matrix)
    assert ref.mo_coeff.devices() == {jax.devices("cpu")[0]}


def _external_model(unrestricted):
    functional = dft.Functional(lambda ingredients: ingredients.density_matrix,
                               lambda params, density: params["a"] * jnp.sum(density**2))
    method = scf.UKS if unrestricted else scf.RKS
    return method(_molecule(unrestricted), xc=functional, params={"a": .01}, grid_block_size=-1)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_external_final_xc_energy_and_potential_have_one_evaluation(monkeypatch, unrestricted):
    mf = _external_model(unrestricted)
    run = autodiff.DifferentiableSCF.run
    evaluate = type(mf.xc).energy_for_density
    build_jk = uks._spin_jk_matrices
    terminal_state = []
    terminal_evaluations = []
    terminal_jk_rebuilds = []

    def observed_run(*args, **kwargs):
        state, info = run(*args, **kwargs)
        terminal_state.append(state)
        return state, info

    def observed_evaluate(functional, params, molecule, density):
        if terminal_state and molecule is terminal_state[0]:
            terminal_evaluations.append(density)
        return evaluate(functional, params, molecule, density)

    def observed_build_jk(*args, **kwargs):
        if terminal_state:
            terminal_jk_rebuilds.append(kwargs.get("with_k"))
        return build_jk(*args, **kwargs)

    monkeypatch.setattr(autodiff.DifferentiableSCF, "run", observed_run)
    monkeypatch.setattr(type(mf.xc), "energy_for_density", observed_evaluate)
    monkeypatch.setattr(uks, "_spin_jk_matrices", observed_build_jk)
    mf.kernel()
    ref = mf.reference
    density = np.asarray(ref.rdm1)
    total_density = density.sum(axis=0)
    j_mat = np.asarray(integrals.build_j_from_eri_pair_matrix(ref.eri_pair_matrix, total_density))
    potential = .02 * density
    expect_fock = ref.h1e + j_mat + (potential if unrestricted else potential.mean(axis=0))
    expect_energy = (np.sum(total_density * ref.h1e) + .5 * np.sum(total_density * j_mat)
                     + .01 * np.sum(density**2) + ref.nuclear_repulsion)
    assert mf.converged
    np.testing.assert_allclose(ref.fock_matrix, expect_fock, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mf.e_tot, expect_energy, rtol=1e-12, atol=1e-12)
    assert len(terminal_evaluations) == 1
    assert not terminal_jk_rebuilds


@pytest.mark.parametrize("unrestricted", [False, True])
def test_external_dense_ao_uses_one_second_order_evaluation(monkeypatch, unrestricted):
    mf = _external_model(unrestricted)
    # Distinct geometries keep the cold-grid test independent of the XC tests above.
    atom = "H .123 0 0" if unrestricted else "He .123 0 0"
    mf.mol = replace(mf.mol, atom=atom)
    evaluated = []
    original = integrals.evaluate_cartesian_ao_with_derivatives

    def evaluate(basis, coords, *, deriv=1, **kwargs):
        evaluated.append((coords, deriv))
        return original(basis, coords, deriv=deriv, **kwargs)

    monkeypatch.setattr(assembly, "evaluate_cartesian_ao_with_derivatives", evaluate)
    monkeypatch.setattr(integrals, "evaluate_cartesian_ao_with_derivatives", evaluate)
    mf.kernel()
    ref = mf.reference
    assert ref.grid_ao is None and ref.ao_laplacian.shape == ref.ao.shape
    assert len(evaluated) == 1
    coords, deriv = evaluated[0]
    assert deriv == 2
    np.testing.assert_array_equal(coords, ref.grid.coords)
