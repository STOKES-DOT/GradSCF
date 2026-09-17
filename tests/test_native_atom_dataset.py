"""Native atomic references must retain measured energies and convergence."""

import builtins

import numpy as np
import pytest

from gradscf.data.graddft_dataset import (
    GradDFTGroundAtomRecord,
    build_graddft_ground_atom_molecule,
)


@pytest.mark.parametrize(
    "symbol,spin,reference_energy",
    [("H", 1, -0.466581849557275), ("He", 0, -2.807783957539974)],
)
def test_native_atom_reference_preserves_scf_energy_without_external_imports(
    monkeypatch, symbol, spin, reference_energy
):
    original_import = builtins.__import__

    def no_external_scf(name, *args, **kwargs):
        if name.split(".")[0] in {"pyscf", "gpu4pyscf"}:
            raise AssertionError(f"Unexpected external SCF import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_external_scf)
    record = GradDFTGroundAtomRecord(symbol, "train", -99.0, spin)
    molecule = build_graddft_ground_atom_molecule(record, basis="sto-3g")
    assert molecule.scf_converged is True
    assert molecule.scf_cycles > 0
    # Cartesian STO-3G, atom at origin, HF, native CPU integrals, float64; Hartree.
    np.testing.assert_allclose(molecule.mf_energy, reference_energy, atol=2e-9, rtol=0)
    assert molecule.mf_energy != record.target_energy_h
    np.testing.assert_allclose(
        np.einsum("spq,pq->", molecule.rdm1, molecule.overlap_matrix),
        {"H": 1, "He": 2}[symbol],
        atol=1e-12,
        rtol=0,
    )


def test_native_atom_reference_rejects_unconverged_result():
    record = GradDFTGroundAtomRecord("He", "train", -99.0, 0)
    with pytest.raises(RuntimeError, match="did not converge.*He"):
        build_graddft_ground_atom_molecule(record, basis="sto-3g", scf_max_cycle=1)


def test_native_atom_reference_rejects_external_builder():
    record = GradDFTGroundAtomRecord("H", "train", -99.0, 1)
    with pytest.raises(ValueError, match="reference_builder"):
        build_graddft_ground_atom_molecule(record, basis="sto-3g", reference_builder="pyscf")


def test_native_atom_chunked_hfx_matches_dense_without_storing_dense_nu():
    import jax
    import jax.numpy as jnp

    record = GradDFTGroundAtomRecord("H", "train", -99.0, 1)
    kwargs = dict(
        basis="sto-3g", compute_local_hfx_features=True,
        compute_local_hfx_aux=True, hfx_omega_values=(0.0, 0.4),
        hfx_chunk_size=128,
    )
    dense = build_graddft_ground_atom_molecule(record, hfx_nu_storage="dense", **kwargs)
    chunked = build_graddft_ground_atom_molecule(record, hfx_nu_storage="chunked", **kwargs)
    assert chunked.hfx_nu is None
    assert chunked.hfx_nu_api is not None
    np.testing.assert_allclose(chunked.hfx_local, dense.hfx_local, atol=1e-12, rtol=0)
    np.testing.assert_allclose(chunked.hfx_fxx, dense.hfx_fxx, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        chunked.hfx_nu_api.grid_chunk(0, 7), dense.hfx_nu[:, :7], atol=1e-12, rtol=0
    )
    ngrid = dense.hfx_nu.shape[1]
    padded = jax.jit(lambda start: chunked.hfx_nu_api.grid_chunk_padded(start, 7))(
        jnp.asarray(ngrid - 3, dtype=jnp.int32)
    )
    np.testing.assert_allclose(padded[:, :3], dense.hfx_nu[:, -3:], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(padded[:, 3:], 0.0)


def test_native_atom_training_target_is_separate_from_scf_energy():
    from gradscf.data.graddft_dataset import build_graddft_ground_atom_datum

    record = GradDFTGroundAtomRecord("He", "train", -99.0, 0)
    datum = build_graddft_ground_atom_datum(record, basis="sto-3g")
    assert float(datum.target_e0_total_h) == -99.0
    np.testing.assert_allclose(datum.molecule.mf_energy, -2.807783957539974, atol=2e-9, rtol=0)
