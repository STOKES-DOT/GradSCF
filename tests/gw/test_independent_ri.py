"""GW may request an auxiliary basis independently of the SCF representation."""
import jax
import numpy as np
import pytest


def test_gw_ri_after_direct_hf_keeps_the_original_starting_potential():
    from gradscf import gto, scf, gw
    assert hasattr(gw.GW, "density_fit"), "GW needs independent explicit auxiliary RI"
    mol = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g")
    mf = scf.RHF(mol, conv_tol=1e-12).direct_scf().run()
    assert mf.converged and mf._scf_inputs.df_factors is None
    assert mf._scf_inputs.eri_pair_matrix is None and mf._scf_inputs.eri is None
    reference_energy = np.array(mf.mo_energy)
    obj = gw.GW(mf, nw=24).density_fit("weigend").run()
    jax.block_until_ready(obj.result)
    assert obj.converged
    assert obj._ao_factors.shape[1:] == (2, 2)
    np.testing.assert_array_equal(mf.mo_energy, reference_energy)
    assert mf._scf_inputs.df_factors is None
    assert mf._scf_inputs.eri_pair_matrix is None
    from gradscf.integrals.backends.native.jk import _selected_jk
    from gradscf.gw import g0w0_cd_restricted
    density = mf.scf_result.density_matrix
    _, exchange = _selected_jk(mf._scf_inputs.direct_basis, density, with_j=False)
    reference = g0w0_cd_restricted(mo_energy=mf.mo_energy, mo_coeff=mf.mo_coeff,
        nocc=1, df_factors=obj._ao_factors, fock_matrix=mf.scf_result.fock_matrix,
        hcore_matrix=mf.scf_result.hcore_matrix, density_matrix=density,
        nw=24, v_mf_matrix=-.5*exchange)
    np.testing.assert_allclose(obj.mo_energy, reference.mo_energy, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_ri_rebinding_releases_old_gw_cache_owners_before_preparation(monkeypatch, unrestricted):
    from types import SimpleNamespace
    from gradscf import gw
    from gradscf.gw import integrals
    owner = object.__new__(gw.UGW if unrestricted else gw.GW)
    owner._scf = SimpleNamespace(_scf_signature=lambda: "new-source")
    owner.auxbasis, owner.max_memory = "weigend", 100.
    owner._ri_key = ("old-source", "weigend")
    owner._ri_factors = owner._ao_factors = np.ones((2, 3, 3))
    def prepare(*args, **kwargs):
        assert owner._ri_factors is owner._ao_factors is None
        return np.zeros((2, 3, 3))
    monkeypatch.setattr(integrals, "auxiliary_factors", prepare)
    result = owner._df_factors(None) if unrestricted else owner._df_factors()
    assert result is owner._ri_factors and owner._ri_key == ("new-source", "weigend")


def test_ri_destination_capacity_is_checked_before_native_preparation():
    from types import SimpleNamespace
    from gradscf import gto
    from gradscf.gw.integrals import auxiliary_factors
    class TinyGPU:
        platform = "gpu"
        def memory_stats(self):
            return {"bytes_limit": 128, "bytes_in_use": 0}
    mol = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g")
    source = SimpleNamespace(mol=mol, _spec=mol.to_spec,
                             mo_energy=SimpleNamespace(device=TinyGPU()))
    with pytest.raises(ValueError, match="destination-device memory"):
        auxiliary_factors(source, "weigend", max_memory=100.)
