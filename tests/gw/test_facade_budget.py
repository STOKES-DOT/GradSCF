"""Matrix GW receives E/C/B/h; the facade must reserve its remaining owners."""
import pytest


@pytest.mark.parametrize("auxiliary", [False, True])
def test_qsgw_keeps_mean_field_residents_without_unused_potential(monkeypatch, auxiliary):
    from gradscf import gto, scf, gw
    from gradscf.gw import integrals, qsgw
    from gradscf.gw.types import GWResult
    from gradscf._memory import tree_storage_bytes
    mf = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g"),
                   conv_tol=1e-12).density_fit("weigend").run()
    r, factors = mf.scf_result, mf._scf_inputs.df_factors
    owner = gw.GW(mf, method="qsgw", nw=4)
    if auxiliary:
        owner.density_fit("weigend")
        monkeypatch.setattr(integrals, "auxiliary_factors", lambda *args, **kwargs: factors)
    def forbidden(*args, **kwargs):
        raise AssertionError("Matrix self-consistency does not consume F-h-J from the starting point")
    monkeypatch.setattr(integrals, "starting_potential", forbidden)
    def solve(**kwargs):
        expected = tree_storage_bytes(mf._scf_inputs, r, mf.reference,
            exclude=(r.mo_energy, r.mo_coeff, factors, r.hcore_matrix))
        assert kwargs["reserved_bytes"] == expected
        return GWResult(r.mo_energy, r.mo_coeff, converged=True, method="qsgw",
                        screening_energy=r.mo_energy)
    # This is an orchestration/ownership test; numerical qsGW is tested independently.
    monkeypatch.setattr(qsgw, "qsgw_cd_restricted", solve)
    owner.run()
