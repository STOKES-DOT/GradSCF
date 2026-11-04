"""Resource controls reach every HF/KS member of the shared solver family."""
import pytest


@pytest.mark.parametrize("method", ["RHF", "RKS", "UHF", "UKS", "GHF", "GKS"])
def test_six_facades_propagate_memory_controls(method):
    from gradscf import gto, scf
    mol = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g")
    mf = getattr(scf, method)(mol, max_memory=1234., grid_block_size=37,
                              grid_max_memory=234.)
    config = mf._config()
    assert config.max_memory == 1234.
    assert config.grid_block_size == 37
    assert config.grid_max_memory == 234.
