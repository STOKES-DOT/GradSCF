"""Metadata-only lower bound; no large molecule, integral or matrix allocation."""
from types import SimpleNamespace
import pytest
from gradscf.integrals.backends.native.jk import native_jk_workspace_bytes


@pytest.mark.parametrize('workers',[1,64])
def test_direct_workspace_covers_at_least_one_canonical_pair_table(workers):
    shells=3000
    topology=SimpleNamespace(angular_momenta=(0,)*shells,primitive_counts=(1,)*shells,
        contraction_counts=(1,)*shells,nuclear_charges=(1,)*shells,cart=True)
    provider=SimpleNamespace(plan=SimpleNamespace(topology=topology),private_output_bytes=0)
    estimated=native_jk_workspace_bytes(provider,workers,0.,private_output_bytes=0)
    pairs=shells*(shells+1)//2
    # LP64 Pair{int,int,unsigned}=12 bytes plus two size_t indices=16.
    # This is only a logical retained-data minimum, not an allocator peak.
    # Sharing immutable metadata is allowed and preferred; no W-fold floor
    # is imposed because that would forbid fixing the current deep copies.
    logical_minimum=28*pairs
    assert estimated>=logical_minimum,(estimated,logical_minimum)
