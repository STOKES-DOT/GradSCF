"""Converged molecular tau-MGGA through actual call-local AO caching.

Fixed before execution: Cartesian def2-SVP, grid3, native direct J/K with two
workers, and 128 decimal MB. Grid3's unpruned upper bounds are 52,850 points
for water and 37,750 for OH. At 25/20 AOs, one AO+gradient cache is <=42.28/
24.16 MB. The native metadata planner reserves <=10.40/10.19 MB for its three
host workspaces. Two cache copies, 4 MB other SCF storage and one-point AO
scratch fit the 102.4 MB usable budget, even at those unpruned upper bounds.

The full primitive-AO working set exceeds that budget, so automatic assembly
must start with real AOGrid blocks. Runtime type assertions below enforce both
sides of this preflight; the budget or numerical gates must not be relaxed
after a failure. No local scientific execution is permitted for these tests.
"""
import numpy as np
import pytest

from gradscf import dft, gto
from gradscf.integrals.grids.blocks import AOGrid, _CachedAOGrid
from gradscf.scf.inputs import grid
from test_hfks_acceptance import _assert_stationary


def _molecule(unrestricted):
    return gto.M(
        atom=('O 0 0 0; H .1 0 .97' if unrestricted else
              'O 0 0 0; H 0 .76 .58; H 0 -.76 .58'),
        basis='def2-svp', unit='Angstrom', cart=True, charge=0,
        spin=1 if unrestricted else 0)


def _solve(spec, unrestricted, block_size):
    cls = dft.UKS if unrestricted else dft.RKS
    return cls(_molecule(unrestricted), xc=spec, grids_level=3,
        grid_block_size=block_size, max_memory=128., jk_backend='direct',
        direct_scf_workers=2, direct_scf_tol=1e-12, init_guess='hcore',
        max_cycle=200, conv_tol=1e-11, conv_tol_density=1e-9,
        conv_tol_grad=1e-8).run()


def _fock(mf):
    # RKS stores the solved result; traditional UKS stores a solved reference.
    state = mf.scf_result if mf.scf_result is not None else mf.reference
    return np.asarray(state.fock_matrix)


@pytest.mark.parametrize('unrestricted', [False, True], ids=['water-rks', 'oh-uks'])
@pytest.mark.parametrize('spec', ['r2scan', 'm06-2x'])
def test_converged_mgga_actual_cache_matches_explicit_streaming(
        monkeypatch, spec, unrestricted):
    # Avoid unrelated previous tests' resident grid buffers changing this
    # fixture's documented budget. The raw production provider remains real.
    monkeypatch.setattr(grid, '_GRID_AO_INPUT_CACHE', {})
    original = grid.prepare_scf_grid
    observations = []

    def observe(raw, **controls):
        prepared = original(raw, **controls)
        observations.append((raw, prepared, controls))
        return prepared

    monkeypatch.setattr(grid, 'prepare_scf_grid', observe)
    cached = _solve(spec, unrestricted, 0)
    cached_residual, cached_cycles = _assert_stationary(cached)
    assert len(observations) == 1
    raw, prepared, controls = observations[0]
    assert type(raw) is AOGrid and raw.adaptive
    assert type(prepared) is _CachedAOGrid, 'Fixed 128-MB fixture must exercise actual AO reuse'
    assert controls['deriv'] == 1 and prepared.values.shape[1] == 4
    assert raw.nao == (20 if unrestricted else 25)
    assert raw.ngrids > raw.block_size > 0
    assert prepared.ngrids == raw.ngrids
    assert prepared.block_size <= raw.block_size
    assert controls['host_workspace_bytes'] > 0

    observations.clear()
    streamed = _solve(spec, unrestricted, raw.block_size)
    streamed_residual, streamed_cycles = _assert_stationary(streamed)
    assert len(observations) == 1
    stream_raw, stream_prepared, stream_controls = observations[0]
    assert type(stream_raw) is AOGrid and not stream_raw.adaptive
    assert stream_prepared is stream_raw, 'Explicit block size must retain streaming'
    assert stream_raw.block_size == raw.block_size
    assert stream_controls['deriv'] == 1
    np.testing.assert_array_equal(stream_raw.coords, raw.coords)
    np.testing.assert_array_equal(stream_raw.weights, raw.weights)

    # Molecular gates match the existing aligned-grid MGGA reference tests:
    # 2e-8 Ha energy (stricter than the plan's 1e-7), 2e-7 density/Fock, rtol=0.
    np.testing.assert_allclose(cached.e_tot, streamed.e_tot, atol=2e-8, rtol=0)
    np.testing.assert_allclose(cached.make_rdm1(), streamed.make_rdm1(), atol=2e-7, rtol=0)
    np.testing.assert_allclose(_fock(cached), _fock(streamed), atol=2e-7, rtol=0)
    print(dict(xc=spec, method='UKS' if unrestricted else 'RKS', max_memory_mb=128.,
        nao=raw.nao, ngrids=raw.ngrids, raw_block_size=raw.block_size,
        cached_block_size=prepared.block_size, cache_bytes=int(prepared.values.nbytes),
        cached_cycles=cached_cycles, streamed_cycles=streamed_cycles,
        cached_orbital_residual_hartree=cached_residual,
        streamed_orbital_residual_hartree=streamed_residual,
        energy_delta_hartree=float(cached.e_tot-streamed.e_tot),
        density_maxabs=float(np.max(np.abs(np.asarray(cached.make_rdm1())
                                          - np.asarray(streamed.make_rdm1())))),
        fock_maxabs=float(np.max(np.abs(_fock(cached)-_fock(streamed))))), flush=True)
