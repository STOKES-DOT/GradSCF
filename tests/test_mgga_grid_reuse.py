"""Tau-MGGA SCF must reuse affordable call-local AO/first-derivative data.

Real Gaussian AO blocks and the installed r2SCAN/M06-2X implementations are
used with tiny synthetic two-orbital Hamiltonian/ERI inputs. These fixed two-
iteration calculations test the shared consumer, not molecular convergence
or production timing. All numerical execution belongs to remote Slurm.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.grids.blocks import AOGrid
from gradscf.scf import rks, uks
from test_scf_grid_execution_budget import observe_ao, quadrature


def _solve(provider, spec, unrestricted):
    factor = jnp.asarray([[.3, .05], [.05, .2]], dtype=jnp.float64)
    common = dict(overlap=jnp.eye(2),
        hcore=jnp.asarray([[-1., .05], [.05, .6]]),
        eri=jnp.einsum('pq,rs->pqrs', factor, factor), nuclear_repulsion=0.,
        ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)),
        grid_weights=provider.weights, grid_ao=provider)
    controls = dict(xc_spec=spec, max_cycle=2, conv_tol=0., max_memory=16.,
                    jk_backend='full')
    if unrestricted:
        return uks.run_uks_from_integrals(**common, nalpha=1, nbeta=1,
            init_density_alpha=jnp.asarray([[.75, .08], [.08, .25]]),
            init_density_beta=jnp.asarray([[.45, -.03], [-.03, .55]]),
            config=uks.UKSConfig(**controls))
    return rks.run_rks_from_integrals(**common, nelectron=2,
        init_density=jnp.asarray([[1.2, .05], [.05, .8]]),
        config=rks.RKSConfig(**controls))


def _observables(result, unrestricted):
    if unrestricted:
        fock = jnp.stack([result.fock_matrix_alpha, result.fock_matrix_beta])
        density = jnp.stack([result.density_matrix_alpha, result.density_matrix_beta])
    else:
        fock, density = result.fock_matrix, result.density_matrix
    return result.total_energy, fock, density


@pytest.mark.parametrize('unrestricted', [False, True], ids=['rks', 'uks'])
@pytest.mark.parametrize('spec', ['r2scan', 'm06-2x'])
def test_tau_mgga_fitting_budget_produces_ao_once_and_matches_streaming(
        monkeypatch, spec, unrestricted):
    # Reuse only the real quadrature and runtime observer. The helper module's
    # controlled_grid fixture replaces XC with a toy formula and is NOT used.
    calls, _ = observe_ao(monkeypatch)
    basis, coords, weights = quadrature(jnp.zeros(5, dtype=jnp.float64))
    raw = AOGrid.from_basis(basis, coords, weights, block_size=3, adaptive=True)
    assert raw.ngrids == 17 and raw.block_size == 3
    identities = tuple(id(leaf) for leaf in jax.tree.leaves(raw))

    streamed = _solve(replace(raw, adaptive=False), spec, unrestricted)
    jax.block_until_ready(streamed)
    jax.effects_barrier()
    streamed_calls = tuple(calls)
    calls.clear()

    cached = _solve(raw, spec, unrestricted)
    jax.block_until_ready(cached)
    jax.effects_barrier()
    cached_calls = tuple(calls)

    # Check the numerical contract BEFORE the performance RED assertion. A
    # missing MGGA cache should fail only the observed AO-production count.
    assert int(cached.cycles) == int(streamed.cycles) == 2
    assert not bool(cached.converged) and not bool(streamed.converged)
    assert float(cached.exact_exchange_fraction) == pytest.approx(
        .54 if spec == 'm06-2x' else 0.)
    for actual, expected in zip(_observables(cached, unrestricted),
                               _observables(streamed, unrestricted), strict=True):
        assert np.isfinite(np.asarray(actual)).all()
        assert np.isfinite(np.asarray(expected)).all()
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert tuple(id(leaf) for leaf in jax.tree.leaves(raw)) == identities
    assert sum(streamed_calls) > raw.ngrids, 'The explicit-streaming control must recompute AO'
    print(dict(xc=spec, method='UKS' if unrestricted else 'RKS', cycles=2,
        streamed_ao_points=sum(streamed_calls), adaptive_ao_points=sum(cached_calls),
        streamed_ao_tiles=streamed_calls, adaptive_ao_tiles=cached_calls), flush=True)
    # Materialization uses five full tiles and an exact two-point tail. No
    # padded full-grid evaluation, repeated AO pass, or trace-time counter.
    assert sorted(cached_calls) == [2, 3, 3, 3, 3, 3], (
        'Affordable tau-MGGA must produce AO/first derivatives once before '
        'initial, iterative and final density-dependent XC evaluations', cached_calls)
