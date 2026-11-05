"""Reliable SCF rebasing and consumer state reuse; tiny arrays, no native calls.

Artificial incremental drift is deliberate. Chemical/end-to-end acceptance is
separate; these fixtures isolate original gates and E/F/aux ownership.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers.scf import SCFConfig, SCFState, orbital_gradient_norm, solve_scf
from gradscf.solvers.scf.types import SCFHistory, SCFMetrics, SCFResult


def _layout(layout):
    if layout == 'unrestricted':
        coefficients = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
        occupations = jnp.array([[1., 0.], [1., 0.]])
    elif layout == 'spinor':
        coefficients = jnp.eye(4, dtype=jnp.complex128)
        c = jnp.sqrt(.5)
        coefficients = coefficients.at[0, 0].set(c).at[2, 2].set(c)
        coefficients = coefficients.at[0, 2].set(1j*c).at[2, 0].set(1j*c)
        occupations = jnp.array([1., 1., 0., 0.])
    else:
        coefficients, occupations = jnp.eye(2), jnp.array([2., 0.])
    density = (coefficients*occupations[..., None, :]) @ coefficients.conj().swapaxes(-1, -2)
    return density, coefficients, occupations


def _drifting_problem(parameter=.7, *, layout='restricted', refresh=True, conv_tol=1e-10,
                      reject_fresh=False, events=None):
    density, coefficients, occupations = _layout(layout)
    parameter = jnp.asarray(parameter)
    zero_fock = jnp.zeros_like(density)
    energies = jnp.zeros_like(occupations)
    initial = SCFState(density, coefficients, energies, parameter**2, zero_fock, jnp.asarray(0))
    stale_fock = jnp.broadcast_to(3*jnp.eye(density.shape[-1]), density.shape).astype(density.dtype)
    fresh_fock = jnp.full_like(density, .2) if reject_fresh else zero_fock

    def evaluate(dm, coeff, eps, previous):
        if previous is None:
            if events is not None:
                jax.debug.callback(lambda: events.append(('fresh',)), ordered=True)
            return parameter**2, fresh_fock, jnp.asarray(0)
        if events is not None:
            before = jnp.stack((previous.energy, jnp.max(previous.fock.real), previous.aux))
            jax.debug.callback(lambda x: events.append(('advance', *map(float, np.asarray(x)))), before, ordered=True)
        return previous.energy-2e-9, stale_fock, previous.aux+1

    result = solve_scf(initial, evaluate=evaluate,
        diagonalize=lambda f: (energies, coefficients), density_from_coeff=lambda c: density,
        diis_error=lambda f, state: jnp.zeros_like(f),
        orbital_gradient=lambda state: orbital_gradient_norm(state.fock, state.mo_coeff, occupations),
        config=SCFConfig(max_cycle=4, conv_tol=conv_tol, conv_tol_density=1e-8,
            conv_tol_grad=1e-7, diis_start_cycle=100, diis_space=1, refresh_on_convergence=refresh))
    return result


@pytest.mark.parametrize('layout', ['restricted', 'unrestricted', 'spinor'])
def test_residual_ready_rebases_drifting_energy_and_stale_best_state(layout):
    events = []
    result = _drifting_problem(layout=layout, events=events)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert bool(result.converged), 'D/g pass but incremental dE drift must not block a fresh check'
    assert int(result.cycles) == int(result.advance_evaluations) == 3
    assert int(result.fresh_evaluations) == 2
    assert [event[0] for event in events] == ['advance', 'advance', 'fresh', 'advance', 'fresh']
    # The next advance must see fresh E/F/aux, not any member of the stale tuple.
    np.testing.assert_allclose(events[3][1:], [.49, 0., 0.], atol=1e-15, rtol=0)
    for state in (result.state, result.best_state):
        np.testing.assert_allclose(state.energy, .49, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(state.fock, jnp.zeros_like(state.fock))
        assert int(state.aux) == 0, 'A tied-density best state must not retain stale JK/energy'
    assert float(result.metrics.energy_delta) < 1e-10
    assert float(result.metrics.density_rms) < 1e-8
    assert float(result.metrics.gradient) < 1e-7


def test_fresh_gradient_failure_cannot_be_hidden_by_incremental_residuals():
    events = []
    result = _drifting_problem(reject_fresh=True, events=events)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert not bool(result.converged) and int(result.cycles) == 4
    assert sum(event[0] == 'fresh' for event in events) == 3
    assert float(result.metrics.gradient) > 1e-7


@pytest.mark.parametrize('refresh,conv_tol', [(False, 1e-10), (True, 0.)])
def test_disabled_refresh_and_fixed_cycle_contract_do_not_add_fresh_evaluations(refresh, conv_tol):
    events = []
    result = _drifting_problem(refresh=refresh, conv_tol=conv_tol, events=events)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert not bool(result.converged) and int(result.cycles) == 4
    assert [event[0] for event in events] == ['advance']*4


def test_reliable_rebase_keeps_explicit_value_jvp_vjp_and_second_response():
    energy = lambda parameter: _drifting_problem(parameter).state.energy
    point, direction = .7, .13
    np.testing.assert_allclose(energy(point), point**2, atol=1e-14, rtol=0)
    tangent = jax.jvp(energy, (point,), (direction,))[1]
    _, back = jax.vjp(energy, point)
    np.testing.assert_allclose(tangent, 2*point*direction, atol=2e-13, rtol=0)
    np.testing.assert_allclose(back(1.)[0], 2*point, atol=2e-13, rtol=0)
    np.testing.assert_allclose(jax.jvp(jax.grad(energy), (point,), (direction,))[1], 2*direction, atol=2e-12, rtol=0)


@pytest.mark.parametrize('xc', ['hf', 'pbe'])
@pytest.mark.parametrize('selected_fresh', [False, True])
def test_unrestricted_reuses_fresh_selected_energy_exc_but_rechecks_stale_best(monkeypatch, xc, selected_fresh):
    from gradscf.scf import uks
    calls = []
    def raw(**kwargs):
        calls.append(kwargs)
        index = len(calls)
        return jnp.asarray(20.+index), jnp.asarray(.01*index), kwargs['h'], kwargs['h']
    def jk(eri, da, db, **kwargs):
        zero = jnp.zeros_like(da)
        return da+db, zero, zero, zero
    def selected(**kwargs):
        dm, coeff, eps = (kwargs[key] for key in ('density_spin', 'mo_coeff_spin', 'mo_energy_spin'))
        prior = SCFState(dm, coeff, eps, jnp.asarray(0.), jnp.zeros_like(dm), kwargs['initial_aux'])
        energy, fock, aux = kwargs['state_evaluator'](dm, coeff, eps, None if selected_fresh else prior)
        state = SCFState(dm, coeff, eps, energy, fock, aux)
        legacy = (dm, coeff, eps, fock, jnp.asarray(selected_fresh), jnp.asarray(2),
                  jnp.zeros(2), jnp.asarray(2), jnp.asarray(2), jnp.asarray(0.), jnp.asarray(0.))
        return (*legacy, state) if kwargs.get('return_state', False) else legacy
    monkeypatch.setattr(uks, '_raw_fock_and_energy_for_state', raw)
    monkeypatch.setattr(uks, '_spin_jk_matrices', jk)
    monkeypatch.setattr(uks, 'run_unrestricted_scf', selected)
    result = uks.run_uks_from_integrals(overlap=jnp.eye(2), hcore=jnp.diag(jnp.array([-1., .5])),
        eri=jnp.zeros((0, 0, 0, 0)), nalpha=1, nbeta=1, nuclear_repulsion=0.,
        ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)), grid_weights=jnp.zeros(0),
        direct_basis=object(), config=uks.UKSConfig(xc_spec=xc, jk_backend='direct', max_cycle=2))
    expected_calls = 2 if selected_fresh else 3
    assert len(calls) == expected_calls, 'Fresh selected state must not be evaluated a second time'
    np.testing.assert_allclose(result.total_energy, 20.+expected_calls, atol=0, rtol=0)
    np.testing.assert_allclose(result.xc_energy, .01*expected_calls, atol=1e-15, rtol=0)
    assert bool(result.converged) == selected_fresh


def _native_double(monkeypatch):
    from gradscf.integrals.backends.native import jk
    factors = jnp.array([[[.7, .2], [.2, .4]], [[.1, -.3], [-.3, .5]]])
    eri = jnp.einsum('Qpq,Qrs->pqrs', factors, factors)
    calls = []
    def selected(provider, density, *, with_j=True, with_k=True, omega=0., screen_density=None, **kwargs):
        calls.append(dict(density=density, with_j=with_j, with_k=with_k, omega=omega, screen_density=screen_density))
        kernel = eri*(.3 if omega else 1.)
        j = jnp.einsum('pqrs,...sr->...pq', kernel, density)
        k = jnp.einsum('prqs,...rs->...pq', kernel, density)
        return j if with_j else jnp.zeros_like(density), k if with_k else jnp.zeros_like(density)
    class Native:
        def get_jk(self, density, **kwargs): return selected(self, density, **kwargs)
    monkeypatch.setattr(jk, 'NativeDirectBasis', Native)
    monkeypatch.setattr(jk, '_selected_jk', selected)
    return Native(), factors, eri, calls


@pytest.mark.parametrize('xc', ['hf', 'pbe0', 'cam-b3lyp'])
@pytest.mark.parametrize('only_dfj', [False, True])
def test_restricted_native_advance_screens_delta_and_fresh_rebuild_clears_screen(monkeypatch, xc, only_dfj):
    from gradscf.scf.rks import RKSConfig, _make_jk_builder
    from gradscf.scf.rsh import rsh_preset
    provider, factors, eri, calls = _native_double(monkeypatch)
    cfg = RKSConfig(xc_spec=xc, jk_backend='df' if only_dfj else 'direct', only_dfj=only_dfj,
                    auxbasis='weigend' if only_dfj else None, direct_scf_tol=1e-12)
    builder = _make_jk_builder(None, cfg, direct_basis=provider, df_factors=factors if only_dfj else None)
    before = jnp.array([[1., .1], [.1, .5]])
    after = before+jnp.array([[.01, .02], [.02, -.01]])
    j0, k0 = builder(before)
    calls.clear()
    incremental = builder(after, None, None, before, j0, k0)
    advance = list(calls)
    assert advance
    for call in advance:
        assert call['screen_density'] is not None
        np.testing.assert_allclose(call['screen_density'], after-before, atol=0, rtol=0)
        np.testing.assert_allclose(call['density'], after-before, atol=0, rtol=0)
    calls.clear()
    fresh = builder(after)
    assert calls and all(call['screen_density'] is None for call in calls)
    for got, expected in zip(incremental, fresh):
        np.testing.assert_allclose(got, expected, atol=2e-13, rtol=2e-13)
    if only_dfj:
        assert all(not call['with_j'] for call in calls)
    preset = rsh_preset(xc)
    if preset is not None:
        assert {call['omega'] for call in advance} == {0., preset.default_omega}


@pytest.mark.parametrize('xc', ['hf', 'pbe0', 'cam-b3lyp'])
def test_spinor_scf_uses_one_four_block_native_batch_per_operator_and_fresh_basis_mask(monkeypatch, xc):
    from gradscf.scf import gks
    from gradscf.scf.rsh import rsh_preset
    from gradscf.dft.xc import hybrid_coeff
    provider, _, eri, calls = _native_double(monkeypatch)
    density, _, _ = _layout('spinor')
    h = jnp.diag(jnp.array([-1., .5]))
    h2 = jnp.kron(jnp.eye(2), h)
    def xc_terms(dm, *args, **kwargs): return jnp.asarray(.07), jnp.zeros_like(dm)
    monkeypatch.setattr(gks, '_generalized_xc', xc_terms)
    def solve(initial, **kwargs):
        calls.clear()
        delta = jnp.diag(jnp.array([.01, -.01, -.01, .01])).astype(initial.density.dtype)
        delta = delta.at[0, 2].set(.003j).at[2, 0].set(-.003j)
        current = initial.density+delta
        energy, fock, aux = kwargs['evaluate'](current, initial.mo_coeff, initial.mo_energy, initial)
        advance = list(calls)
        calls.clear()
        ef, ff, af = kwargs['evaluate'](current, initial.mo_coeff, initial.mo_energy, None)
        fresh = list(calls)
        preset = rsh_preset(xc)
        assert kwargs['config'].refresh_on_convergence
        assert len(advance) == (2 if preset is not None else 1), 'One traversal must carry all four spin J/K blocks'
        blocks = (current-initial.density).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 2, 2)
        for call in advance:
            np.testing.assert_allclose(call['density'].reshape(4, 2, 2), blocks, atol=0, rtol=0)
            assert call['screen_density'] is not None
            np.testing.assert_allclose(call['screen_density'].reshape(4, 2, 2), blocks, atol=0, rtol=0)
        assert all(call['screen_density'] is None for call in fresh)
        np.testing.assert_allclose(energy, ef, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(fock, ff, atol=2e-12, rtol=2e-12)
        if preset is not None:
            assert {call['omega'] for call in advance} == {0., preset.default_omega}
        full_blocks = current.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)
        spatial_j = jnp.einsum('pqrs,sr->pq', eri, current[:2, :2]+current[2:, 2:])
        expected_j = jnp.kron(jnp.eye(2), spatial_j)
        expected_k = jnp.einsum('prqs,abrs->abpq', eri, full_blocks).transpose(0, 2, 1, 3).reshape(4, 4)
        weight = hybrid_coeff(xc) if preset is None else preset.default_sr_hf_fraction+.3*(preset.default_lr_hf_fraction-preset.default_sr_hf_fraction)
        expected_f = h2+expected_j-weight*expected_k
        np.testing.assert_allclose(ff, expected_f, atol=3e-12, rtol=3e-12)
        np.testing.assert_allclose(ef, jnp.einsum('pq,qp->', current, h2+.5*(expected_j-weight*expected_k)).real+.07+.2, atol=3e-12, rtol=3e-12)
        assert np.linalg.norm(np.asarray(expected_k[:2, 2:]).imag) > 1e-4
        state = SCFState(current, initial.mo_coeff, initial.mo_energy, ef, ff, af)
        history = SCFHistory(jnp.zeros(4), jnp.zeros(4), jnp.zeros(4), jnp.ones(4, bool), ())
        return SCFResult(state, state, jnp.asarray(1), jnp.asarray(True), jnp.asarray(1), SCFMetrics(jnp.asarray(0.), jnp.asarray(0.), jnp.asarray(0.)), history)
    monkeypatch.setattr(gks, 'solve_scf', solve)
    result = gks.run_gks_from_integrals(overlap=jnp.eye(2), hcore=h, eri=None, nelectron=2,
        nuclear_repulsion=.2, ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)), grid_weights=jnp.zeros(0),
        init_density=density, direct_basis=provider,
        config=gks.GKSConfig(xc_spec=xc, jk_backend='direct', direct_scf_tol=1e-12, max_cycle=4))
    np.testing.assert_allclose(result.xc_energy, .07, atol=1e-15, rtol=0)


@pytest.mark.parametrize('selection', ['final', 'best_rms', 'first_converged'])
def test_optional_selected_state_keeps_legacy_tuple_and_selected_energy_aux(monkeypatch, selection):
    from gradscf.scf import iteration
    def state(value):
        matrix = jnp.full((2, 1, 1), value)
        return SCFState(matrix, jnp.ones_like(matrix), jnp.zeros((2, 1)),
                        jnp.asarray(value+10.), matrix, (jnp.asarray(value*.1),))
    final, best = state(3.), state(2.)
    rms = jnp.array([.3, .1, .2])
    history = SCFHistory(rms, jnp.ones(3), jnp.ones(3), jnp.zeros(3, bool), ())
    result = SCFResult(final, best, jnp.asarray(2), jnp.asarray(False), jnp.asarray(3),
                      SCFMetrics(jnp.asarray(1.), rms[-1], jnp.asarray(1.)), history)
    monkeypatch.setattr(iteration, 'solve_scf', lambda *args, **kwargs: result)
    kwargs = dict(fock_builder=lambda dm, *_: (jnp.zeros_like(dm), jnp.zeros_like(dm), jnp.asarray(0.)),
        density_spin=jnp.ones((2, 1, 1)), mo_coeff_spin=jnp.ones((2, 1, 1)),
        mo_occ_spin=jnp.ones((2, 1)), mo_energy_spin=jnp.zeros((2, 1)), overlap=jnp.eye(1),
        max_cycle=3, damping=0., conv_tol=0., conv_tol_density=1e-8, conv_tol_grad=1e-7,
        orthogonalization_eps=1e-10, iterate_selection=selection)
    legacy = iteration.run_unrestricted_scf(**kwargs)
    assert len(legacy) == 11
    expanded = iteration.run_unrestricted_scf(**kwargs, return_state=True)
    assert len(expanded) == 12
    expected = final if selection == 'final' else best
    for value, reference in zip(expanded[:-1], legacy):
        np.testing.assert_array_equal(value, reference)
    selected = expanded[-1]
    np.testing.assert_array_equal(selected.density, expected.density)
    np.testing.assert_array_equal(selected.fock, expected.fock)
    np.testing.assert_array_equal(selected.energy, expected.energy)
    np.testing.assert_array_equal(selected.aux[0], expected.aux[0])
