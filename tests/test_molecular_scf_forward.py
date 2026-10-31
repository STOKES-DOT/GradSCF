"""Existing DIIS forward plus the same fully live implicit stationary response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('xc',['hf','pbe'])
def test_scf_forward_matches_minimum_state_and_complete_geometry_response(xc):
    from gradscf import gto,dft
    from gradscf_tools.molecular_response import MolecularEnergy
    mf=dft.RKS(gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis='sto-3g',cart=True),
        xc=xc,grids_level=1,conv_tol=1e-12,conv_tol_grad=1e-10).run()
    old=MolecularEnergy(mf,gradient_tolerance=1e-9)
    new=MolecularEnergy(mf,gradient_tolerance=1e-9,forward='scf')
    for r in (new.coordinates,new.coordinates.at[1,2].add(.01)):
        previous=old.solve(r);updated=new.solve(r)
        assert bool(previous.stationary) and bool(updated.stationary)
        np.testing.assert_allclose(updated.total_energy,previous.total_energy,atol=2e-10,rtol=0)
        np.testing.assert_allclose(updated.density_matrix,previous.density_matrix,atol=2e-7,rtol=0)
    r=new.coordinates
    gradient=jax.jit(jax.grad(new))(r)
    previous=jax.jit(jax.grad(old))(r)
    np.testing.assert_allclose(gradient,previous,atol=1e-6,rtol=1e-6)
    direction=jnp.zeros_like(r).at[1,2].set(1.)
    value=jax.jit(new)
    for step in (2e-4,1e-4):
        fd=(value(r+step*direction)-value(r-step*direction))/(2*step)
        np.testing.assert_allclose(jnp.vdot(gradient,direction),fd,atol=1e-5,rtol=1e-5)


def test_scf_seed_strategy_does_not_redefine_explicit_iteration_derivatives():
    from gradscf import gto,scf
    from gradscf_tools.molecular_response import MolecularEnergy
    mf=scf.RHF(gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g',cart=True)).run()
    with pytest.raises(ValueError,match='implicit'):
        MolecularEnergy(mf,mode='explicit',forward='scf')


def test_scf_seed_strategy_rejects_unrestricted_reference():
    from gradscf import gto,scf
    from gradscf_tools.molecular_response import MolecularEnergy
    mf=scf.UHF(gto.M(atom='H 0 0 0',basis='sto-3g',spin=1,cart=True)).run()
    with pytest.raises(ValueError,match='restricted'):
        MolecularEnergy(mf,forward='scf')


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_scf_forward_rejects_failed_numerical_producer(monkeypatch, compiled):
    from types import SimpleNamespace
    from gradscf import gto, scf
    from gradscf.scf import rks
    from gradscf_tools.molecular_response import MolecularEnergy

    mf = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", cart=True)).run()
    model = MolecularEnergy(mf, forward="scf")
    # The stationary coefficients alone must not override a failed producer's
    # energy/density convergence flag.
    failed = SimpleNamespace(mo_coeff=mf.mo_coeff, converged=jnp.asarray(False))
    monkeypatch.setattr(rks, "run_rks_from_integrals_traceable", lambda **kwargs: failed)
    evaluate = jax.jit(model) if compiled else model
    try:
        value = evaluate(model.coordinates)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        assert not np.isfinite(float(value))


@pytest.fixture(scope="module", params=["pbe0", "wb97xd"])
def hybrid_reference(request):
    from gradscf import dft, gto

    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True)
    mf = dft.RKS(mol, xc=request.param, grids_level=0, max_cycle=200,
                 conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10).run()
    assert mf.converged
    return mf


def test_direct_scf_forward_uses_only_diis_with_complete_geometry_response(
        hybrid_reference, monkeypatch):
    from gradscf.scf import orbital_optimization, rks
    from gradscf_tools.molecular_response import MolecularEnergy

    full = MolecularEnergy(hybrid_reference, gradient_tolerance=1e-9)
    direct = MolecularEnergy(hybrid_reference, forward="scf", jk_backend="direct",
                             gradient_tolerance=1e-9)
    coordinates = full.coordinates
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    # Break H2 parity so the occupied orbital must respond to the changing
    # geometry, beyond metric transport of symmetry-fixed zero-field orbitals.
    field = jnp.array([0., 0., .001])
    full_value = lambda r: full(r, field)
    full_calculate = jax.jit(jax.value_and_grad(full_value))
    full_state = jax.jit(lambda r: full.solve(r, field))(coordinates)
    expected_value, expected_gradient = full_calculate(coordinates)
    expected_hvp = jax.jit(lambda r: jax.jvp(
        jax.grad(full_value), (r,), (direction,))[1])(coordinates)
    assert bool(full_state.stationary)

    def forbidden(*args, **kwargs):
        raise AssertionError("forward='scf' must not run orbital L-BFGS or Newton polishing")

    original_diis = rks.run_rks_from_integrals_traceable
    diis_calls = []

    def diis(*args, **kwargs):
        diis_calls.append(True)
        return original_diis(*args, **kwargs)

    monkeypatch.setattr(orbital_optimization, "solve_orbitals", forbidden)
    monkeypatch.setattr(rks, "run_rks_from_integrals_traceable", diis)
    state = jax.jit(lambda r: direct.solve(r, field))(coordinates)
    assert diis_calls
    assert bool(state.stationary)
    assert int(state.iterations) == 0 and int(state.evaluations) == 1
    assert int(state.polishing_steps) == 0
    assert state.stage_history == () and state.polishing_history == ()
    np.testing.assert_allclose(state.total_energy, full_state.total_energy, atol=2e-10, rtol=0)
    np.testing.assert_allclose(state.density_matrix, full_state.density_matrix, atol=2e-7, rtol=0)

    value = lambda r: direct(r, field)
    calculate = jax.jit(jax.value_and_grad(value))
    observed_value, observed_gradient = calculate(coordinates)
    observed_hvp = jax.jit(lambda r: jax.jvp(
        jax.grad(value), (r,), (direction,))[1])(coordinates)
    for response in (observed_value, expected_gradient, observed_gradient, expected_hvp, observed_hvp):
        assert np.isfinite(np.asarray(response)).all()
    np.testing.assert_allclose(observed_value, expected_value, atol=2e-10, rtol=0)
    np.testing.assert_allclose(observed_gradient, expected_gradient, atol=3e-8, rtol=2e-7)
    np.testing.assert_allclose(observed_hvp, expected_hvp, atol=3e-7, rtol=2e-6)
    for step in (3e-4, 1e-4):
        plus = calculate(coordinates + step*direction)
        minus = calculate(coordinates - step*direction)
        np.testing.assert_allclose(jnp.sum(observed_gradient*direction),
            (plus[0]-minus[0])/(2*step), atol=3e-6, rtol=2e-5)
        np.testing.assert_allclose(observed_hvp, (plus[1]-minus[1])/(2*step),
                                   atol=4e-6, rtol=2e-5)


def _zero_iteration_state(parameter, mode):
    from gradscf import scf

    return scf.minimize_roks_from_integrals(
        overlap=jnp.eye(2), hcore=jnp.array([[-1., .2+parameter], [.2+parameter, .5]]),
        eri=jnp.zeros((2, 2, 2, 2)), nuclear_repulsion=0.,
        ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)), grid_weights=jnp.zeros(0),
        mo_coeff=jnp.eye(2), mo_occ=np.array([[1., 0.], [1., 0.]]), xc_spec="hf",
        max_iterations=0, gradient_tolerance=1e-9, gradient_mode=mode)


def test_zero_iteration_implicit_state_rejects_nonstationary_response(monkeypatch):
    from gradscf.scf import orbital_optimization

    def forbidden(*args, **kwargs):
        raise AssertionError("Zero-iteration stationarity validation must not optimize the seed")

    monkeypatch.setattr(orbital_optimization, "solve_orbitals", forbidden)
    state = jax.jit(lambda t: _zero_iteration_state(t, "implicit"))(jnp.asarray(0.))
    assert not bool(state.stationary)
    assert float(state.gradient_norm) > 1e-9
    assert int(state.iterations) == 0 and int(state.polishing_steps) == 0
    derivative = jax.jit(jax.grad(lambda t: _zero_iteration_state(t, "implicit").total_energy))(
        jnp.asarray(0.))
    assert np.isnan(np.asarray(derivative)).all()


def test_zero_iterations_remain_invalid_for_explicit_response():
    with pytest.raises(ValueError, match="(?i)(implicit|iteration)"):
        _zero_iteration_state(jnp.asarray(0.), "explicit")
